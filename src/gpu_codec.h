// zero: the GPU codec: runs kernels/zero.cl through the Level Zero runtime (gpu.h) on batches of blocks.
#pragma once
#include "archive.h"
#include "gpu.h"
#include "kernels_gen.h"
#include <string>
#include <vector>

namespace zero {

struct GpuCodec : Codec {
  static constexpr uint32_t G = 256;  // work-group size (the kernels are built for it)
  gpu::Device dev;
  bool opened = false;
  uint32_t blog = 0;
  std::unique_ptr<gpu::Program> prog;
  gpu::Kernel k_hcount, k_bscan, k_scatter, k_match, k_parse, k_tokens, k_streams, k_rans, k_pack, k_dstreams, k_dlz;
  // encoder buffers (sized for enc_nb blocks)
  size_t enc_nb = 0;
  gpu::Buffer e_in, e_match, e_mark, e_hcount, e_bstart, e_hfill, e_srt, e_idx, e_lit, e_tokLB, e_tokM, e_tokD, e_cnt, e_tok, e_ext,
      e_offsym, e_offw, e_lanes, e_freq, e_lanelen, e_sinfo, e_out, e_outlen;
  // decoder buffers (sized for dec_nb blocks)
  size_t dec_nb = 0, dec_comp_cap = 0;
  gpu::Buffer d_comp, d_boff, d_binfo, d_lit, d_tok, d_ext, d_offsym, d_cnt, d_err, d_out;
  // the decoder writes straight into host memory: on an integrated GPU that is the same RAM, and it saves the copy
  // back (measured: the LZ kernel 79 -> 71 ms and no download, 1275 -> 1407 MB/s)
  static constexpr bool host_out_direct = true;
  char* d_out_host = nullptr;
  size_t d_out_host_n = 0;
  std::vector<uint8_t> host_out, host_comp;
  std::vector<uint32_t> host_u32;
  Report* rep = nullptr;
  // ZERO_PROFILE=1: every stage waited for and timed, the times printed per batch (slow: no overlap)
  const bool profile = getenv("ZERO_PROFILE") && atoi(getenv("ZERO_PROFILE")) != 0;
  double prof_t0 = 0;
  std::string prof_line;
  void stage(const char* name) {
    if (!profile) return;
    dev.sync();
    const double t = zero_now();
    char b[64];
    snprintf(b, sizeof b, " %s %.1f", name, (t - prof_t0) * 1e3);
    prof_line += b;
    prof_t0 = t;
  }
  void prof_begin() { if (profile) { dev.sync(); prof_t0 = zero_now(); prof_line.clear(); } }
  void prof_end(const char* what, size_t bytes) {
    if (!profile) return;
    fprintf(stderr, "[profile] %s %.1f MB (ms):%s\n", what, bytes / 1e6, prof_line.c_str());
  }

  // derived sizes, as the kernels define them
  size_t B() const { return (size_t)1 << blog; }
  size_t HB() const { return (size_t)1 << (blog - 3); }
  size_t TCAP() const { return B() / 4 + 16; }
  size_t ECAP() const { return B() / 4 + B() / 128 + 16; }
  size_t OCAP() const { return 5 * B() / 8 + 8; }
  size_t OUTCAP() const { return B() + B() / 8 + 8192; }
  size_t LC0() const { return 2 * (B() / 64) + 16; }
  size_t LC1() const { return 2 * (TCAP() / 64 + 1) + 16; }
  size_t LC2() const { return 2 * (ECAP() / 64 + 1) + 16; }
  size_t LANES_PER_BLOCK() const { return 64 * (LC0() + 2 * LC1() + LC2()); }

  int level = 5;  // 1 fast .. 9 thorough: how many bucket neighbours the match finder checks, and when it stops
  std::string name() const override { return "GPU " + dev.name; }
  // the match finder's build options for a level: neighbours before / after the position in srt, "good enough" length
  static std::string level_opts(int level) {
    if (level <= 2) return "-DNCANDL=4u -DNCANDR=2u -DGOOD=32u";
    if (level <= 6) return "-DNCANDL=8u -DNCANDR=2u -DGOOD=64u";
    return "-DNCANDL=8u -DNCANDR=8u -DGOOD=274u";
  }
  // the encoder's batch (about 20 bytes of GPU scratch per input byte; 64 MiB measured slower end to end)
  size_t batch_bytes(uint32_t) const override { return (size_t)16 << 20; }
  size_t batch_bytes_decode(uint32_t) const override { return (size_t)256 << 20; }  // (more blocks in flight: one sub-group each)

  void open(Report& r) {
    rep = &r;
    if (opened) return;
    dev.open();
    opened = true;
    char b[256];
    snprintf(b, sizeof b, "GPU: %s (driver %s, %u EUs, %s)", dev.name.c_str(), dev.driver_version.c_str(), dev.eus,
             dev.integrated ? "integrated" : "discrete");
    r.say(b);
  }

  int built_level = -1;
  void ensure_program(uint32_t block_log) {
    if (blog == block_log && prog && built_level == level) return;
    blog = block_log;
    built_level = level;
    prog = std::make_unique<gpu::Program>();
    std::string src((const char*)kZeroCL, kZeroCL_len);
    char opts[128];
    // (ZERO_KOPTS: extra build options for experiments, e.g. "-DNCANDL=4u -DGOOD=64u", in place of the level's)
    const char* extra = getenv("ZERO_KOPTS");
    snprintf(opts, sizeof opts, "-DBLOG=%u -DG=%u %s", blog, G, extra ? extra : level_opts(level).c_str());
    prog->build(dev, src, opts);
    if (rep) {
      char b[160];
      snprintf(b, sizeof b, "kernels for %u KiB blocks %s (%.2f s)", 1u << (blog - 10), prog->from_cache ? "loaded from the cache" : "compiled",
               prog->build_seconds);
      rep->say(b);
    }
    k_hcount.create(*prog, "k_hash_count");
    k_bscan.create(*prog, "k_bucket_scan");
    k_scatter.create(*prog, "k_scatter");
    k_match.create(*prog, "k_match");
    k_parse.create(*prog, "k_parse");
    k_tokens.create(*prog, "k_tokens");
    k_streams.create(*prog, "k_streams");
    k_rans.create(*prog, "k_rans_enc");
    k_pack.create(*prog, "k_pack");
    k_dstreams.create(*prog, "k_dec_streams");
    k_dlz.create(*prog, "k_dec_lz");
    enc_nb = dec_nb = 0;  // (the buffers depend on the block size)
  }

  void ensure_enc(size_t nb) {
    if (nb <= enc_nb) return;
    const size_t N = nb * B();
    e_in.alloc(dev, N + 16);  // (16 bytes of slack: load4u reads the word after a block)
    e_match.alloc(dev, N * 4);
    e_mark.alloc(dev, N);
    e_hcount.alloc(dev, nb * HB() * 4);
    e_bstart.alloc(dev, nb * HB() * 4);
    e_hfill.alloc(dev, nb * HB() * 4);
    e_srt.alloc(dev, N * 4);
    e_idx.alloc(dev, N * 4);
    e_lit.alloc(dev, N);
    e_tokLB.alloc(dev, nb * TCAP() * 4);
    e_tokM.alloc(dev, nb * TCAP() * 4);
    e_tokD.alloc(dev, nb * TCAP() * 4);
    e_cnt.alloc(dev, nb * 8 * 4);
    e_tok.alloc(dev, nb * TCAP());
    e_ext.alloc(dev, nb * ECAP());
    e_offsym.alloc(dev, nb * TCAP());
    e_offw.alloc(dev, nb * OCAP());
    e_lanes.alloc(dev, nb * LANES_PER_BLOCK());
    e_freq.alloc(dev, nb * 4 * 256 * 2);
    e_lanelen.alloc(dev, nb * 4 * 64 * 4);
    e_sinfo.alloc(dev, nb * 4 * 4 * 4);
    e_out.alloc(dev, nb * OUTCAP());
    e_outlen.alloc(dev, nb * 4);
    enc_nb = nb;
  }

  void compress(const uint8_t* in, size_t n, uint32_t block_log, std::vector<BlockOut>& out) override {
    ensure_program(block_log);
    const size_t Bs = B();
    const size_t nb = (n + Bs - 1) / Bs;
    out.assign(nb, BlockOut{});
    if (nb == 0) return;
    ensure_enc(nb);
    const uint32_t nlast = (uint32_t)(n - (nb - 1) * Bs);
    const uint32_t nb32 = (uint32_t)nb;
    prof_begin();
    e_in.write(in, n);
    e_hcount.zero(nb * HB() * 4);
    e_mark.zero(nb * Bs);
    e_offw.zero(nb * OCAP());
    stage("upload");
    const uint32_t pos_groups = (uint32_t)(nb * Bs / G);
    k_hcount.arg(0, e_in); k_hcount.arg(1, e_hcount); k_hcount.arg(2, nb32); k_hcount.arg(3, nlast);
    k_hcount.launch(pos_groups, G);
    stage("hcount");
    k_bscan.arg(0, e_hcount); k_bscan.arg(1, e_bstart); k_bscan.arg(2, e_hfill); k_bscan.arg(3, nb32);
    k_bscan.launch(nb32, G);
    stage("bscan");
    k_scatter.arg(0, e_in); k_scatter.arg(1, e_bstart); k_scatter.arg(2, e_hfill); k_scatter.arg(3, e_srt); k_scatter.arg(4, e_idx);
    k_scatter.arg(5, nb32); k_scatter.arg(6, nlast);
    k_scatter.launch(pos_groups, G);
    stage("scatter");
    k_match.arg(0, e_in); k_match.arg(1, e_hcount); k_match.arg(2, e_bstart); k_match.arg(3, e_srt); k_match.arg(4, e_idx);
    k_match.arg(5, e_match); k_match.arg(6, nb32); k_match.arg(7, nlast);
    k_match.launch(pos_groups, G);
    stage("match");
    k_parse.arg(0, e_match); k_parse.arg(1, e_mark); k_parse.arg(2, nb32); k_parse.arg(3, nlast);
    k_parse.launch(nb32, G);
    stage("parse");
    k_tokens.arg(0, e_in); k_tokens.arg(1, e_match); k_tokens.arg(2, e_mark); k_tokens.arg(3, e_lit); k_tokens.arg(4, e_tokLB);
    k_tokens.arg(5, e_tokM); k_tokens.arg(6, e_tokD); k_tokens.arg(7, e_cnt); k_tokens.arg(8, nb32); k_tokens.arg(9, nlast);
    k_tokens.launch(nb32, G);
    stage("tokens");
    k_streams.arg(0, e_tokLB); k_streams.arg(1, e_tokM); k_streams.arg(2, e_tokD); k_streams.arg(3, e_cnt); k_streams.arg(4, e_tok);
    k_streams.arg(5, e_ext); k_streams.arg(6, e_offsym); k_streams.arg(7, e_offw); k_streams.arg(8, nb32);
    k_streams.launch(nb32, G);
    stage("streams");
    k_rans.arg(0, e_lit); k_rans.arg(1, e_tok); k_rans.arg(2, e_ext); k_rans.arg(3, e_offsym); k_rans.arg(4, e_cnt); k_rans.arg(5, e_lanes);
    k_rans.arg(6, e_freq); k_rans.arg(7, e_lanelen); k_rans.arg(8, e_sinfo); k_rans.arg(9, nb32);
    k_rans.launch(nb32 * 4, G);
    stage("rans");
    k_pack.arg(0, e_lit); k_pack.arg(1, e_tok); k_pack.arg(2, e_ext); k_pack.arg(3, e_offsym); k_pack.arg(4, e_offw); k_pack.arg(5, e_cnt);
    k_pack.arg(6, e_lanes); k_pack.arg(7, e_freq); k_pack.arg(8, e_lanelen); k_pack.arg(9, e_sinfo); k_pack.arg(10, e_out);
    k_pack.arg(11, e_outlen); k_pack.arg(12, nb32); k_pack.arg(13, nlast);
    k_pack.launch(nb32, G);
    stage("pack");
    host_u32.resize(nb);
    e_outlen.read(host_u32.data(), nb * 4);
    // the payloads: one read of the slabs that hold a compressed block (contiguous range), then each copied out
    size_t first = nb, last = 0;
    for (size_t b = 0; b < nb; ++b)
      if (host_u32[b] != 0xFFFFFFFFu) { if (b < first) first = b; last = b; }
    if (first < nb) {
      const size_t bytes = (last - first + 1) * OUTCAP();
      host_out.resize(bytes);
      e_out.read(host_out.data(), bytes, first * OUTCAP());
    }
    for (size_t b = 0; b < nb; ++b) {
      const uint32_t len = host_u32[b];
      if (len == 0xFFFFFFFFu) { out[b].method = 0; continue; }
      if (len > OUTCAP()) throw zb::Error("GPU encoder produced an impossible block size");
      out[b].method = 1;
      const uint8_t* p = host_out.data() + (b - first) * OUTCAP();
      out[b].data.assign(p, p + len);
    }
    stage("download");
    prof_end("compress", n);
  }

  void ensure_dec(size_t nb, size_t comp_bytes) {
    if (nb > dec_nb) {
      d_boff.alloc(dev, nb * 4);
      d_binfo.alloc(dev, nb * 4 * 4);
      d_lit.alloc(dev, nb * B());
      d_tok.alloc(dev, nb * TCAP() + 4096);  // (+4096: the decoder stages chunks past a block's streams)
      d_ext.alloc(dev, nb * ECAP() + 4096);
      d_offsym.alloc(dev, nb * TCAP() + 4096);
      d_cnt.alloc(dev, nb * 8 * 4);
      d_err.alloc(dev, nb * 4);
      if (host_out_direct) {
        if (d_out_host) { dev.sync(); zeMemFree(dev.ctx, d_out_host); }
        d_out_host_n = nb * B();
        d_out_host = (char*)dev.alloc_host(d_out_host_n);
      } else {
        d_out.alloc(dev, nb * B());
      }
      dec_nb = nb;
    }
    if (comp_bytes > dec_comp_cap) {
      d_comp.alloc(dev, comp_bytes + 4096);
      dec_comp_cap = comp_bytes + 4096;
    }
  }

  const uint8_t* decompress(const std::vector<BlockIn>& blocks) override {
    const size_t nb = blocks.size();
    if (nb == 0) return nullptr;
    if (!prog) throw zb::Error("GPU decoder: the block size is not set");
    // the payloads as one blob (already so when they come from one archive read)
    size_t total = 0, raw_total = 0;
    for (const BlockIn& k : blocks) { total += k.comp_len; raw_total += k.raw_len; }
    bool contiguous = true;
    for (size_t b = 1; b < nb && contiguous; ++b)
      contiguous = blocks[b].data == blocks[b - 1].data + blocks[b - 1].comp_len;
    const uint8_t* blob = blocks[0].data;
    if (!contiguous) {
      host_comp.resize(total);
      size_t o = 0;
      for (const BlockIn& k : blocks) { memcpy(host_comp.data() + o, k.data, k.comp_len); o += k.comp_len; }
      blob = host_comp.data();
    }
    ensure_dec(nb, total);
    std::vector<uint32_t> boff(nb), binfo(nb * 4);
    size_t o = 0;
    for (size_t b = 0; b < nb; ++b) {
      if (blocks[b].raw_len > B()) throw zb::Error("block larger than the archive's block size");
      boff[b] = (uint32_t)o;
      binfo[b * 4 + 0] = blocks[b].raw_len;
      binfo[b * 4 + 1] = blocks[b].comp_len;
      binfo[b * 4 + 2] = blocks[b].method;
      binfo[b * 4 + 3] = 0;
      o += blocks[b].comp_len;
    }
    prof_begin();
    d_comp.write(blob, total);
    d_boff.write(boff.data(), nb * 4);
    d_binfo.write(binfo.data(), nb * 16);
    d_err.zero(nb * 4);
    stage("upload");
    const uint32_t nb32 = (uint32_t)nb;
    k_dstreams.arg(0, d_comp); k_dstreams.arg(1, d_boff); k_dstreams.arg(2, d_binfo); k_dstreams.arg(3, d_lit); k_dstreams.arg(4, d_tok);
    k_dstreams.arg(5, d_ext); k_dstreams.arg(6, d_offsym); k_dstreams.arg(7, d_cnt); k_dstreams.arg(8, d_err); k_dstreams.arg(9, nb32);
    k_dstreams.launch(nb32 * 4, 64);
    stage("rans");
    k_dlz.arg(0, d_comp); k_dlz.arg(1, d_boff); k_dlz.arg(2, d_binfo); k_dlz.arg(3, d_lit); k_dlz.arg(4, d_tok); k_dlz.arg(5, d_ext);
    k_dlz.arg(6, d_offsym); k_dlz.arg(7, d_cnt);
    if (host_out_direct) { const char* hp = d_out_host; ZE_OK(zeKernelSetArgumentValue(k_dlz.k, 8, sizeof(hp), &hp)); }
    else k_dlz.arg(8, d_out);
    k_dlz.arg(9, d_err); k_dlz.arg(10, nb32);
    k_dlz.launch((nb32 + G / 16 - 1) / (G / 16), G);
    stage("lz");
    host_u32.resize(nb);
    d_err.read(host_u32.data(), nb * 4);
    for (size_t b = 0; b < nb; ++b)
      if (host_u32[b]) throw zb::Error("block damaged (GPU decoder, code " + std::to_string(host_u32[b]) + ")");
    // the blocks' bytes are B apart on the GPU: one copy into the host staging buffer, handed back as is when
    // only the last block is short (the usual case), else compacted there
    const size_t span = (nb - 1) * B() + blocks[nb - 1].raw_len;
    char* stg;
    if (host_out_direct) {
      stg = d_out_host;  // (already in host memory)
    } else {
      stg = dev.staging(span);
      dev.copy(stg, d_out.p, span);
      dev.sync();
    }
    if (span != raw_total) {
      size_t w = 0;
      for (size_t b = 0; b < nb; ++b) { memmove(stg + w, stg + b * B(), blocks[b].raw_len); w += blocks[b].raw_len; }
    }
    stage("download");
    prof_end("decompress", raw_total);
    return (const uint8_t*)stg;
  }
  // sets the block size the decoder will see (the archive's), compiling the kernels for it
  void prepare(uint32_t block_log) override { ensure_program(block_log); }
  ~GpuCodec() { if (d_out_host) { dev.sync(); zeMemFree(dev.ctx, d_out_host); } }
};

}  // namespace zero
