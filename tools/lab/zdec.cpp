// zdec: measures the ZB2 decoder on the GPU, kernels only, against the bytes the CPU encoder made. It also checks that the CPU
// decoder and the GPU decoder give back the input, bit for bit. Check tools/gpu_gate.ps1 first: this uses the GPU.
//
//   zdec <kernels/zero.cl> <file> [--block 20] [--K 255] [--T 0] [--depth 8] [--rounds 5] [--opts "-DCHUNK=8"] [--max MiB]
#include "archive.h"
#include "gpu.h"
#include <cstdio>
#include <fstream>
#include <sstream>

static int run(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: zdec <zero.cl> <file> [options]\n"); return 2; }
  uint32_t block_log = 20, rounds = 5;
  zb2::Params P;
  std::string extra;
  size_t max_bytes = (size_t)1 << 40;
  bool nocheck = false;  // (an ablation that may leave wrong bytes: time only)
  for (int i = 3; i + 1 < argc; i += 2) {
    const std::string s = argv[i];
    if (s == "--block") block_log = (uint32_t)atoi(argv[i + 1]);
    else if (s == "--K") P.max_group = (uint32_t)atoi(argv[i + 1]);
    else if (s == "--T") P.cut_below = (uint32_t)atoi(argv[i + 1]);
    else if (s == "--depth") P.depth = atoi(argv[i + 1]);
    else if (s == "--rounds") rounds = (uint32_t)atoi(argv[i + 1]);
    else if (s == "--opts") extra = argv[i + 1];
    else if (s == "--max") max_bytes = (size_t)atoll(argv[i + 1]) << 20;
    else if (s == "--nocheck") nocheck = atoi(argv[i + 1]) != 0;
  }
  std::ifstream src(argv[1], std::ios::binary);
  std::stringstream ss;
  ss << src.rdbuf();
  FILE* f = fopen(argv[2], "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", argv[2]); return 1; }
  fseek(f, 0, SEEK_END);
  const size_t size = std::min((size_t)ftell(f), max_bytes);
  fseek(f, 0, SEEK_SET);
  std::vector<uint8_t> data(size);
  if (fread(data.data(), 1, size, f) != size) return 1;
  fclose(f);
  const size_t B = (size_t)1 << block_log, nb = (size + B - 1) / B;
  std::vector<std::vector<uint8_t>> payload(nb);
  std::vector<uint8_t> stored(nb, 0);
  zero::parallel_for(nb, [&](size_t b) {
    const uint32_t len = (uint32_t)std::min(B, size - b * B);
    if (!zb2::compress_block(data.data() + b * B, len, payload[b], P)) stored[b] = 1;
  });
  size_t comp = 0;
  for (size_t b = 0; b < nb; ++b) comp += stored[b] ? std::min(B, size - b * B) : payload[b].size();
  printf("%s: %.1f MiB, %.2f %%, %zu blocks of %zu KiB, K %u T %u depth %d\n", argv[2], size / 1048576.0, 100.0 * comp / size, nb, B >> 10, P.max_group, P.cut_below, P.depth);
  // the CPU decoder first: a format bug should not reach the GPU
  std::vector<uint8_t> back(size);
  double t0 = zero_now();
  zero::parallel_for(nb, [&](size_t b) {
    const uint32_t len = (uint32_t)std::min(B, size - b * B);
    if (stored[b]) memcpy(back.data() + b * B, data.data() + b * B, len);
    else zb2::decompress_block(payload[b].data(), payload[b].size(), back.data() + b * B, len);
  });
  printf("CPU decoder: %.0f MB/s, %s\n", size / (zero_now() - t0) / 1e6, back == data ? "identical" : "MISMATCH");
  if (back != data) return 1;
  gpu::Device dev;
  dev.open();
  gpu::Program prog;
  char opts[256];
  snprintf(opts, sizeof opts, "-DBLOG=%u -DG=256 %s", block_log, extra.c_str());
  prog.build(dev, ss.str(), opts);
  gpu::Kernel ks, kl;
  ks.create(prog, "k_dec_streams");
  kl.create(prog, "k_dec_lz");
  const size_t TCAP = B / 4 + 16, ECAP = B / 5 + 64;
  gpu::Buffer d_comp, d_boff, d_binfo, d_lit, d_tok, d_ext, d_sym, d_grp, d_cnt, d_err, d_out;
  d_comp.alloc_host(dev, comp + 4096 + nb * 8);
  d_out.alloc_host(dev, nb * B + 64);
  d_boff.alloc(dev, nb * 4);
  d_binfo.alloc(dev, nb * 16);
  d_lit.alloc(dev, nb * B + 64);
  d_tok.alloc(dev, nb * TCAP + 64);
  d_ext.alloc(dev, nb * ECAP + 64);
  d_sym.alloc(dev, nb * TCAP + 64);
  d_grp.alloc(dev, nb * TCAP + 64);
  d_cnt.alloc(dev, nb * 32);
  d_err.alloc(dev, nb * 4);
  std::vector<uint32_t> boff(nb), binfo(nb * 4);
  size_t o = 0;
  for (size_t b = 0; b < nb; ++b) {
    const uint32_t len = (uint32_t)std::min(B, size - b * B);
    const size_t cl = stored[b] ? len : payload[b].size();
    boff[b] = (uint32_t)o;
    binfo[b * 4] = len;
    binfo[b * 4 + 1] = (uint32_t)cl;
    binfo[b * 4 + 2] = stored[b] ? 0 : 1;
    memcpy(d_comp.p + o, stored[b] ? data.data() + b * B : payload[b].data(), cl);
    o += cl;
  }
  d_boff.write(boff.data(), nb * 4);
  d_binfo.write(binfo.data(), nb * 16);
  const uint32_t nb32 = (uint32_t)nb;
  ks.arg(0, d_comp); ks.arg(1, d_boff); ks.arg(2, d_binfo); ks.arg(3, d_lit); ks.arg(4, d_tok); ks.arg(5, d_ext); ks.arg(6, d_sym); ks.arg(7, d_grp); ks.arg(8, d_cnt); ks.arg(9, d_err); ks.arg(10, nb32);
  kl.arg(0, d_comp); kl.arg(1, d_boff); kl.arg(2, d_binfo); kl.arg(3, d_lit); kl.arg(4, d_tok); kl.arg(5, d_ext); kl.arg(6, d_sym); kl.arg(7, d_grp); kl.arg(8, d_cnt); kl.arg(9, d_out); kl.arg(10, d_err); kl.arg(11, nb32);
  std::vector<double> rs, ls;
  bool same = true;
  for (uint32_t r = 0; r < rounds; ++r) {
    d_err.zero(nb * 4);
    dev.sync();
    double a = zero_now();
    ks.launch(nb32 * 5, 64);
    dev.sync();
    double m = zero_now();
    kl.launch((nb32 + 15) / 16, 256);
    dev.sync();
    double e = zero_now();
    rs.push_back(m - a);
    ls.push_back(e - m);
    std::vector<uint32_t> errs(nb);
    d_err.read(errs.data(), nb * 4);
    size_t bad = 0;
    for (uint32_t x : errs) if (x) ++bad;
    if (!nocheck) same = same && !bad && memcmp(d_out.p, data.data(), size) == 0;
    if (bad || r == 0) printf("round %u: streams %.1f ms, lz %.1f ms, %zu blocks in error, %s\n", r, (m - a) * 1e3, (e - m) * 1e3, bad, same ? "identical" : "MISMATCH");
    if (!same) {
      for (size_t b = 0; b < nb; ++b) {
        const uint32_t len = (uint32_t)std::min(B, size - b * B);
        if (memcmp(d_out.p + b * B, data.data() + b * B, len)) {
          size_t at = 0;
          while (at < len && (uint8_t)d_out.p[b * B + at] == data[b * B + at]) ++at;
          printf("first difference: block %zu at %zu of %u, err %u\n", b, at, len, errs[b]);
          break;
        }
      }
      return 1;
    }
  }
  std::sort(rs.begin(), rs.end());
  std::sort(ls.begin(), ls.end());
  const double rm = rs[rs.size() / 2], lm = ls[ls.size() / 2];
  printf("median: streams %.1f ms, lz %.1f ms, total %.1f ms, %.0f MB/s (lz alone %.0f MB/s)\n", rm * 1e3, lm * 1e3, (rm + lm) * 1e3, size / (rm + lm) / 1e6, size / lm / 1e6);
  return 0;
}

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 4096);
  try {
    return run(argc, argv);
  } catch (const std::exception& e) {
    printf("error: %s\n", e.what());
    return 1;
  }
}
