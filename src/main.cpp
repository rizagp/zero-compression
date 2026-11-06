// zero: lossless compression of files and folders on an Intel GPU (Level Zero). The command line, and with
// ZERO_GUI the window (gui_win.h / gui_gtk.h). One source for Windows and Linux.
#include "archive.h"
#include "gpu.h"
#include "safe_codec.h"
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#define ZERO_VERSION "0.1.0"

namespace zero {

inline std::string human(uint64_t b) {
  char s[64];
  if (b < 1024) snprintf(s, sizeof s, "%llu B", (unsigned long long)b);
  else if (b < (1ull << 20)) snprintf(s, sizeof s, "%.1f KiB", b / 1024.0);
  else if (b < (1ull << 30)) snprintf(s, sizeof s, "%.1f MiB", b / 1048576.0);
  else snprintf(s, sizeof s, "%.2f GiB", b / 1073741824.0);
  return s;
}
inline std::string rate(uint64_t bytes, double secs) {
  if (secs <= 0) return "-";
  char s[64];
  snprintf(s, sizeof s, "%.0f MB/s", bytes / secs / 1e6);
  return s;
}

// The codec the user asked for: the GPU unless --cpu, the CPU when the GPU cannot be opened (said once).
struct CodecChoice {
  std::unique_ptr<Codec> codec;
  std::string note;
};
inline CodecChoice choose_codec(bool want_cpu, Report& rep, int level) {
  CodecChoice c;
  if (!want_cpu) {
    try {
      auto g = std::make_unique<GpuCodec>();
      g->level = level;
      g->open(rep);
      auto s = std::make_unique<SafeCodec>(std::move(g));
      s->set_report(&rep);
      s->cpu.depth = level <= 1 ? 8 : level >= 9 ? 256 : 32;
      c.codec = std::move(s);
      return c;
    } catch (const std::exception& e) {
      c.note = std::string("GPU unavailable (") + e.what() + "); using the CPU";
      rep.say(c.note);
    }
  }
  auto cpu = std::make_unique<CpuCodec>();
  cpu->depth = level <= 1 ? 8 : level >= 9 ? 256 : 32;
  c.codec = std::move(cpu);
  return c;
}

inline void usage() {
  printf("zero %s: lossless compression of files and folders on the GPU (Intel Level Zero)\n\n"
         "  zero c <file|folder>... -o <archive.zero> [--block N] [--cpu] [--level 1..9]\n"
         "  zero x <archive.zero> [-o <folder>] [--cpu]\n"
         "  zero l <archive.zero>\n"
         "  zero t <archive.zero> [--cpu]\n"
         "  zero bench <file> [--block N] [--cpu]\n"
         "  zero info\n\n"
         "  --block N   block size as log2 bytes, 16 (64 KiB) .. 20 (1 MiB); default 18 (256 KiB)\n"
         "  --cpu       use the CPU instead of the GPU (smaller archives; the CPU is busy)\n"
         "  --level L   how hard to look for matches: 1 fast .. 9 thorough (default 5)\n"
         "  -q          no progress\n",
         ZERO_VERSION);
}

struct Args {
  std::string cmd, out;
  std::vector<std::string> inputs;
  uint32_t block_log = 18;
  bool cpu = false, quiet = false;
  int level = 5;
};

inline bool parse(int argc, char** argv, Args& a) {
  if (argc < 2) return false;
  a.cmd = argv[1];
  for (int i = 2; i < argc; ++i) {
    const std::string s = argv[i];
    if (s == "-o" && i + 1 < argc) a.out = argv[++i];
    else if (s == "--block" && i + 1 < argc) a.block_log = (uint32_t)atoi(argv[++i]);
    else if (s == "--level" && i + 1 < argc) a.level = atoi(argv[++i]);
    else if (s == "--cpu") a.cpu = true;
    else if (s == "--gpu") a.cpu = false;
    else if (s == "-q") a.quiet = true;
    else if (!s.empty() && s[0] == '-' && s != "-") { fprintf(stderr, "unknown option %s\n", s.c_str()); return false; }
    else a.inputs.push_back(s);
  }
  return true;
}

struct ConsoleProgress {
  bool quiet;
  double last = 0;
  void operator()(uint64_t done, uint64_t total) {
    if (quiet) return;
    const double t = zero_now();
    if (done < total && t - last < 0.1) return;
    last = t;
    fprintf(stderr, "\r  %5.1f %%  %s / %s      ", total ? 100.0 * done / total : 100.0, human(done).c_str(), human(total).c_str());
    if (done >= total) fprintf(stderr, "\n");
    fflush(stderr);
  }
};

inline int cmd_info() {
  try {
    gpu::Device d;
    d.open();
    printf("GPU: %s\n", d.name.c_str());
    printf("  driver %s, %u EUs, %s memory, work-group up to %u, local memory %u KiB%s\n", d.driver_version.c_str(), d.eus,
           human(d.memory).c_str(), d.max_group, d.local_mem / 1024, d.integrated ? ", integrated" : "");
    printf("  kernel cache: %s\n", d.cache_dir.empty() ? "(off)" : d.cache_dir.c_str());
    GpuCodec g;
    Report rep;
    rep.log = [](const std::string& s) { printf("  %s\n", s.c_str()); };
    g.open(rep);
    g.prepare(19);
    printf("  kernels: ready\n");
  } catch (const std::exception& e) {
    printf("GPU: not available: %s\n", e.what());
  }
  printf("CPU: %u threads\n", hardware_threads());
  return 0;
}

inline int cmd_compress(const Args& a) {
  if (a.inputs.empty()) { fprintf(stderr, "nothing to compress\n"); return 2; }
  std::string out = a.out;
  if (out.empty()) {
    fs::path p = zero_path(a.inputs[0]);
    while (!p.empty() && !p.has_filename()) p = p.parent_path();
    out = zero_utf8(p.filename()) + ".zero";
  }
  Report rep;
  rep.log = [](const std::string& s) { fprintf(stderr, "%s\n", s.c_str()); };
  ConsoleProgress prog{a.quiet};
  rep.progress = [&](uint64_t d, uint64_t t) { prog(d, t); };
  CodecChoice c = choose_codec(a.cpu, rep, a.level);
  fprintf(stderr, "codec: %s, blocks of %s\n", c.codec->name().c_str(), human((uint64_t)1 << a.block_log).c_str());
  const CompressStats st = compress_archive(a.inputs, out, *c.codec, a.block_log, rep);
  printf("%s: %s -> %s (%.1f %%), %llu blocks (%llu stored), %.2f s (%s; codec %.2f s, %s)\n", out.c_str(),
         human(st.raw).c_str(), human(st.compressed).c_str(), st.raw ? 100.0 * st.compressed / st.raw : 0.0,
         (unsigned long long)st.blocks, (unsigned long long)st.stored_blocks, st.seconds, rate(st.raw, st.seconds).c_str(),
         st.codec_seconds, rate(st.raw, st.codec_seconds).c_str());
  return 0;
}

inline int cmd_extract(const Args& a) {
  if (a.inputs.size() != 1) { fprintf(stderr, "give one archive\n"); return 2; }
  std::string out = a.out;
  if (out.empty()) {
    fs::path p = zero_path(a.inputs[0]);
    out = zero_utf8(p.stem());
    if (out.empty() || out == a.inputs[0]) out = ".";
  }
  Report rep;
  rep.log = [](const std::string& s) { fprintf(stderr, "%s\n", s.c_str()); };
  ConsoleProgress prog{a.quiet};
  rep.progress = [&](uint64_t d, uint64_t t) { prog(d, t); };
  CodecChoice c = choose_codec(a.cpu, rep, a.level);
  fprintf(stderr, "codec: %s\n", c.codec->name().c_str());
  const ExtractStats st = extract_archive(a.inputs[0], out, *c.codec, rep);
  printf("%s: %zu files, %zu folders, %s in %.2f s (%s; codec %.2f s, %s), all checksums match\n", out.c_str(), st.files,
         st.folders, human(st.raw).c_str(), st.seconds, rate(st.raw, st.seconds).c_str(), st.codec_seconds,
         rate(st.raw, st.codec_seconds).c_str());
  return 0;
}

inline int cmd_test(const Args& a) {
  if (a.inputs.size() != 1) { fprintf(stderr, "give one archive\n"); return 2; }
  Report rep;
  rep.log = [](const std::string& s) { fprintf(stderr, "%s\n", s.c_str()); };
  ConsoleProgress prog{a.quiet};
  rep.progress = [&](uint64_t d, uint64_t t) { prog(d, t); };
  CodecChoice c = choose_codec(a.cpu, rep, a.level);
  const ExtractStats st = test_archive(a.inputs[0], *c.codec, rep);
  printf("%s: OK, %zu files, %zu folders, %s in %llu blocks, %.2f s (%s)\n", a.inputs[0].c_str(), st.files, st.folders,
         human(st.raw).c_str(), (unsigned long long)st.blocks, st.seconds, rate(st.raw, st.seconds).c_str());
  return 0;
}

inline int cmd_list(const Args& a) {
  if (a.inputs.size() != 1) { fprintf(stderr, "give one archive\n"); return 2; }
  const ArchiveInfo info = list_archive(a.inputs[0]);
  uint64_t comp = info.table_off - 24 - info.n_blocks * 16;
  printf("%s: %zu entries, %s in %llu blocks of %s, compressed to %s (%.1f %%)\n", a.inputs[0].c_str(), info.entries.size(),
         human(info.total_raw).c_str(), (unsigned long long)info.n_blocks, human((uint64_t)1 << info.block_log).c_str(),
         human(comp).c_str(), info.total_raw ? 100.0 * comp / info.total_raw : 0.0);
  for (const Entry& e : info.entries) {
    if (e.dir) printf("%14s  %s/\n", "", e.name.c_str());
    else printf("%14llu  %s\n", (unsigned long long)e.size, e.name.c_str());
  }
  return 0;
}

// compresses one file in memory with the chosen codec, decompresses with both codecs, checks, prints the speeds
inline int cmd_bench(const Args& a) {
  if (a.inputs.size() != 1) { fprintf(stderr, "give one file\n"); return 2; }
  FILE* f = open_file(zero_path(a.inputs[0]), "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", a.inputs[0].c_str()); return 1; }
  std::vector<uint8_t> data;
  {
    zseek(f, 0, SEEK_END);
    const int64_t n = ztell(f);
    zseek(f, 0, SEEK_SET);
    data.resize((size_t)n);
    if (fread(data.data(), 1, data.size(), f) != data.size()) { fclose(f); fprintf(stderr, "read failed\n"); return 1; }
    fclose(f);
  }
  Report rep;
  rep.log = [](const std::string& s) { fprintf(stderr, "%s\n", s.c_str()); };
  CodecChoice c = choose_codec(a.cpu, rep, a.level);
  CpuCodec cpu;
  const size_t B = (size_t)1 << a.block_log;
  printf("%s: %s, blocks of %s, codec %s\n", a.inputs[0].c_str(), human(data.size()).c_str(), human(B).c_str(), c.codec->name().c_str());
  std::vector<BlockOut> outs;
  size_t batch = std::max(B, (c.codec->batch_bytes(a.block_log) / B) * B);
  std::vector<std::vector<BlockOut>> all;
  c.codec->prepare(a.block_log);  // (the kernels built or loaded before the clock starts)
  double t0 = zero_now();
  for (size_t off = 0; off < data.size(); off += batch) {
    c.codec->compress(data.data() + off, std::min(batch, data.size() - off), a.block_log, outs);
    all.push_back(outs);
  }
  const double tc = zero_now() - t0;
  uint64_t comp = 0, stored = 0, blocks = 0;
  std::vector<BlockIn> ins;
  std::vector<uint8_t> raw;
  size_t pos = 0;
  for (auto& batch_out : all)
    for (BlockOut& o : batch_out) {
      const uint32_t len = (uint32_t)std::min(B, data.size() - pos);
      BlockIn in;
      in.raw_len = len;
      if (o.method == 0 || o.data.size() >= len) { in.method = 0; in.data = data.data() + pos; in.comp_len = len; ++stored; }
      else { in.method = 1; in.data = o.data.data(); in.comp_len = (uint32_t)o.data.size(); }
      comp += in.comp_len;
      ins.push_back(in);
      pos += len;
      ++blocks;
    }
  printf("  compress:   %.3f s, %s, %s -> %s (%.2f %%), %llu blocks, %llu stored\n", tc, rate(data.size(), tc).c_str(),
         human(data.size()).c_str(), human(comp).c_str(), data.empty() ? 0.0 : 100.0 * comp / data.size(),
         (unsigned long long)blocks, (unsigned long long)stored);
  auto run_decode = [&](Codec& dc, const char* label) {
    raw.assign(data.size(), 0);
    dc.prepare(a.block_log);
    const size_t per = std::max<size_t>(1, dc.batch_bytes_decode(a.block_log) / B);
    const double d0 = zero_now();
    size_t o = 0;
    for (size_t i = 0; i < ins.size(); i += per) {
      std::vector<BlockIn> part(ins.begin() + i, ins.begin() + std::min(ins.size(), i + per));
      const uint8_t* r = dc.decompress(part);
      size_t n = 0;
      for (const BlockIn& k : part) n += k.raw_len;
      memcpy(raw.data() + o, r, n);
      o += n;
    }
    const double td = zero_now() - d0;
    const bool ok = raw == data;
    printf("  decompress: %.3f s, %s with %s: %s\n", td, rate(data.size(), td).c_str(), label, ok ? "identical" : "MISMATCH");
    return ok;
  };
  bool ok = run_decode(*c.codec, c.codec->name().c_str());
  if (const char* e = getenv("ZERO_BENCH_REPEAT"))
    for (int i = 1; i < atoi(e); ++i) ok = run_decode(*c.codec, c.codec->name().c_str()) && ok;
  if (c.codec->name() != cpu.name()) ok = run_decode(cpu, cpu.name().c_str()) && ok;
  return ok ? 0 : 1;
}

inline int cli_main(int argc, char** argv) {
  Args a;
  if (!parse(argc, argv, a)) { usage(); return 2; }
  const double t_main = zero_now();
  struct AtEnd {
    double t0;
    ~AtEnd() { if (getenv("ZERO_PROFILE")) fprintf(stderr, "[profile] main, up to teardown %.3f s\n", zero_now() - t0); }
  } at_end{t_main};
  try {
    if (a.cmd == "c" || a.cmd == "a") return cmd_compress(a);
    if (a.cmd == "x") return cmd_extract(a);
    if (a.cmd == "t") return cmd_test(a);
    if (a.cmd == "l") return cmd_list(a);
    if (a.cmd == "bench") return cmd_bench(a);
    if (a.cmd == "info") return cmd_info();
    if (a.cmd == "-h" || a.cmd == "--help" || a.cmd == "help") { usage(); return 0; }
  } catch (const Cancelled&) {
    fprintf(stderr, "cancelled\n");
    return 3;
  } catch (const std::exception& e) {
    fprintf(stderr, "zero: %s\n", e.what());
    return 1;
  }
  usage();
  return 2;
}

}  // namespace zero

#ifdef ZERO_GUI
#ifdef _WIN32
#include "gui_win.h"
#else
#include "gui_gtk.h"
#endif
#else
int main(int argc, char** argv) { return zero::cli_main(argc, argv); }
#endif
