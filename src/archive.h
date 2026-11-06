// zero: the .zero container, and compressing / listing / extracting files and folders with any block codec.
//
// Layout (little endian):
//   Header (24 bytes): "ZERO", u8 version (1), u8 block_log (16..20), u16 flags (0), u64 total_raw, u64 n_blocks
//   Blocks, each: u32 raw_len, u32 comp_len, u8 method (0 stored, 1 ZB1), u8 pad[3], u32 crc32 of the raw bytes,
//                 comp_len bytes
//   Entry table: u32 n_entries; each: u8 type (0 file, 1 folder), u8 pad[3], u32 name_len, name (UTF-8, '/'
//                separators, relative), u64 size, i64 mtime (Unix seconds), u32 attributes (reserved), u32 crc32
//   Footer (24 bytes): u64 table_offset, u64 table_len, u32 table_crc32, "ZEND"
// The files' bytes, in entry order, form one "solid" stream that is cut into blocks of 2^block_log bytes: many
// small files share blocks, and every block decompresses on its own.
#pragma once
#include <future>
#include "codec.h"
#include "crc32.h"
#include "platform.h"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace zero {

namespace fs = std::filesystem;

// ---- what the UI (CLI or window) hears while an operation runs
struct Report {
  std::function<void(const std::string&)> log;              // a line of text
  std::function<void(uint64_t done, uint64_t total)> progress;  // bytes of the solid stream done
  std::atomic<bool>* cancel = nullptr;
  void say(const std::string& s) const { if (log) log(s); }
  void at(uint64_t d, uint64_t t) const { if (progress) progress(d, t); }
  bool cancelled() const { return cancel && cancel->load(); }
};

struct Cancelled : std::runtime_error {
  Cancelled() : std::runtime_error("cancelled") {}
};

// ---- a block codec: compresses or decompresses a batch of blocks (the CPU with threads, the GPU with kernels)
struct BlockOut {
  uint8_t method = 0;           // 0 stored, 1 ZB1
  std::vector<uint8_t> data;    // the payload when method 1 (stored blocks are written from the input)
};
struct BlockIn {
  uint8_t method = 0;
  uint32_t raw_len = 0;
  const uint8_t* data = nullptr;  // comp_len bytes
  uint32_t comp_len = 0;
};
struct Codec {
  virtual ~Codec() = default;
  virtual std::string name() const = 0;
  // in: n bytes = whole blocks of block_size (the last may be shorter); out gets one entry per block
  virtual void compress(const uint8_t* in, size_t n, uint32_t block_log, std::vector<BlockOut>& out) = 0;
  // blocks -> their raw bytes, one after the other (sum of raw_len bytes), in memory the codec owns; valid until
  // the next call
  virtual const uint8_t* decompress(const std::vector<BlockIn>& blocks) = 0;
  // how many bytes of input one batch should hold (the GPU has a memory budget; the CPU a cache one)
  virtual size_t batch_bytes(uint32_t block_log) const = 0;
  virtual size_t batch_bytes_decode(uint32_t block_log) const { return batch_bytes(block_log); }
  // called with an archive's block size before its blocks are decompressed (the GPU builds its kernels for it)
  virtual void prepare(uint32_t) {}
};

inline unsigned hardware_threads() {
  const unsigned n = std::thread::hardware_concurrency();
  return n ? n : 4;
}

// runs f(i) for i in [0, n) on all cores
inline void parallel_for(size_t n, const std::function<void(size_t)>& f) {
  const unsigned nt = (unsigned)std::min<size_t>(hardware_threads(), n);
  if (nt <= 1) { for (size_t i = 0; i < n; ++i) f(i); return; }
  std::atomic<size_t> next{0};
  std::vector<std::thread> ts;
  std::exception_ptr err;
  std::atomic<bool> failed{false};
  for (unsigned t = 0; t < nt; ++t)
    ts.emplace_back([&] {
      for (;;) {
        const size_t i = next.fetch_add(1);
        if (i >= n || failed.load()) return;
        try { f(i); } catch (...) { if (!failed.exchange(true)) err = std::current_exception(); return; }
      }
    });
  for (auto& t : ts) t.join();
  if (err) std::rethrow_exception(err);
}

// ---- the CPU codec (the reference; also the fallback without a GPU)
struct CpuCodec : Codec {
  int depth = 32;
  std::string name() const override { return "CPU (" + std::to_string(hardware_threads()) + " threads)"; }
  size_t batch_bytes(uint32_t) const override { return (size_t)64 << 20; }
  void compress(const uint8_t* in, size_t n, uint32_t block_log, std::vector<BlockOut>& out) override {
    const size_t B = (size_t)1 << block_log;
    const size_t nb = (n + B - 1) / B;
    out.assign(nb, BlockOut{});
    parallel_for(nb, [&](size_t b) {
      const size_t off = b * B;
      const uint32_t len = (uint32_t)std::min(B, n - off);
      std::vector<uint8_t> payload;
      if (zb::compress_block_cpu(in + off, len, payload, depth)) { out[b].method = 1; out[b].data = std::move(payload); }
      else out[b].method = 0;
    });
  }
  std::vector<uint8_t> outbuf;
  const uint8_t* decompress(const std::vector<BlockIn>& blocks) override {
    std::vector<size_t> offs(blocks.size() + 1, 0);
    for (size_t i = 0; i < blocks.size(); ++i) offs[i + 1] = offs[i] + blocks[i].raw_len;
    outbuf.resize(offs.back());
    uint8_t* out = outbuf.data();
    parallel_for(blocks.size(), [&](size_t b) {
      const BlockIn& k = blocks[b];
      if (k.method == 0) {
        if (k.comp_len != k.raw_len) throw zb::Error("stored block with a wrong length");
        memcpy(out + offs[b], k.data, k.raw_len);
      } else if (k.method == 1) {
        zb::decompress_block_cpu(k.data, k.comp_len, out + offs[b], k.raw_len);
      } else {
        throw zb::Error("unknown block method");
      }
    });
    return out;
  }
};

// ---- entries
struct Entry {
  bool dir = false;
  std::string name;  // relative, '/' separated
  uint64_t size = 0;
  int64_t mtime = 0;
  uint32_t attr = 0;
  uint32_t crc = 0;
  fs::path source;   // when compressing: where it is on disk
};

inline int64_t mtime_of(const fs::path& p) {
  std::error_code ec;
  const auto t = fs::last_write_time(p, ec);
  if (ec) return 0;
  const auto sys = std::chrono::clock_cast<std::chrono::system_clock>(t);
  return std::chrono::duration_cast<std::chrono::seconds>(sys.time_since_epoch()).count();
}
inline void set_mtime(const fs::path& p, int64_t secs) {
  if (!secs) return;
  std::error_code ec;
  const auto sys = std::chrono::system_clock::time_point(std::chrono::seconds(secs));
  fs::last_write_time(p, std::chrono::clock_cast<std::chrono::file_clock>(sys), ec);
}

inline std::string generic(const fs::path& p) {
  std::string s = p.generic_string();
  return s;
}

// The entries for a list of inputs: a file is one entry under its own name; a folder is itself and everything in
// it, under its name. Sorted inside each folder for a stable archive. Names that would collide keep the first.
inline std::vector<Entry> collect(const std::vector<std::string>& inputs, Report& rep) {
  std::vector<Entry> es;
  std::vector<std::string> seen;
  auto add = [&](Entry e) {
    for (const auto& s : seen)
      if (s == e.name) { rep.say("skipped (same name already in the archive): " + e.name); return; }
    seen.push_back(e.name);
    es.push_back(std::move(e));
  };
  std::function<void(const fs::path&, const std::string&)> walk = [&](const fs::path& dir, const std::string& rel) {
    std::vector<fs::directory_entry> items;
    std::error_code ec;
    for (fs::directory_iterator it(dir, fs::directory_options::skip_permission_denied, ec), end; it != end; it.increment(ec)) {
      if (ec) break;
      items.push_back(*it);
    }
    std::sort(items.begin(), items.end(), [](const fs::directory_entry& a, const fs::directory_entry& b) {
      return a.path().filename().u8string() < b.path().filename().u8string();
    });
    for (const auto& it : items) {
      if (rep.cancelled()) throw Cancelled();
      const std::string name = rel + "/" + zero_utf8(it.path().filename());
      std::error_code e2;
      if (it.is_symlink(e2)) continue;  // (links are not followed; a later version may store them)
      if (it.is_directory(e2)) {
        Entry d;
        d.dir = true;
        d.name = name;
        d.mtime = mtime_of(it.path());
        d.source = it.path();
        add(std::move(d));
        walk(it.path(), name);
      } else if (it.is_regular_file(e2)) {
        Entry f;
        f.name = name;
        f.size = it.file_size(e2);
        if (e2) f.size = 0;
        f.mtime = mtime_of(it.path());
        f.source = it.path();
        add(std::move(f));
      }
    }
  };
  for (const std::string& in : inputs) {
    fs::path p = zero_path(in);
    std::error_code ec;
    p = fs::absolute(p, ec);
    while (!p.empty() && !p.has_filename()) p = p.parent_path();  // ("folder/" -> "folder")
    if (!fs::exists(p, ec)) throw zb::Error("not found: " + in);
    const std::string top = zero_utf8(p.filename());
    if (fs::is_directory(p, ec)) {
      Entry d;
      d.dir = true;
      d.name = top;
      d.mtime = mtime_of(p);
      d.source = p;
      add(std::move(d));
      walk(p, top);
    } else if (fs::is_regular_file(p, ec)) {
      Entry f;
      f.name = top;
      f.size = fs::file_size(p, ec);
      f.mtime = mtime_of(p);
      f.source = p;
      add(std::move(f));
    } else {
      rep.say("skipped (not a file or folder): " + in);
    }
  }
  return es;
}

// ---- writing
inline void w32(FILE* f, uint32_t v) { uint8_t b[4] = {(uint8_t)v, (uint8_t)(v >> 8), (uint8_t)(v >> 16), (uint8_t)(v >> 24)}; fwrite(b, 1, 4, f); }
inline void w64(FILE* f, uint64_t v) { w32(f, (uint32_t)v); w32(f, (uint32_t)(v >> 32)); }
inline void w8(FILE* f, uint8_t v) { fwrite(&v, 1, 1, f); }

struct CompressStats {
  uint64_t raw = 0, compressed = 0, blocks = 0, stored_blocks = 0;
  size_t files = 0, folders = 0;
  double seconds = 0, codec_seconds = 0;
};

inline FILE* open_file(const fs::path& p, const char* mode) {
#ifdef _WIN32
  std::wstring m(mode, mode + strlen(mode));
  return _wfopen(p.wstring().c_str(), m.c_str());
#else
  return fopen(p.string().c_str(), mode);
#endif
}
// 64-bit seek and tell (archives can be larger than 2 GiB)
inline int zseek(FILE* f, int64_t off, int whence) {
#ifdef _WIN32
  return _fseeki64(f, off, whence);
#else
  return fseeko(f, (off_t)off, whence);
#endif
}
inline int64_t ztell(FILE* f) {
#ifdef _WIN32
  return _ftelli64(f);
#else
  return (int64_t)ftello(f);
#endif
}

// Reads the entries' bytes in order, delivering batches of up to `batch` bytes, and the CRC of each file.
// a running CRC over a large piece, on all the CPU's threads: the GPU would otherwise wait for one thread's checksum
inline uint32_t crc_update_mt(uint32_t crc, const uint8_t* p, size_t n) {
  const size_t CH = (size_t)1 << 20;
  if (n < 4 * CH) return crc32::update(crc, p, n);
  const size_t pieces = (n + CH - 1) / CH;
  std::vector<uint32_t> c(pieces);
  parallel_for(pieces, [&](size_t i) { c[i] = crc32::of(p + i * CH, std::min(CH, n - i * CH)); });
  for (size_t i = 0; i < pieces; ++i) crc = crc32::combine(crc, c[i], std::min(CH, n - i * CH));
  return crc;
}

struct SolidReader {
  std::vector<Entry>& es;
  size_t idx = 0;        // the entry being read
  uint64_t pos = 0;      // bytes read of that entry
  FILE* f = nullptr;
  uint32_t crc = 0;
  explicit SolidReader(std::vector<Entry>& e) : es(e) {}
  ~SolidReader() { if (f) fclose(f); }
  // fills buf with up to n bytes; returns the count (0 at the end)
  size_t read(uint8_t* buf, size_t n) {
    size_t got = 0;
    while (got < n && idx < es.size()) {
      Entry& e = es[idx];
      if (e.dir || pos >= e.size) {
        if (!e.dir) e.crc = crc;
        if (f) { fclose(f); f = nullptr; }
        crc = 0;
        ++idx;
        pos = 0;
        continue;
      }
      if (!f) {
        f = open_file(e.source, "rb");
        if (!f) throw zb::Error("cannot read " + zero_utf8(e.source));
      }
      const size_t want = (size_t)std::min<uint64_t>(n - got, e.size - pos);
      const size_t r = fread(buf + got, 1, want, f);
      if (r != want) {
        // the file shrank while being read: the rest is zeros, so the archive stays consistent with its table
        memset(buf + got + r, 0, want - r);
      }
      crc = crc_update_mt(crc, buf + got, want);
      got += want;
      pos += want;
    }
    return got;
  }
  // after the last read: the CRC of a file that ended exactly at the stream's end, the file closed
  void finish() {
    if (idx < es.size() && !es[idx].dir && pos >= es[idx].size) es[idx].crc = crc;
    if (f) { fclose(f); f = nullptr; }
    idx = es.size();
  }
};

inline CompressStats compress_archive(const std::vector<std::string>& inputs, const std::string& out_path, Codec& codec,
                                      uint32_t block_log, Report& rep) {
  if (block_log < zb::MIN_BLOCK_LOG || block_log > zb::MAX_BLOCK_LOG) throw zb::Error("block size must be 64 KiB .. 1 MiB (log2 16..20)");
  const double t0 = zero_now();
  CompressStats st;
  std::vector<Entry> es = collect(inputs, rep);
  uint64_t total = 0;
  for (const Entry& e : es) { total += e.size; if (e.dir) ++st.folders; else ++st.files; }
  st.raw = total;
  rep.say(std::to_string(st.files) + " files, " + std::to_string(st.folders) + " folders, " + std::to_string(total) + " bytes");
  const fs::path outp = zero_path(out_path);
  FILE* f = open_file(outp, "wb");
  if (!f) throw zb::Error("cannot create " + out_path);
  // (an archive left unfinished by an error or Cancel is removed: it could only fail later)
  struct Closer {
    FILE* f;
    fs::path p;
    ~Closer() { if (f) { fclose(f); std::error_code ec; fs::remove(p, ec); } }
  } closer{f, outp};
  const size_t B = (size_t)1 << block_log;
  const uint64_t n_blocks = (total + B - 1) / B;
  // header
  fwrite("ZERO", 1, 4, f);
  w8(f, 1);
  w8(f, (uint8_t)block_log);
  w8(f, 0); w8(f, 0);
  w64(f, total);
  w64(f, n_blocks);
  // blocks
  size_t batch = codec.batch_bytes(block_log);
  batch = (batch / B) * B;
  if (batch < B) batch = B;
  std::vector<uint8_t> buf(batch), next_buf;
  std::vector<uint32_t> crcs, next_crcs;
  SolidReader reader(es);
  uint64_t done = 0;
  std::vector<BlockOut> outs;
  rep.at(0, total);
  const auto fill = [&reader, batch, B](std::vector<uint8_t>& b, std::vector<uint32_t>& c, uint64_t left) {
    if (b.size() < batch) b.resize(batch);
    const size_t n = reader.read(b.data(), std::min<uint64_t>(batch, left));
    c.resize((n + B - 1) / B);
    parallel_for(c.size(), [&](size_t i) { c[i] = crc32::of(b.data() + i * B, std::min(B, n - i * B)); });
    return n;
  };
  size_t n = fill(buf, crcs, total);
  uint64_t queued = n;
  while (n) {
    if (rep.cancelled()) throw Cancelled();
    // the next batch is read and checksummed while the GPU works on this one
    std::future<size_t> ahead;
    if (queued < total) ahead = std::async(std::launch::async, fill, std::ref(next_buf), std::ref(next_crcs), total - queued);
    const double c0 = zero_now();
    codec.compress(buf.data(), n, block_log, outs);
    st.codec_seconds += zero_now() - c0;
    for (size_t b = 0; b < outs.size(); ++b) {
      const size_t off = b * B;
      const uint32_t len = (uint32_t)std::min(B, n - off);
      const uint32_t crc = crcs[b];
      const BlockOut& o = outs[b];
      const bool stored = o.method == 0 || o.data.size() >= len;
      w32(f, len);
      w32(f, stored ? len : (uint32_t)o.data.size());
      w8(f, stored ? 0 : o.method);
      w8(f, 0); w8(f, 0); w8(f, 0);
      w32(f, crc);
      if (stored) { fwrite(buf.data() + off, 1, len, f); ++st.stored_blocks; st.compressed += len; }
      else { fwrite(o.data.data(), 1, o.data.size(), f); st.compressed += o.data.size(); }
      ++st.blocks;
    }
    done += n;
    rep.at(done, total);
    n = 0;
    if (ahead.valid()) {
      n = ahead.get();
      queued += n;
      buf.swap(next_buf);
      crcs.swap(next_crcs);
    }
  }
  reader.finish();
  // table
  std::vector<uint8_t> tbl;
  zb::put32(tbl, (uint32_t)es.size());
  for (const Entry& e : es) {
    tbl.push_back(e.dir ? 1 : 0);
    tbl.push_back(0); tbl.push_back(0); tbl.push_back(0);
    zb::put32(tbl, (uint32_t)e.name.size());
    tbl.insert(tbl.end(), e.name.begin(), e.name.end());
    zb::put32(tbl, (uint32_t)e.size);
    zb::put32(tbl, (uint32_t)(e.size >> 32));
    zb::put32(tbl, (uint32_t)(uint64_t)e.mtime);
    zb::put32(tbl, (uint32_t)((uint64_t)e.mtime >> 32));
    zb::put32(tbl, e.attr);
    zb::put32(tbl, e.crc);
  }
  const uint64_t table_off = 24 + st.compressed + st.blocks * 16;
  fwrite(tbl.data(), 1, tbl.size(), f);
  w64(f, table_off);
  w64(f, tbl.size());
  w32(f, crc32::of(tbl.data(), tbl.size()));
  fwrite("ZEND", 1, 4, f);
  if (fflush(f) != 0 || ferror(f)) throw zb::Error("writing " + out_path + " failed (disk full?)");
  fclose(f);
  closer.f = nullptr;
  st.seconds = zero_now() - t0;
  return st;
}

// ---- reading
struct ArchiveInfo {
  uint32_t block_log = 0;
  uint64_t total_raw = 0, n_blocks = 0, table_off = 0, file_size = 0;
  std::vector<Entry> entries;
};

inline uint64_t r64(const uint8_t* p) { return (uint64_t)zb::get32(p) | (uint64_t)zb::get32(p + 4) << 32; }

inline ArchiveInfo read_info(FILE* f) {
  ArchiveInfo a;
  uint8_t h[24];
  if (fread(h, 1, 24, f) != 24 || memcmp(h, "ZERO", 4) != 0) throw zb::Error("not a .zero archive");
  if (h[4] != 1) throw zb::Error("archive version " + std::to_string(h[4]) + " is newer than this program");
  a.block_log = h[5];
  a.total_raw = r64(h + 8);
  a.n_blocks = r64(h + 16);
  if (a.block_log < zb::MIN_BLOCK_LOG || a.block_log > zb::MAX_BLOCK_LOG) throw zb::Error("archive damaged: block size");
  if (zseek(f, 0, SEEK_END) != 0) throw zb::Error("cannot seek");
  const int64_t size = ztell(f);
  if (size < 48) throw zb::Error("archive damaged: too short");
  a.file_size = (uint64_t)size;
  uint8_t ft[24];
  zseek(f, size - 24, SEEK_SET);
  if (fread(ft, 1, 24, f) != 24 || memcmp(ft + 20, "ZEND", 4) != 0) throw zb::Error("archive damaged: footer (the file was cut short?)");
  a.table_off = r64(ft);
  const uint64_t tl = r64(ft + 8);
  const uint32_t tc = zb::get32(ft + 16);
  if (a.table_off + tl + 24 != a.file_size) throw zb::Error("archive damaged: table position");
  std::vector<uint8_t> tbl(tl);
  zseek(f, (int64_t)a.table_off, SEEK_SET);
  if (fread(tbl.data(), 1, tl, f) != tl) throw zb::Error("archive damaged: table cut");
  if (crc32::of(tbl.data(), tl) != tc) throw zb::Error("archive damaged: table checksum");
  if (tl < 4) throw zb::Error("archive damaged: table");
  const uint8_t* p = tbl.data();
  const uint8_t* end = p + tl;
  const uint32_t n = zb::get32(p);
  p += 4;
  uint64_t sum = 0;
  for (uint32_t i = 0; i < n; ++i) {
    if (end - p < 8) throw zb::Error("archive damaged: entry");
    Entry e;
    e.dir = p[0] == 1;
    const uint32_t nl = zb::get32(p + 4);
    p += 8;
    if ((size_t)(end - p) < nl + 24) throw zb::Error("archive damaged: entry name");
    e.name.assign((const char*)p, nl);
    p += nl;
    e.size = r64(p);
    e.mtime = (int64_t)r64(p + 8);
    e.attr = zb::get32(p + 16);
    e.crc = zb::get32(p + 20);
    p += 24;
    if (!e.dir) sum += e.size;
    a.entries.push_back(std::move(e));
  }
  if (sum != a.total_raw) throw zb::Error("archive damaged: sizes");
  return a;
}

inline ArchiveInfo list_archive(const std::string& path) {
  FILE* f = open_file(zero_path(path), "rb");
  if (!f) throw zb::Error("cannot open " + path);
  struct Closer { FILE* f; ~Closer() { fclose(f); } } c{f};
  return read_info(f);
}

// a name from an archive is only ever written under the target folder
inline bool safe_name(const std::string& n) {
  if (n.empty() || n[0] == '/' || n.find('\\') != std::string::npos || n.find(':') != std::string::npos) return false;
  size_t i = 0;
  while (i <= n.size()) {
    size_t j = n.find('/', i);
    if (j == std::string::npos) j = n.size();
    const std::string part = n.substr(i, j - i);
    if (part.empty() || part == "." || part == "..") return false;
    i = j + 1;
  }
  return true;
}

// ZERO_PROFILE=1: where extraction's wall time goes
struct IoProfile {
  double read = 0, crc_blocks = 0, fwrite = 0, crc_file = 0, close = 0, mtime = 0, open = 0;
  static IoProfile& get() { static IoProfile p; return p; }
};
struct ExtractStats {
  uint64_t raw = 0, compressed = 0, blocks = 0;
  size_t files = 0, folders = 0;
  double seconds = 0, codec_seconds = 0;
  bool verified = true;
};

// Decompresses every block in order and hands the solid stream to `sink(bytes, n)`; verifies every block's CRC.
inline ExtractStats read_blocks(const std::string& path, Codec& codec, Report& rep, ArchiveInfo& a,
                                const std::function<void(const uint8_t*, size_t)>& sink) {
  const double t0 = zero_now();
  ExtractStats st;
  FILE* f = open_file(zero_path(path), "rb");
  if (!f) throw zb::Error("cannot open " + path);
  struct Closer { FILE* f; ~Closer() { fclose(f); } } c{f};
  a = read_info(f);
  codec.prepare(a.block_log);
  zseek(f, 24, SEEK_SET);
  const size_t B = (size_t)1 << a.block_log;
  size_t batch = codec.batch_bytes_decode(a.block_log);
  batch = std::max(B, (batch / B) * B);
  const size_t per_batch = batch / B;
  std::vector<uint8_t> comp;
  std::vector<BlockIn> ins;
  std::vector<uint32_t> crcs;
  uint64_t done = 0, blocks_left = a.n_blocks;
  uint64_t pos = 24;
  rep.at(0, a.total_raw);
  while (blocks_left > 0) {
    if (rep.cancelled()) throw Cancelled();
    const double t_batch = zero_now();
    const size_t nb = (size_t)std::min<uint64_t>(per_batch, blocks_left);
    // the batch's block headers and payloads, read into comp
    comp.clear();
    ins.clear();
    crcs.clear();
    std::vector<size_t> offs;
    size_t raw_total = 0;
    for (size_t b = 0; b < nb; ++b) {
      uint8_t h[16];
      if (fread(h, 1, 16, f) != 16) throw zb::Error("archive damaged: block header cut");
      pos += 16;
      BlockIn in;
      in.raw_len = zb::get32(h);
      in.comp_len = zb::get32(h + 4);
      in.method = h[8];
      if (in.raw_len > B || (in.method == 0 && in.comp_len != in.raw_len) || in.method > 1) throw zb::Error("archive damaged: block header");
      if (pos + in.comp_len > a.table_off) throw zb::Error("archive damaged: block past the table");
      offs.push_back(comp.size());
      comp.resize(comp.size() + in.comp_len);
      if (fread(comp.data() + offs.back(), 1, in.comp_len, f) != in.comp_len) throw zb::Error("archive damaged: block cut");
      pos += in.comp_len;
      crcs.push_back(zb::get32(h + 12));
      raw_total += in.raw_len;
      st.compressed += in.comp_len;
      ins.push_back(in);
    }
    for (size_t b = 0; b < nb; ++b) ins[b].data = comp.data() + offs[b];
    const double c0 = zero_now();
    IoProfile::get().read += c0 - t_batch;
    const uint8_t* raw = codec.decompress(ins);
    st.codec_seconds += zero_now() - c0;
    const double c1 = zero_now();
    std::vector<size_t> raw_off(nb);
    for (size_t b = 0, o = 0; b < nb; o += ins[b].raw_len, ++b) raw_off[b] = o;
    std::vector<uint8_t> crc_ok(nb);
    parallel_for(nb, [&](size_t b) { crc_ok[b] = crc32::of(raw + raw_off[b], ins[b].raw_len) == crcs[b]; });
    for (size_t b = 0; b < nb; ++b) {
      if (!crc_ok[b]) {
        st.verified = false;
        throw zb::Error("block " + std::to_string(st.blocks + b) + " does not match its checksum: the archive is damaged");
      }
    }
    IoProfile::get().crc_blocks += zero_now() - c1;
    sink(raw, raw_total);
    done += raw_total;
    blocks_left -= nb;
    st.blocks += nb;
    rep.at(done, a.total_raw);
  }
  if (done != a.total_raw) throw zb::Error("archive damaged: the blocks hold " + std::to_string(done) + " bytes, the table " + std::to_string(a.total_raw));
  st.raw = done;
  st.seconds = zero_now() - t0;
  return st;
}

// Writes the solid stream back into files under `dir`, folders made as needed.
struct SolidWriter {
  const std::vector<Entry>& es;
  fs::path root;
  size_t idx = 0;
  uint64_t pos = 0;
  FILE* f = nullptr;
  uint32_t crc = 0;
  Report& rep;
  size_t files = 0, folders = 0;
  fs::path last_parent;
  SolidWriter(const std::vector<Entry>& e, const fs::path& r, Report& rp) : es(e), root(r), rep(rp) {}
  ~SolidWriter() { if (f) fclose(f); }
  void open_next() {
    const Entry& e = es[idx];
    if (!safe_name(e.name)) throw zb::Error("refused: unsafe name in the archive: " + e.name);
    const fs::path p = root / fs::path(e.name);
    const double t0 = zero_now();
    if (p.parent_path() != last_parent) {  // (a folder is checked once, not again for every file in it)
      std::error_code ec;
      fs::create_directories(p.parent_path(), ec);
      last_parent = p.parent_path();
    }
    f = open_file(p, "wb");
    IoProfile::get().open += zero_now() - t0;
    if (!f) throw zb::Error("cannot create " + zero_utf8(p));
    ++files;
  }
  void finish_file() {
    const Entry& e = es[idx];
    const double t0 = zero_now();
    if (f) { fclose(f); f = nullptr; }
    const double t1 = zero_now();
    IoProfile::get().close += t1 - t0;
    if (!e.dir && crc != e.crc && e.size > 0) throw zb::Error("file " + e.name + " does not match its checksum");
    set_mtime(root / fs::path(e.name), e.mtime);
    IoProfile::get().mtime += zero_now() - t1;
    crc = 0;
  }
  // advances over folders and empty files
  void settle() {
    while (idx < es.size()) {
      const Entry& e = es[idx];
      if (e.dir) {
        if (!safe_name(e.name)) throw zb::Error("refused: unsafe name in the archive: " + e.name);
        std::error_code ec;
        fs::create_directories(root / fs::path(e.name), ec);
        ++folders;
        ++idx;
        continue;
      }
      if (e.size == 0) { open_next(); finish_file(); ++idx; continue; }
      if (pos >= e.size) { finish_file(); ++idx; pos = 0; continue; }
      return;
    }
  }
  void write(const uint8_t* p, size_t n) {
    settle();
    while (n > 0) {
      if (idx >= es.size()) throw zb::Error("archive damaged: more data than the table lists");
      const Entry& e = es[idx];
      if (!f) open_next();
      const size_t want = (size_t)std::min<uint64_t>(n, e.size - pos);
      const double t0 = zero_now();
      if (fwrite(p, 1, want, f) != want) throw zb::Error("writing " + e.name + " failed (disk full?)");
      const double t1 = zero_now();
      crc = crc_update_mt(crc, p, want);
      IoProfile::get().fwrite += t1 - t0;
      IoProfile::get().crc_file += zero_now() - t1;
      p += want;
      n -= want;
      pos += want;
      settle();
    }
  }
  void close() {
    settle();
    // (folders' times last: writing files into them changed them)
    for (const Entry& e : es)
      if (e.dir) set_mtime(root / fs::path(e.name), e.mtime);
  }
};

inline ExtractStats extract_archive(const std::string& path, const std::string& out_dir, Codec& codec, Report& rep) {
  ArchiveInfo a;
  const fs::path root = zero_path(out_dir);
  std::error_code ec;
  fs::create_directories(root, ec);
  std::unique_ptr<SolidWriter> w;
  ExtractStats st = read_blocks(path, codec, rep, a, [&](const uint8_t* p, size_t n) {
    if (!w) w = std::make_unique<SolidWriter>(a.entries, root, rep);
    w->write(p, n);
  });
  if (!w) w = std::make_unique<SolidWriter>(a.entries, root, rep);
  w->close();
  if (getenv("ZERO_PROFILE")) {
    const IoProfile& q = IoProfile::get();
    fprintf(stderr, "[profile] extract (s): read %.3f codec %.3f crc-blocks %.3f open %.3f fwrite %.3f crc-file %.3f close %.3f mtime %.3f\n",
            q.read, st.codec_seconds, q.crc_blocks, q.open, q.fwrite, q.crc_file, q.close, q.mtime);
  }
  st.files = w->files;
  st.folders = w->folders;
  return st;
}

// decompresses everything and checks every block and file checksum, writing nothing
inline ExtractStats test_archive(const std::string& path, Codec& codec, Report& rep) {
  ArchiveInfo a;
  size_t idx = 0;
  uint64_t pos = 0;
  uint32_t crc = 0;
  size_t files = 0;
  auto settle = [&] {
    while (idx < a.entries.size()) {
      const Entry& e = a.entries[idx];
      if (e.dir) { ++idx; continue; }
      if (pos >= e.size) {
        if (e.size > 0 && crc != e.crc) throw zb::Error("file " + e.name + " does not match its checksum");
        ++files;
        ++idx;
        pos = 0;
        crc = 0;
        continue;
      }
      return;
    }
  };
  ExtractStats st = read_blocks(path, codec, rep, a, [&](const uint8_t* p, size_t n) {
    settle();
    while (n > 0) {
      if (idx >= a.entries.size()) throw zb::Error("archive damaged: more data than the table lists");
      const Entry& e = a.entries[idx];
      const size_t want = (size_t)std::min<uint64_t>(n, e.size - pos);
      crc = crc_update_mt(crc, p, want);
      p += want;
      n -= want;
      pos += want;
      settle();
    }
  });
  settle();
  st.files = files;
  for (const Entry& e : a.entries) if (e.dir) ++st.folders;
  return st;
}

}  // namespace zero
