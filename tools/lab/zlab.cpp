// zlab: tries block format ideas on the CPU, one at a time, and reports what each costs in size and what it buys in GPU
// parallelism. No GPU is used, and the process runs at idle priority. Decode speed is not measured here: it can only be
// measured on the GPU. Sizes come from the real rANS coder (src/codec.h), except where a line says "est".
//
//   zlab <corpus name>=<file> ... [--block 19] [--only name,name] [--threads N]
//
// The rules a parse can follow, so that a GPU can decode without waiting for its own writes:
//   K (token groups)  K tokens decode side by side, one per lane. A match may read only output from before its group,
//                     or the literals of its own token (which come from the literal stream, not from the output).
//   G (byte schedule) output is fenced every G bytes. A match may read only below the last fence, or its own literals.
#include "../../src/codec.h"
#include <atomic>
#include <chrono>
#include <cmath>
#include <functional>
#include <cstdio>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#endif

struct Cfg {
  std::string name;
  uint32_t minm = 4;
  int depth = 32;
  int lazy = 0;       // positions looked ahead before taking a match
  uint32_t K = 0;     // tokens per group (0: no rule)
  uint32_t G = 0;     // bytes per fence (0: no rule)
  int rep = 0;        // recent offsets kept (0: none)
  int hbits = 16;
  uint32_t hlen = 4;  // bytes hashed
  uint32_t block_log = 0;  // (0: the command line's)
  int slot = 0;       // top bits of an offset's extra bits coded with its bit length
  int align = 0;      // low bits of an offset coded in a stream of their own
  bool rep_recode = false;  // recent offsets used only when the parse happened to pick one (a GPU's parse cannot look)
  bool rep_sync = false;    // recent offsets change only at group boundaries (lanes of a group need no earlier lane)
  bool split_len = false;   // literal and match lengths in two streams instead of one token byte
  uint32_t flex = 0;        // a group closes early when that makes a match this many bytes longer (4 bits per group)
  bool raw_tok = false, raw_sym = false;  // these streams without rANS: less decoding, larger
  // the design a GPU can build: parse with no rule (segments in parallel), then one pass per block that makes groups
  bool post = false;
  uint32_t post_k = 32;     // tokens per group at most
  uint32_t post_t = 0;      // a match losing less than this many bytes is cut to what is legal; otherwise the group closes
  uint32_t maxm = 274;      // the longest match the parse may take
};

struct Stats {
  uint64_t raw = 0, out = 0, stored = 0;
  uint64_t lit = 0, tok = 0, ext = 0, sym = 0, bits = 0;  // bytes per stream, coded
  uint64_t tokens = 0, literals = 0, matched = 0, fences = 0, short_tokens = 0, rep_hits = 0, groups = 0;
  double lit_o0 = 0, lit_o1 = 0;  // Shannon bits of the literals, order 0 and order 1 on the previous literal's high bits
  void add(const Stats& o) {
    raw += o.raw; out += o.out; stored += o.stored; lit += o.lit; tok += o.tok; ext += o.ext; sym += o.sym; bits += o.bits;
    tokens += o.tokens; literals += o.literals; matched += o.matched; fences += o.fences; short_tokens += o.short_tokens; rep_hits += o.rep_hits; groups += o.groups;
    lit_o0 += o.lit_o0; lit_o1 += o.lit_o1;
  }
};

static double entropy_bits(const uint32_t* c, int n) {
  uint64_t t = 0;
  for (int i = 0; i < n; ++i) t += c[i];
  if (!t) return 0;
  double b = 0;
  for (int i = 0; i < n; ++i) if (c[i]) b -= c[i] * std::log2((double)c[i] / t);
  return b;
}

static inline uint32_t bitlen(uint32_t d) {  // the b of D in [2^b, 2^(b+1))
  uint32_t b = 0;
  while ((d >> (b + 1)) != 0) ++b;
  return b;
}

struct Tok { uint32_t op, i, m, d; };
static void compress_block_post(const uint8_t* in, uint32_t n, const Cfg& c, Stats& st);

static void compress_block(const uint8_t* in, uint32_t n, const Cfg& c, Stats& st, std::vector<Tok>* cap = nullptr) {
  if (c.post && !cap) { compress_block_post(in, n, c, st); return; }
  st.raw += n;
  const uint32_t HB = 1u << c.hbits;
  std::vector<int32_t> head(HB, -1), prev(n, -1);
  const auto hash = [&](uint32_t i) -> uint32_t {
    uint64_t v = 0;
    for (uint32_t k = 0; k < c.hlen; ++k) v |= (uint64_t)in[i + k] << (8 * k);
    if (c.hlen <= 4) return ((uint32_t)v * 2654435761u) >> (32 - c.hbits);
    return (uint32_t)((v * 0x9E3779B97F4A7C15ull) >> (64 - c.hbits));
  };
  uint32_t inserted = 0;
  const auto insert_to = [&](uint32_t end) {
    for (; inserted < end; ++inserted) {
      if (inserted + c.hlen > n) continue;
      const uint32_t h = hash(inserted);
      prev[inserted] = head[h];
      head[h] = (int32_t)inserted;
    }
  };
  std::vector<uint8_t> lit, tok, ext, sym, bits, alb, tok2;
  uint32_t group_recent[3] = {1, 4, 8};
  uint32_t gtok = 0;  // tokens in the open group
  uint32_t last_group = 0;
  uint64_t bitbuf = 0;
  int bitn = 0;
  uint32_t lit_start = 0, tokens = 0, group_start = 0;
  uint32_t recent[3] = {1, 4, 8};
  uint32_t fenced = 0;
  const uint32_t MAXM = c.maxm;

  // how far a match of j at position i may run, under the rules
  const auto limit = [&](uint32_t i, uint32_t j, uint32_t op) -> uint32_t {
    uint32_t lim = std::min(MAXM, n - i);
    if (j >= op) return lim;                 // (its own literals: from the literal stream)
    if (c.K) { if (j >= group_start) return 0; lim = std::min(lim, group_start - j); }
    if (c.G) { const uint32_t f = i / c.G * c.G; if (j >= f) return 0; lim = std::min(lim, f - j); }
    return lim;
  };
  const auto length = [&](uint32_t i, uint32_t j, uint32_t lim) -> uint32_t {
    uint32_t l = 0;
    while (l < lim && in[j + l] == in[i + l]) ++l;
    return l;
  };
  struct M { uint32_t len, dist; int rep; };
  // the best match at i: by length, a recent offset preferred because it costs no offset bits
  const auto best_at = [&](uint32_t i, uint32_t op) -> M {
    M b{0, 0, -1};
    if (i + c.hlen > n) return b;
    double best_score = 0;
    for (int r = 0; r < c.rep && !c.rep_recode; ++r) {
      const uint32_t d = c.rep_sync ? group_recent[r] : recent[r];
      if (d == 0 || d > i) continue;
      const uint32_t l = length(i, i - d, limit(i, i - d, op));
      if (l >= c.minm) {
        const double s = 8.0 * l;  // no offset bits
        if (s > best_score) { best_score = s; b = M{l, d, r}; }
      }
    }
    insert_to(i);
    int32_t j = head[hash(i)];
    for (int k = 0; k < c.depth && j >= 0; ++k, j = prev[j]) {
      const uint32_t lim = limit(i, (uint32_t)j, op);
      if (lim < c.minm) continue;
      if (b.len && in[(uint32_t)j + std::min(b.len, lim - 1)] != in[i + std::min(b.len, lim - 1)]) continue;
      const uint32_t l = length(i, (uint32_t)j, lim);
      if (l < c.minm) continue;
      const uint32_t d = i - (uint32_t)j;
      const double s = c.rep ? 8.0 * l - (bitlen(d) + 3) : 8.0 * l;
      if (s > best_score) { best_score = s; b = M{l, d, -1}; }
    }
    return b;
  };
  const auto emit = [&](uint32_t i, const M& m0) {
    M m = m0;
    if (cap) { cap->push_back(Tok{lit_start, i, m.len, m.dist}); lit_start = i + m.len; return; }
    if (c.K && gtok == 0) { group_start = lit_start; memcpy(group_recent, recent, sizeof recent); ++st.groups; }
    if (c.rep_recode && m.rep < 0)
      for (int r = 0; r < c.rep; ++r) if ((c.rep_sync ? group_recent[r] : recent[r]) == m.dist) { m.rep = r; break; }
    uint32_t L = i - lit_start;
    lit.insert(lit.end(), in + lit_start, in + i);
    const uint32_t Mb = m.len - 4 >= 15 ? 15 : (m.len >= 4 ? m.len - 4 : 0);
    if (c.split_len) { tok.push_back((uint8_t)std::min(L, 15u)); tok2.push_back((uint8_t)Mb); }
    else tok.push_back((uint8_t)((std::min(L, 15u) << 4) | Mb));
    if (L >= 15) { uint32_t x = L - 15; while (x >= 255) { ext.push_back(255); x -= 255; } ext.push_back((uint8_t)x); }
    if (m.len - 4 >= 15 && m.len >= 4) ext.push_back((uint8_t)(m.len - 4 - 15));
    if (m.rep >= 0) {
      sym.push_back((uint8_t)(250 + m.rep));
      ++st.rep_hits;
    } else {
      const uint32_t b = bitlen(m.dist);
      uint32_t extra = m.dist - (1u << b), nb = b;
      // the top bits go with the symbol, the low ones in their own stream, the rest stays raw
      const uint32_t top = (uint32_t)std::min<int>(c.slot, (int)nb);
      const uint32_t tv = top ? extra >> (nb - top) : 0;
      nb -= top;
      extra &= nb < 32 ? (1u << nb) - 1 : ~0u;
      sym.push_back((uint8_t)(((b << c.slot) | (tv << (c.slot - top))) & 0xFF));
      const uint32_t low = (uint32_t)std::min<int>(c.align, (int)nb);
      if (c.align) { alb.push_back((uint8_t)(extra & ((1u << low) - 1))); extra >>= low; nb -= low; }
      bitbuf |= (uint64_t)extra << bitn;
      bitn += (int)nb;
      while (bitn >= 8) { bits.push_back((uint8_t)bitbuf); bitbuf >>= 8; bitn -= 8; }
    }
    if (c.rep) {
      if (m.rep != 0) {
        const int from = m.rep < 0 ? c.rep - 1 : m.rep;
        (void)last_group;
        for (int r = from; r > 0; --r) recent[r] = recent[r - 1];
        recent[0] = m.dist;
      }
    }
    // what the decoder of today does: a fence before a match that reads what was written since the last one
    if (i - m.dist + std::min(m.len, m.dist) > fenced) { ++st.fences; fenced = i; }
    ++tokens;
    if (c.K && ++gtok == c.K) gtok = 0;
    st.matched += m.len;
    if (L + m.len <= 16) ++st.short_tokens;
    lit_start = i + m.len;
  };
  uint32_t i = 0;
  while (i < n) {
    if (c.K && gtok == 0) { group_start = lit_start; memcpy(group_recent, recent, sizeof recent); }
    M m = best_at(i, lit_start);
    if (c.flex && gtok > 0) {
      const uint32_t keep = group_start;
      uint32_t keep_r[3];
      memcpy(keep_r, group_recent, sizeof keep_r);
      group_start = lit_start;
      memcpy(group_recent, recent, sizeof recent);
      const M u = best_at(i, lit_start);
      if (u.len >= c.minm && u.len >= (m.len >= c.minm ? m.len : 0) + c.flex) { gtok = 0; m = u; }
      else { group_start = keep; memcpy(group_recent, keep_r, sizeof keep_r); }
    }
    if (m.len < c.minm) { ++i; continue; }
    for (int la = 0; la < c.lazy && i + 1 < n; ++la) {
      const M m2 = best_at(i + 1, lit_start);
      const double s1 = 8.0 * m.len - (m.rep >= 0 ? 0 : bitlen(m.dist));
      const double s2 = 8.0 * m2.len - (m2.rep >= 0 ? 0 : bitlen(m2.dist)) - 8;
      if (m2.len >= c.minm && s2 > s1) { ++i; m = m2; } else break;
    }
    emit(i, m);
    i += m.len;
  }
  if (cap) return;
  lit.insert(lit.end(), in + lit_start, in + n);
  if (bitn) bits.push_back((uint8_t)bitbuf);
  st.tokens += tokens;
  st.literals += lit.size();
  // literals: Shannon, order 0 and order 1 on the previous literal's high 4 bits (estimates for the second)
  {
    uint32_t c0[256] = {};
    static thread_local uint32_t c1[16][256];
    memset(c1, 0, sizeof c1);
    uint8_t p = 0;
    for (uint8_t x : lit) { ++c0[x]; ++c1[p >> 4][x]; p = x; }
    st.lit_o0 += entropy_bits(c0, 256);
    for (int k = 0; k < 16; ++k) st.lit_o1 += entropy_bits(c1[k], 256);
  }
  std::vector<uint8_t> a, b, d, e, f, h;
  zb::encode_stream(alb.data(), (uint32_t)alb.size(), f);
  zb::encode_stream(tok2.data(), (uint32_t)tok2.size(), h);
  zb::encode_stream(lit.data(), (uint32_t)lit.size(), a);
  zb::encode_stream(tok.data(), (uint32_t)tok.size(), b);
  zb::encode_stream(ext.data(), (uint32_t)ext.size(), d);
  zb::encode_stream(sym.data(), (uint32_t)sym.size(), e);
  if (c.raw_tok) b.assign(tok.size(), 0);
  if (c.raw_sym) e.assign(sym.size() * 5 / 8 + 1, 0);  // (5 bits per symbol)
  const uint64_t total = 16 + a.size() + b.size() + d.size() + e.size() + bits.size() + (c.align ? f.size() : 0) + (c.split_len ? h.size() : 0) + (c.flex ? tokens / 2 / std::max(1u, c.K) : 0);
  if (total >= n) { st.out += n + 16; ++st.stored; return; }
  st.out += total + 16;
  st.lit += a.size(); st.tok += b.size() + (c.split_len ? h.size() : 0); st.ext += d.size(); st.sym += e.size() + (c.align ? f.size() : 0); st.bits += bits.size();
}

// Parse with no rule, then in one pass per block: tokens gather in groups that decode side by side. A match whose source
// is in its own group either is cut to its legal part (the rest becomes literals of the next token, so no position moves)
// or closes the group. This is exactly what the format and a GPU encoder would do. Lengths past 15 are u16, so a lane
// finds its own by counting flags, and a match can be long. An offset equal to one of the last three of the group before
// costs no bits: lanes need nothing from each other.
static void compress_block_post(const uint8_t* in, uint32_t n, const Cfg& c, Stats& st) {
  std::vector<Tok> tk;
  Stats ignore;
  compress_block(in, n, c, ignore, &tk);
  st.raw += n;
  std::vector<uint8_t> lit, tok, ext, sym, bits, groups;
  lit.reserve(n);
  uint64_t bitbuf = 0;
  int bitn = 0;
  uint32_t gstart = 0, gt = 0, carry_op = 0;
  uint32_t recent[3] = {1, 4, 8}, last3[3] = {1, 4, 8};
  uint32_t emitted = 0, gcount = 0;
  uint64_t matched_here = 0;  // (literals + matched must come to the whole block: checked at the end)
  const auto close = [&](uint32_t op) {
    if (gt) { groups.push_back((uint8_t)gt); ++gcount; }
    gt = 0;
    gstart = op;
    memcpy(recent, last3, sizeof recent);
  };
  const auto put16 = [&](uint32_t v) { ext.push_back((uint8_t)v); ext.push_back((uint8_t)(v >> 8)); };
  for (size_t x = 0; x < tk.size(); ++x) {
    // (op can be before the parse's start of this token: the bytes of a match that was cut or dropped are literals here)
    const uint32_t op = carry_op, i = tk[x].i;
    uint32_t m = tk[x].m;
    const uint32_t d = tk[x].d;
    if (gt == c.post_k) close(op);
    const uint32_t j = i - d;
    if (j < op && j + std::min(m, d) > gstart) {     // reads what its own group writes
      const uint32_t legal = d >= m && gstart > j ? gstart - j : 0;  // (an overlapping match cannot be cut this way)
      const uint32_t use = legal >= 4 ? legal : 0;
      if (m - use < c.post_t) {
        if (use == 0) { carry_op = op; continue; }   // (its bytes become literals of the next token)
        m = use;
      } else {
        close(op);
      }
    }
    uint32_t L = i - op;
    lit.insert(lit.end(), in + op, in + i);
    tok.push_back((uint8_t)((std::min(L, 15u) << 4) | std::min(m - 4, 15u)));
    if (L >= 15) put16(std::min(L - 15, 65535u));    // (longer runs are rare enough to count as if they fit)
    if (m - 4 >= 15) put16(std::min(m - 19, 65535u));
    int rep = -1;
    for (int r = 0; r < 3; ++r) if (recent[r] == d) { rep = r; break; }
    if (rep >= 0) { sym.push_back((uint8_t)(250 + rep)); ++st.rep_hits; }
    else {
      const uint32_t b = bitlen(d);
      sym.push_back((uint8_t)b);
      bitbuf |= (uint64_t)(d - (1u << b)) << bitn;
      bitn += (int)b;
      while (bitn >= 8) { bits.push_back((uint8_t)bitbuf); bitbuf >>= 8; bitn -= 8; }
    }
    last3[2] = last3[1]; last3[1] = last3[0]; last3[0] = d;
    ++gt; ++emitted;
    matched_here += m;
    st.matched += m;
    if (L + m <= 16) ++st.short_tokens;
    carry_op = i + m;
  }
  if (gt) { groups.push_back((uint8_t)gt); ++gcount; }
  lit.insert(lit.end(), in + carry_op, in + n);
  if (bitn) bits.push_back((uint8_t)bitbuf);
  if (lit.size() + matched_here != n) { fprintf(stderr, "lost bytes: %zu + %llu != %u\n", lit.size(), (unsigned long long)matched_here, n); exit(3); }
  st.tokens += emitted;
  st.groups += gcount;
  st.fences += gcount;
  st.literals += lit.size();
  std::vector<uint8_t> a, b, d2, e, g;
  zb::encode_stream(lit.data(), (uint32_t)lit.size(), a);
  zb::encode_stream(tok.data(), (uint32_t)tok.size(), b);
  zb::encode_stream(ext.data(), (uint32_t)ext.size(), d2);
  zb::encode_stream(sym.data(), (uint32_t)sym.size(), e);
  zb::encode_stream(groups.data(), (uint32_t)groups.size(), g);
  const uint64_t total = 16 + a.size() + b.size() + d2.size() + e.size() + g.size() + bits.size();
  if (total >= n) { st.out += n + 16; ++st.stored; return; }
  st.out += total + 16;
  st.lit += a.size(); st.tok += b.size() + g.size(); st.ext += d2.size(); st.sym += e.size(); st.bits += bits.size();
}

int main(int argc, char** argv) {
#ifdef _WIN32
  SetPriorityClass(GetCurrentProcess(), IDLE_PRIORITY_CLASS);  // (nothing else on this machine should notice)
#endif
  std::vector<std::pair<std::string, std::string>> corpora;
  uint32_t block_log = 19;
  unsigned threads = std::max(1u, std::thread::hardware_concurrency() / 3);
  std::string only;
  for (int i = 1; i < argc; ++i) {
    const std::string s = argv[i];
    if (s == "--block" && i + 1 < argc) block_log = (uint32_t)atoi(argv[++i]);
    else if (s == "--threads" && i + 1 < argc) threads = (unsigned)atoi(argv[++i]);
    else if (s == "--only" && i + 1 < argc) only = "," + std::string(argv[++i]) + ",";
    else { const size_t e = s.find('='); if (e != std::string::npos) corpora.push_back({s.substr(0, e), s.substr(e + 1)}); }
  }
  std::vector<Cfg> cfgs;
  const auto add = [&](const char* name, const std::function<void(Cfg&)>& f) { Cfg c; c.name = name; f(c); cfgs.push_back(c); };
  // 0. where it stands
  add("base", [](Cfg&) {});
  // 1. the parse
  add("lazy1", [](Cfg& c) { c.lazy = 1; });
  add("lazy2", [](Cfg& c) { c.lazy = 2; });
  add("depth8", [](Cfg& c) { c.depth = 8; });
  add("depth128", [](Cfg& c) { c.depth = 128; });
  // 2. what a match is
  add("min3", [](Cfg& c) { c.minm = 4; c.hlen = 3; });
  add("hash5", [](Cfg& c) { c.hlen = 5; c.minm = 5; });
  add("hash6", [](Cfg& c) { c.hlen = 6; c.minm = 6; });
  // 3. recent offsets, which cost no offset bits
  add("rep1", [](Cfg& c) { c.rep = 1; });
  add("rep3", [](Cfg& c) { c.rep = 3; });
  add("rep3+lazy1", [](Cfg& c) { c.rep = 3; c.lazy = 1; });
  add("rep3+lazy2", [](Cfg& c) { c.rep = 3; c.lazy = 2; });
  // 4. token groups: K tokens decode side by side
  for (uint32_t k : {2u, 4u, 8u, 16u, 32u}) {
    static std::vector<std::string> names;
    names.push_back("K" + std::to_string(k));
    add(names.back().c_str(), [k](Cfg& c) { c.K = k; });
  }
  add("K16+lazy1", [](Cfg& c) { c.K = 16; c.lazy = 1; });
  add("K16+rep3+lazy1", [](Cfg& c) { c.K = 16; c.rep = 3; c.lazy = 1; });
  // 5. a fixed fence schedule
  for (uint32_t g : {64u, 256u, 1024u, 4096u}) {
    static std::vector<std::string> names;
    names.push_back("G" + std::to_string(g));
    add(names.back().c_str(), [g](Cfg& c) { c.G = g; });
  }
  // 7. how offsets are coded
  add("slot1", [](Cfg& c) { c.slot = 1; });
  add("slot2", [](Cfg& c) { c.slot = 2; });
  add("slot3", [](Cfg& c) { c.slot = 3; });
  add("align2", [](Cfg& c) { c.align = 2; });
  add("align4", [](Cfg& c) { c.align = 4; });
  add("slot2+align4", [](Cfg& c) { c.slot = 2; c.align = 4; });
  add("splitlen", [](Cfg& c) { c.split_len = true; });
  // 8. recent offsets a GPU can have
  add("recode3", [](Cfg& c) { c.rep = 3; c.rep_recode = true; });
  add("recode3+lazy1", [](Cfg& c) { c.rep = 3; c.rep_recode = true; c.lazy = 1; });
  add("K16+sync3+lazy1", [](Cfg& c) { c.K = 16; c.rep = 3; c.rep_sync = true; c.lazy = 1; });
  add("K16+recode3+lazy1", [](Cfg& c) { c.K = 16; c.rep = 3; c.rep_recode = true; c.rep_sync = true; c.lazy = 1; });
  // 9. together, and with no rule as the reference
  for (uint32_t k : {0u, 8u, 16u, 32u})
    for (uint32_t bl : {19u, 20u}) {
      static std::vector<std::string> names;
      names.push_back("all_K" + std::to_string(k) + "_b" + std::to_string(bl));
      add(names.back().c_str(), [k, bl](Cfg& c) { c.K = k; c.block_log = bl; c.rep = 3; c.rep_sync = k != 0; c.lazy = 1; c.slot = 2; c.align = 4; });
    }
  // 10. the same with what a GPU's parse can do today: few candidates, recent offsets only recoded
  for (uint32_t k : {8u, 16u, 32u}) {
    static std::vector<std::string> names;
    names.push_back("gpu_K" + std::to_string(k));
    add(names.back().c_str(), [k](Cfg& c) { c.K = k; c.block_log = 20; c.rep = 3; c.rep_sync = true; c.rep_recode = true; c.lazy = 1; c.slot = 2; c.align = 4; c.depth = 8; });
  }
  // 11. what a GPU's parse can do, the reference for what follows
  const auto gpu_base = [](Cfg& c) { c.block_log = 20; c.rep = 3; c.rep_sync = true; c.rep_recode = true; c.lazy = 1; c.depth = 8; };
  add("today_gpu", [](Cfg& c) { c.depth = 8; });
  for (uint32_t k : {4u, 8u, 16u, 32u}) {
    static std::vector<std::string> names;
    names.push_back("g_K" + std::to_string(k));
    add(names.back().c_str(), [=](Cfg& c) { gpu_base(c); c.K = k; });
    for (uint32_t t : {2u, 4u, 8u, 16u}) {
      names.push_back("g_K" + std::to_string(k) + "_flex" + std::to_string(t));
      add(names.back().c_str(), [=](Cfg& c) { gpu_base(c); c.K = k; c.flex = t; });
    }
  }
  add("g_K16_b19", [=](Cfg& c) { gpu_base(c); c.K = 16; c.block_log = 19; });
  add("g_K8_b19", [=](Cfg& c) { gpu_base(c); c.K = 8; c.block_log = 19; });
  add("g_K16_rawtok", [=](Cfg& c) { gpu_base(c); c.K = 16; c.raw_tok = true; });
  add("g_K16_rawsym", [=](Cfg& c) { gpu_base(c); c.K = 16; c.raw_sym = true; });
  add("g_K16_rawboth", [=](Cfg& c) { gpu_base(c); c.K = 16; c.raw_tok = true; c.raw_sym = true; });
  add("g_K16_depth4", [=](Cfg& c) { gpu_base(c); c.K = 16; c.depth = 4; });
  add("g_K16_depth16", [=](Cfg& c) { gpu_base(c); c.K = 16; c.depth = 16; });
  add("g_K16_nolazy", [=](Cfg& c) { gpu_base(c); c.K = 16; c.lazy = 0; });
  // 12. the pass after the parse
  for (int depth : {8, 16, 32})
    for (uint32_t k : {16u, 32u, 64u, 255u})
      for (uint32_t t : {0u, 4u, 8u, 16u, 32u, 64u}) {
        static std::vector<std::string> names;
        names.push_back("p_d" + std::to_string(depth) + "_K" + std::to_string(k) + "_T" + std::to_string(t));
        add(names.back().c_str(), [=](Cfg& c) { c.post = true; c.post_k = k; c.post_t = t; c.lazy = 1; c.depth = depth; c.block_log = 20; c.maxm = 4096; });
      }
  add("p_b19", [](Cfg& c) { c.post = true; c.post_k = 32; c.post_t = 8; c.lazy = 1; c.depth = 8; c.block_log = 19; c.maxm = 4096; });
  add("p_max274", [](Cfg& c) { c.post = true; c.post_k = 32; c.post_t = 8; c.lazy = 1; c.depth = 8; c.block_log = 20; c.maxm = 274; });
  add("p_nolazy", [](Cfg& c) { c.post = true; c.post_k = 32; c.post_t = 8; c.depth = 8; c.block_log = 20; c.maxm = 4096; });
  // 6. block size, with the best of the above in mind
  for (uint32_t b : {16u, 17u, 18u, 20u}) {
    static std::vector<std::string> names;
    names.push_back("block" + std::to_string(b));
    add(names.back().c_str(), [b](Cfg& c) { c.block_log = b; });
  }
  printf("corpus\tcfg\ttok_per_group\tratio%%\tblocks_stored\ttok_per_MiB\tbytes_per_tok\tfence_per_tok\tshort%%\tlit%%out\ttok%%\text%%\tsym%%\tbits%%\tlit_o1_est_gain%%\trep%%\tsecs\n");
  fflush(stdout);
  for (const auto& cp : corpora) {
    FILE* f = fopen(cp.second.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", cp.second.c_str()); return 1; }
    fseek(f, 0, SEEK_END);
    const long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> data((size_t)sz);
    if (fread(data.data(), 1, data.size(), f) != data.size()) return 1;
    fclose(f);
    for (const Cfg& c : cfgs) {
      if (!only.empty() && only.find("," + c.name + ",") == std::string::npos) continue;
      const uint32_t bl = c.block_log ? c.block_log : block_log;
      const size_t B = (size_t)1 << bl, nb = (data.size() + B - 1) / B;
      std::atomic<size_t> next{0};
      std::mutex mu;
      Stats total;
      const auto t0 = std::chrono::steady_clock::now();
      std::vector<std::thread> ts;
      for (unsigned t = 0; t < threads; ++t)
        ts.emplace_back([&] {
          Stats local;
          for (;;) {
            const size_t b = next.fetch_add(1);
            if (b >= nb) break;
            compress_block(data.data() + b * B, (uint32_t)std::min(B, data.size() - b * B), c, local);
          }
          std::lock_guard<std::mutex> lk(mu);
          total.add(local);
        });
      for (auto& t : ts) t.join();
      const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      const double mib = total.raw / 1048576.0, o = (double)std::max<uint64_t>(total.out, 1);
      printf("%s\t%s\t%.1f\t%.2f\t%llu\t%.0f\t%.1f\t%.2f\t%.0f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.0f\t%.1f\n", cp.first.c_str(), c.name.c_str(), total.groups ? (double)total.tokens / total.groups : 0.0, 100.0 * total.out / total.raw,
             (unsigned long long)total.stored, total.tokens / mib, total.tokens ? (double)total.raw / total.tokens : 0.0, total.tokens ? (double)total.fences / total.tokens : 0.0,
             total.tokens ? 100.0 * total.short_tokens / total.tokens : 0.0, 100.0 * total.lit / o, 100.0 * total.tok / o, 100.0 * total.ext / o, 100.0 * total.sym / o, 100.0 * total.bits / o,
             total.lit_o0 > 0 ? 100.0 * (total.lit_o0 - total.lit_o1) / 8 / o : 0.0, total.tokens ? 100.0 * total.rep_hits / total.tokens : 0.0, secs);
      fflush(stdout);
    }
  }
  return 0;
}
