// zero: "ZB2", the block format built for the GPU's decoder. Tokens gather in groups that decode side by side, one token per
// lane, with no memory fence inside a group:
//
//   A match reads only output from before its group, or the literals of its own token (those come from the literal stream,
//   not from the output, so an overlapping match such as a run is fine there).
//   Lengths past 15 have a fixed size, so a lane finds its own by counting flags: L >= 15 has L-15 in 3 bytes, then M-4 >= 15
//   has M-19 in 2 bytes.
//   Offsets: per token a symbol, b = 0..19 for an offset in [2^b, 2^(b+1)) with b raw bits, or 250..252 for one of the last
//   three offsets of the group before (lanes need nothing from each other).
//
// The parse knows nothing of groups, so segments can be parsed in parallel. One pass per block then makes the groups: a
// match that reads its own group either is cut to its legal part, when that loses fewer than T bytes (the rest become
// literals of the next token, so no position moves), or closes the group. T = 0 only closes groups. tools/lab/zlab.cpp has
// the measurements behind this.
//
// Block payload: u32 n_tokens, n_lits, n_ext (bytes), n_offbits (bytes), n_groups; streams lit, tok, ext, sym, groups, as in
// zb::encode_stream (raw or rANS); the offset bits, LSB first.
#pragma once
#include "codec.h"

namespace zb2 {

constexpr uint32_t HDR_BYTES = 20;
constexpr uint32_t SYM_REP = 250;

struct Params {
  int depth = 8;          // hash chain steps
  int lazy = 1;           // positions looked ahead before taking a match
  uint32_t max_group = 255;
  uint32_t cut_below = 0; // T above
  uint32_t max_match = 4096;
};

struct RawToken { uint32_t i, m, d; };

inline uint32_t bitlen(uint32_t d) {
  uint32_t b = 0;
  while ((d >> (b + 1)) != 0) ++b;
  return b;
}

// LZ77 with hash chains and lazy matching; no rule (a GPU's parse would give the same kind of tokens its own way)
inline void parse(const uint8_t* in, uint32_t n, const Params& P, std::vector<RawToken>& out) {
  out.clear();
  if (n < 4) return;
  constexpr uint32_t HBITS = 16;
  std::vector<int32_t> head((size_t)1 << HBITS, -1), prev(n, -1);
  const auto hash = [&](uint32_t i) { return (zb::get32(in + i) * 2654435761u) >> (32 - HBITS); };
  uint32_t inserted = 0;
  const auto best_at = [&](uint32_t i, uint32_t& bd) -> uint32_t {
    for (; inserted < i; ++inserted) {
      if (inserted + 4 > n) continue;
      const uint32_t h = hash(inserted);
      prev[inserted] = head[h];
      head[h] = (int32_t)inserted;
    }
    uint32_t best = 0;
    if (i + 4 > n) return 0;
    const uint32_t lim = std::min(P.max_match, n - i);
    int32_t j = head[hash(i)];
    for (int k = 0; k < P.depth && j >= 0; ++k, j = prev[j]) {
      if (best && in[(uint32_t)j + std::min(best, lim - 1)] != in[i + std::min(best, lim - 1)]) continue;
      uint32_t l = 0;
      while (l < lim && in[(uint32_t)j + l] == in[i + l]) ++l;
      if (l > best) { best = l; bd = i - (uint32_t)j; }
    }
    return best >= 4 ? best : 0;
  };
  uint32_t i = 0;
  while (i < n) {
    uint32_t d = 0, m = best_at(i, d);
    if (!m) { ++i; continue; }
    for (int la = 0; la < P.lazy && i + 1 < n; ++la) {
      uint32_t d2 = 0;
      const uint32_t m2 = best_at(i + 1, d2);
      if (m2 && 8.0 * m2 - bitlen(d2) - 8 > 8.0 * m - bitlen(d)) { ++i; m = m2; d = d2; } else break;
    }
    out.push_back(RawToken{i, m, d});
    i += m;
  }
}

// false when the block is better stored
inline bool compress_block(const uint8_t* in, uint32_t n, std::vector<uint8_t>& out, const Params& P = Params{}) {
  std::vector<RawToken> tk;
  parse(in, n, P, tk);
  std::vector<uint8_t> lit, tok, ext, sym, bits, groups;
  lit.reserve(n);
  uint64_t bitbuf = 0;
  int bitn = 0;
  uint32_t gstart = 0, gt = 0, op = 0, tokens = 0;
  uint32_t recent[3] = {1, 4, 8}, last3[3] = {1, 4, 8};
  const auto close = [&](uint32_t at) {
    if (gt) groups.push_back((uint8_t)gt);
    gt = 0;
    gstart = at;
    memcpy(recent, last3, sizeof recent);
  };
  const auto put16 = [&](uint32_t v) { ext.push_back((uint8_t)v); ext.push_back((uint8_t)(v >> 8)); };
  const auto put24 = [&](uint32_t v) { put16(v); ext.push_back((uint8_t)(v >> 16)); };
  const auto note_offset = [&](uint32_t d) { last3[2] = last3[1]; last3[1] = last3[0]; last3[0] = d; };
  for (const RawToken& x : tk) {
    const uint32_t i = x.i, d = x.d;
    uint32_t m = x.m;
    if (gt == P.max_group) close(op);
    const uint32_t j = i - d;
    if (j < op && j + std::min(m, d) > gstart) {  // reads what its own group writes
      const uint32_t legal = d >= m && gstart > j ? gstart - j : 0;
      const uint32_t use = legal >= 4 ? legal : 0;
      if (m - use < P.cut_below) {
        if (use == 0) continue;                    // (its bytes become literals of the next token)
        m = use;
      } else {
        close(op);
      }
    }
    const uint32_t L = i - op;
    lit.insert(lit.end(), in + op, in + i);
    tok.push_back((uint8_t)((std::min(L, 15u) << 4) | std::min(m - 4, 15u)));
    if (L >= 15) put24(L - 15);
    if (m - 4 >= 15) put16(m - 19);
    int rep = -1;
    for (int r = 0; r < 3; ++r) if (recent[r] == d) { rep = r; break; }
    if (rep >= 0) {
      sym.push_back((uint8_t)(SYM_REP + rep));
    } else {
      const uint32_t b = bitlen(d);
      sym.push_back((uint8_t)b);
      bitbuf |= (uint64_t)(d - (1u << b)) << bitn;
      bitn += (int)b;
      while (bitn >= 8) { bits.push_back((uint8_t)bitbuf); bitbuf >>= 8; bitn -= 8; }
    }
    note_offset(d);
    ++gt; ++tokens;
    op = i + m;
  }
  if (gt) groups.push_back((uint8_t)gt);
  lit.insert(lit.end(), in + op, in + n);
  if (bitn) bits.push_back((uint8_t)bitbuf);
  out.clear();
  zb::put32(out, tokens);
  zb::put32(out, (uint32_t)lit.size());
  zb::put32(out, (uint32_t)ext.size());
  zb::put32(out, (uint32_t)bits.size());
  zb::put32(out, (uint32_t)groups.size());
  zb::encode_stream(lit.data(), (uint32_t)lit.size(), out);
  zb::encode_stream(tok.data(), (uint32_t)tok.size(), out);
  zb::encode_stream(ext.data(), (uint32_t)ext.size(), out);
  zb::encode_stream(sym.data(), (uint32_t)sym.size(), out);
  zb::encode_stream(groups.data(), (uint32_t)groups.size(), out);
  out.insert(out.end(), bits.begin(), bits.end());
  return out.size() < n;
}

inline void decompress_block(const uint8_t* in, size_t in_len, uint8_t* out, uint32_t n) {
  if (in_len < HDR_BYTES) throw zb::Error("block damaged: header cut");
  const uint8_t* p = in;
  const uint8_t* end = in + in_len;
  const uint32_t nt = zb::get32(p), nl = zb::get32(p + 4), ne = zb::get32(p + 8), nb = zb::get32(p + 12), ng = zb::get32(p + 16);
  p += HDR_BYTES;
  if (nl > n || nt > n || ne > 5 * (uint64_t)nt || ng > nt) throw zb::Error("block damaged: counts");
  std::vector<uint8_t> lit(nl), tok(nt), ext(ne + 3), sym(nt), grp(ng);
  p += zb::decode_stream(p, end, nl, lit.data());
  p += zb::decode_stream(p, end, nt, tok.data());
  p += zb::decode_stream(p, end, ne, ext.data());
  p += zb::decode_stream(p, end, nt, sym.data());
  p += zb::decode_stream(p, end, ng, grp.data());
  if ((size_t)(end - p) < nb) throw zb::Error("block damaged: offset bits cut");
  const uint8_t* bits = p;
  uint32_t o = 0, li = 0, ei = 0, t = 0;
  uint64_t bp = 0;
  uint32_t recent[3] = {1, 4, 8}, last3[3] = {1, 4, 8};
  for (uint32_t g = 0; g < ng; ++g) {
    const uint32_t gn = grp[g];
    if (gn == 0 || t + gn > nt) throw zb::Error("block damaged: groups");
    for (uint32_t k = 0; k < gn; ++k, ++t) {
      const uint32_t tv = tok[t], sv = sym[t];
      uint32_t L = tv >> 4, M = (tv & 15) + 4;
      if (L == 15) { if (ei + 3 > ne) throw zb::Error("block damaged: ext"); L += (uint32_t)ext[ei] | (uint32_t)ext[ei + 1] << 8 | (uint32_t)ext[ei + 2] << 16; ei += 3; }
      if ((tv & 15) == 15) { if (ei + 2 > ne) throw zb::Error("block damaged: ext"); M += (uint32_t)ext[ei] | (uint32_t)ext[ei + 1] << 8; ei += 2; }
      uint32_t D = 0;
      if (sv >= SYM_REP) { if (sv > SYM_REP + 2) throw zb::Error("block damaged: symbol"); D = recent[sv - SYM_REP]; }
      else {
        if (sv > 19) throw zb::Error("block damaged: symbol");
        uint32_t x = 0;
        for (uint32_t q = 0; q < sv; ++q, ++bp) {
          if ((bp >> 3) >= nb) throw zb::Error("block damaged: offset bits");
          x |= (uint32_t)((bits[bp >> 3] >> (bp & 7)) & 1) << q;
        }
        D = (1u << sv) + x;
      }
      if (li + L > nl || (uint64_t)o + L + M > n || D == 0 || D > o + L) throw zb::Error("block damaged: token");
      memcpy(out + o, lit.data() + li, L);
      o += L; li += L;
      for (uint32_t i = 0; i < M; ++i) out[o + i] = out[o + i - D];
      o += M;
      last3[2] = last3[1]; last3[1] = last3[0]; last3[0] = D;
    }
    memcpy(recent, last3, sizeof recent);
  }
  if (t != nt || o + (nl - li) != n) throw zb::Error("block damaged: lengths");
  memcpy(out + o, lit.data() + li, nl - li);
}

}  // namespace zb2
