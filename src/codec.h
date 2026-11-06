// zero: the block codec, "ZB1". A block (up to 1 MiB) is compressed on its own, so blocks can be processed in
// parallel on the GPU, and the format is the same whether the CPU or the GPU made it:
//
//   LZ77 (greedy parse, matches of 4..274 bytes, offsets within the block) into five streams
//     lit     the literal bytes
//     tok     one byte per match: (min(L,15) << 4) | min(M-4, 15); L literals run before the match of M bytes
//     ext     the lengths past 15: for L >= 15 the bytes of L-15 in 255-steps (LZ4 style), then for M-4 >= 15 one
//             byte M-4-15; in token order
//     offsym  per match the bit length b of the offset D (D in [2^b, 2^(b+1))), 0..20
//     offbits the b low bits of D - 2^b per match, packed LSB first
//   then lit, tok, ext and offsym are entropy coded with rANS (12-bit frequencies, 64 interleaved lanes), each
//   stream kept raw instead when that is smaller.
//
// Block payload (little endian):
//   u32 n_tokens, u32 n_lits, u32 n_ext, u32 offbits_bytes
//   stream lit, stream tok, stream ext, stream offsym:
//     u8 kind (0 raw, 1 rANS), u32 len (the bytes that follow for this stream)
//     raw:  len bytes
//     rANS: u8 present[32] (a bit per symbol), u16 freq[popcount] (sum 4096), u32 lane_len[64], the lanes' bytes
//   offbits: offbits_bytes bytes
// A block whose payload would not be smaller than itself is stored as is (method 0 in the container).
//
// This header has the constants, the shared helpers (frequency normalization, stream headers) and the CPU
// reference encoder and decoder. The GPU kernels (kernels/zero.cl) produce and consume the same bytes.
#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>

namespace zb {

constexpr uint32_t MINMATCH = 4;
constexpr uint32_t MAXMATCH = 4 + 15 + 255;  // 274
constexpr uint32_t NLANES = 64;
constexpr uint32_t RANS_BITS = 12;
constexpr uint32_t RANS_SCALE = 1u << RANS_BITS;
constexpr uint32_t RANS_L = 1u << 23;
constexpr uint32_t MAX_BLOCK_LOG = 20;
constexpr uint32_t MIN_BLOCK_LOG = 16;
constexpr uint32_t NSTREAMS = 4;
constexpr uint32_t HDR_BYTES = 16;

struct Error : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// ---- little-endian helpers
inline void put32(std::vector<uint8_t>& o, uint32_t v) { for (int i = 0; i < 4; ++i) o.push_back((uint8_t)(v >> (8 * i))); }
inline void put16(std::vector<uint8_t>& o, uint16_t v) { o.push_back((uint8_t)v); o.push_back((uint8_t)(v >> 8)); }
inline uint32_t get32(const uint8_t* p) { return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24; }
inline uint16_t get16(const uint8_t* p) { return (uint16_t)(p[0] | p[1] << 8); }

// ---- frequencies: counts -> 12-bit frequencies summing to RANS_SCALE, every present symbol at least 1.
// Deterministic integer arithmetic; the GPU does the same steps.
inline void normalize(const uint32_t counts[256], uint32_t total, uint16_t freq[256]) {
  uint32_t sum = 0;
  int present = 0;
  for (int s = 0; s < 256; ++s) {
    freq[s] = 0;
    if (!counts[s]) continue;
    ++present;
    uint32_t f = (uint32_t)(((uint64_t)counts[s] << RANS_BITS) / total);
    if (f == 0) f = 1;
    freq[s] = (uint16_t)f;
    sum += f;
  }
  while (sum > RANS_SCALE) {  // (at most `present` too many: one taken from the largest each time)
    int best = -1;
    for (int s = 0; s < 256; ++s)
      if (freq[s] > 1 && (best < 0 || freq[s] > freq[best])) best = s;
    --freq[best];
    --sum;
  }
  if (sum < RANS_SCALE) {
    int best = 0;
    for (int s = 1; s < 256; ++s)
      if (freq[s] > freq[best]) best = s;
    freq[best] = (uint16_t)(freq[best] + (RANS_SCALE - sum));
  }
}

// the bit length of an offset D >= 1: b with 2^b <= D < 2^(b+1)
inline uint32_t off_bits(uint32_t d) {
  uint32_t b = 0;
  while ((d >> (b + 1)) != 0) ++b;
  return b;
}

// ---- rANS, one lane (ryg_rans, byte-wise renormalization)
struct RansEnc {
  uint32_t x = RANS_L;
  // encodes into buf backwards: end points one past the last byte; returns the new begin
  static uint8_t* put(uint8_t* ptr, uint32_t& x, uint32_t freq, uint32_t cum) {
    const uint32_t x_max = freq << (23 - RANS_BITS + 8);  // ((L >> 12) << 8) * freq
    while (x >= x_max) { *--ptr = (uint8_t)x; x >>= 8; }
    x = ((x / freq) << RANS_BITS) + (x % freq) + cum;
    return ptr;
  }
  static uint8_t* flush(uint8_t* ptr, uint32_t x) {
    ptr -= 4;
    ptr[0] = (uint8_t)x; ptr[1] = (uint8_t)(x >> 8); ptr[2] = (uint8_t)(x >> 16); ptr[3] = (uint8_t)(x >> 24);
    return ptr;
  }
};

// stream header: kind + len, then the body; returns the body size for a rANS stream with `present` symbols
inline size_t rans_header_bytes(int present) { return 32 + 2 * (size_t)present + 4 * NLANES; }

// Entropy codes `n` symbols (bytes) into `out` as one stream section (kind, len, body). Raw when rANS is not smaller.
inline void encode_stream(const uint8_t* sym, uint32_t n, std::vector<uint8_t>& out) {
  if (n == 0) { out.push_back(0); put32(out, 0); return; }
  uint32_t counts[256] = {};
  for (uint32_t i = 0; i < n; ++i) ++counts[sym[i]];
  uint16_t freq[256];
  normalize(counts, n, freq);
  uint32_t cum[256];
  uint32_t c = 0;
  int present = 0;
  for (int s = 0; s < 256; ++s) { cum[s] = c; c += freq[s]; present += freq[s] != 0; }
  // lanes: symbol i belongs to lane i % NLANES, encoded in reverse
  std::vector<uint8_t> body;
  body.reserve(rans_header_bytes(present) + n + 8);
  std::vector<uint8_t> lane_data;
  std::vector<uint32_t> lane_len(NLANES);
  std::vector<uint8_t> tmp(2 * ((n + NLANES - 1) / NLANES) + 16);
  for (uint32_t j = 0; j < NLANES; ++j) {
    uint8_t* end = tmp.data() + tmp.size();
    uint8_t* ptr = end;
    uint32_t x = RANS_L;
    if (j < n) {
      uint32_t i = j + ((n - 1 - j) / NLANES) * NLANES;  // the last index of this lane
      for (;;) {
        const uint8_t s = sym[i];
        ptr = RansEnc::put(ptr, x, freq[s], cum[s]);
        if (i < NLANES) break;
        i -= NLANES;
      }
    }
    ptr = RansEnc::flush(ptr, x);
    lane_len[j] = (uint32_t)(end - ptr);
    lane_data.insert(lane_data.end(), ptr, end);
  }
  const size_t total = rans_header_bytes(present) + lane_data.size();
  if (total >= n) {  // raw is as small or smaller
    out.push_back(0);
    put32(out, n);
    out.insert(out.end(), sym, sym + n);
    return;
  }
  out.push_back(1);
  put32(out, (uint32_t)total);
  for (int b = 0; b < 32; ++b) {
    uint8_t bits = 0;
    for (int k = 0; k < 8; ++k) bits |= (uint8_t)((freq[b * 8 + k] != 0) << k);
    out.push_back(bits);
  }
  for (int s = 0; s < 256; ++s)
    if (freq[s]) put16(out, freq[s]);
  for (uint32_t j = 0; j < NLANES; ++j) put32(out, lane_len[j]);
  out.insert(out.end(), lane_data.begin(), lane_data.end());
}

// Decodes one stream section at p (n symbols expected) into sym; returns the bytes consumed. Throws on damage.
inline size_t decode_stream(const uint8_t* p, const uint8_t* end, uint32_t n, uint8_t* sym) {
  if (end - p < 5) throw Error("block damaged: stream header cut");
  const uint8_t kind = p[0];
  const uint32_t len = get32(p + 1);
  const uint8_t* body = p + 5;
  if ((size_t)(end - body) < len) throw Error("block damaged: stream body cut");
  if (kind == 0) {
    if (len != n) throw Error("block damaged: raw stream length");
    memcpy(sym, body, n);
    return 5 + len;
  }
  if (kind != 1) throw Error("block damaged: unknown stream kind");
  if (len < 32 + 4 * NLANES) throw Error("block damaged: rANS header cut");
  const uint8_t* present = body;
  int np = 0;
  for (int b = 0; b < 32; ++b)
    for (int k = 0; k < 8; ++k) np += (present[b] >> k) & 1;
  if (len < rans_header_bytes(np)) throw Error("block damaged: rANS table cut");
  const uint8_t* fp = body + 32;
  uint16_t freq[256];
  uint32_t cum[256];
  uint32_t c = 0;
  for (int s = 0; s < 256; ++s) {
    freq[s] = 0;
    if ((present[s >> 3] >> (s & 7)) & 1) { freq[s] = get16(fp); fp += 2; }
    cum[s] = c;
    c += freq[s];
  }
  if (c != RANS_SCALE) throw Error("block damaged: rANS frequencies");
  std::vector<uint8_t> slot(RANS_SCALE);
  for (int s = 0; s < 256; ++s)
    for (uint32_t k = 0; k < freq[s]; ++k) slot[cum[s] + k] = (uint8_t)s;
  const uint8_t* lp = fp;
  const uint8_t* data = lp + 4 * NLANES;
  const uint8_t* body_end = body + len;
  for (uint32_t j = 0; j < NLANES; ++j) {
    const uint32_t ll = get32(lp + 4 * j);
    if ((size_t)(body_end - data) < ll || ll < 4) throw Error("block damaged: rANS lane cut");
    const uint8_t* in = data;
    const uint8_t* in_end = data + ll;
    uint32_t x = get32(in);
    in += 4;
    for (uint32_t i = j; i < n; i += NLANES) {
      const uint32_t sl = x & (RANS_SCALE - 1);
      const uint8_t s = slot[sl];
      sym[i] = s;
      x = freq[s] * (x >> RANS_BITS) + sl - cum[s];
      while (x < RANS_L) {
        if (in >= in_end) throw Error("block damaged: rANS lane short");
        x = (x << 8) | *in++;
      }
    }
    data += ll;
  }
  return 5 + len;
}

// ---- the CPU reference encoder
struct Tokens {
  std::vector<uint8_t> lit, tok, ext, offsym, offbits;
  uint32_t n_tokens = 0;
};

// Greedy LZ77 with hash chains (deterministic; the GPU finds matches its own way, the format is the same).
inline void lz_parse(const uint8_t* in, uint32_t n, Tokens& T, int depth = 32) {
  T = Tokens{};
  if (n == 0) return;
  constexpr uint32_t HBITS = 16;
  std::vector<int32_t> head((size_t)1 << HBITS, -1);
  std::vector<int32_t> prev(n, -1);
  auto hash4 = [&](uint32_t i) -> uint32_t {
    const uint32_t v = (uint32_t)in[i] | (uint32_t)in[i + 1] << 8 | (uint32_t)in[i + 2] << 16 | (uint32_t)in[i + 3] << 24;
    return (v * 2654435761u) >> (32 - HBITS);
  };
  auto insert = [&](uint32_t i) {
    if (i + 4 > n) return;
    const uint32_t h = hash4(i);
    prev[i] = head[h];
    head[h] = (int32_t)i;
  };
  uint32_t lit_start = 0;  // first literal of the current run
  uint64_t bitbuf = 0;
  int nbits = 0;
  auto put_bits = [&](uint32_t v, uint32_t b) {
    bitbuf |= (uint64_t)v << nbits;
    nbits += (int)b;
    while (nbits >= 8) { T.offbits.push_back((uint8_t)bitbuf); bitbuf >>= 8; nbits -= 8; }
  };
  uint32_t i = 0;
  while (i < n) {
    uint32_t best_len = 0, best_off = 0;
    if (i + MINMATCH <= n) {
      const uint32_t h = hash4(i);
      int32_t j = head[h];
      const uint32_t maxlen = std::min<uint32_t>(MAXMATCH, n - i);
      for (int d = 0; j >= 0 && d < depth; ++d, j = prev[j]) {
        const uint32_t off = i - (uint32_t)j;
        if (in[j + best_len] != in[i + best_len]) continue;
        uint32_t l = 0;
        while (l < maxlen && in[j + l] == in[i + l]) ++l;
        if (l > best_len) { best_len = l; best_off = off; if (l == maxlen) break; }
      }
    }
    if (best_len >= MINMATCH) {
      const uint32_t L = i - lit_start;
      const uint32_t M = best_len;
      T.tok.push_back((uint8_t)((std::min<uint32_t>(L, 15) << 4) | std::min<uint32_t>(M - 4, 15)));
      if (L >= 15) {
        uint32_t rem = L - 15;
        while (rem >= 255) { T.ext.push_back(255); rem -= 255; }
        T.ext.push_back((uint8_t)rem);
      }
      if (M - 4 >= 15) T.ext.push_back((uint8_t)(M - 4 - 15));
      const uint32_t b = off_bits(best_off);
      T.offsym.push_back((uint8_t)b);
      put_bits(best_off - (1u << b), b);
      ++T.n_tokens;
      for (uint32_t k = 0; k < M; ++k) insert(i + k);
      i += M;
      lit_start = i;
    } else {
      T.lit.push_back(in[i]);
      insert(i);
      ++i;
    }
  }
  if (nbits > 0) T.offbits.push_back((uint8_t)bitbuf);
}

// the streams -> the block payload
inline void pack_block(const Tokens& T, std::vector<uint8_t>& out) {
  out.clear();
  put32(out, T.n_tokens);
  put32(out, (uint32_t)T.lit.size());
  put32(out, (uint32_t)T.ext.size());
  put32(out, (uint32_t)T.offbits.size());
  encode_stream(T.lit.data(), (uint32_t)T.lit.size(), out);
  encode_stream(T.tok.data(), (uint32_t)T.tok.size(), out);
  encode_stream(T.ext.data(), (uint32_t)T.ext.size(), out);
  encode_stream(T.offsym.data(), (uint32_t)T.offsym.size(), out);
  out.insert(out.end(), T.offbits.begin(), T.offbits.end());
}

// Compresses one block on the CPU; false when the block is better stored as is (out then untouched).
inline bool compress_block_cpu(const uint8_t* in, uint32_t n, std::vector<uint8_t>& out, int depth = 32) {
  Tokens T;
  lz_parse(in, n, T, depth);
  pack_block(T, out);
  return out.size() < n;
}

// Decompresses one block payload of `in_len` bytes into out (exactly `n` bytes). Throws on damage.
inline void decompress_block_cpu(const uint8_t* in, size_t in_len, uint8_t* out, uint32_t n) {
  if (in_len < HDR_BYTES) throw Error("block damaged: header cut");
  const uint8_t* p = in;
  const uint8_t* end = in + in_len;
  const uint32_t n_tokens = get32(p), n_lits = get32(p + 4), n_ext = get32(p + 8), n_offb = get32(p + 12);
  p += HDR_BYTES;
  if (n_lits > n || n_tokens > n) throw Error("block damaged: counts");
  std::vector<uint8_t> lit(n_lits), tok(n_tokens), ext(n_ext), offsym(n_tokens);
  p += decode_stream(p, end, n_lits, lit.data());
  p += decode_stream(p, end, n_tokens, tok.data());
  p += decode_stream(p, end, n_ext, ext.data());
  p += decode_stream(p, end, n_tokens, offsym.data());
  if ((size_t)(end - p) < n_offb) throw Error("block damaged: offset bits cut");
  const uint8_t* offb = p;
  // replay
  uint32_t o = 0, li = 0, ei = 0;
  uint64_t bitpos = 0;
  for (uint32_t k = 0; k < n_tokens; ++k) {
    const uint8_t t = tok[k];
    uint32_t L = t >> 4, M = (t & 15) + 4;
    if (L == 15) {
      uint32_t b;
      do {
        if (ei >= n_ext) throw Error("block damaged: ext short");
        b = ext[ei++];
        L += b;
      } while (b == 255);
    }
    if (M - 4 == 15) {
      if (ei >= n_ext) throw Error("block damaged: ext short");
      M += ext[ei++];
    }
    const uint32_t b = offsym[k];
    if (b > 20) throw Error("block damaged: offset symbol");
    uint32_t extra = 0;
    if (b) {
      const uint64_t bp = bitpos;
      if ((bp + b + 7) / 8 > n_offb) throw Error("block damaged: offset bits short");
      uint64_t w = 0;
      for (int q = 0; q < 4; ++q) {
        const uint64_t byte = (bp >> 3) + (uint64_t)q;
        if (byte < n_offb) w |= (uint64_t)offb[byte] << (8 * q);
      }
      extra = (uint32_t)((w >> (bp & 7)) & ((1u << b) - 1));
      bitpos += b;
    }
    const uint32_t D = (1u << b) + extra;
    if (li + L > n_lits || o + L + M > n) throw Error("block damaged: token past the end");
    memcpy(out + o, lit.data() + li, L);
    o += L;
    li += L;
    if (D > o) throw Error("block damaged: offset before the block");
    for (uint32_t q = 0; q < M; ++q) out[o + q] = out[o + q - D];
    o += M;
  }
  const uint32_t tail = n_lits - li;
  if (o + tail != n) throw Error("block damaged: size mismatch");
  memcpy(out + o, lit.data() + li, tail);
}

}  // namespace zb
