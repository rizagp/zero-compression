// zero: what the block format stands on: the entropy coder. rANS with 12-bit frequencies and 64 interleaved lanes, each stream
// kept raw when that is smaller. src/codec2.h has the block format.
//
//   a stream: u8 kind (0 raw, 1 rANS), u32 len (the bytes that follow)
//     raw:  len bytes
//     rANS: u8 present[32] (a bit per symbol), u16 freq[popcount] (sum 4096), u32 lane_len[64], the lanes' bytes
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

}  // namespace zb
