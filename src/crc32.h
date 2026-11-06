// zero: CRC-32 (IEEE 802.3, the one of zip / zlib / PNG), slice-by-8.
#pragma once
#include <cstdint>
#include <cstddef>

namespace crc32 {
struct Tables {
  uint32_t t[8][256];
  Tables() {
    for (uint32_t i = 0; i < 256; ++i) {
      uint32_t c = i;
      for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
      t[0][i] = c;
    }
    for (uint32_t i = 0; i < 256; ++i)
      for (int k = 1; k < 8; ++k) t[k][i] = (t[k - 1][i] >> 8) ^ t[0][t[k - 1][i] & 0xff];
  }
};
inline const Tables& tables() { static const Tables T; return T; }

// continues a running CRC (start with crc = 0)
inline uint32_t update(uint32_t crc, const void* data, size_t n) {
  const Tables& T = tables();
  const uint8_t* p = (const uint8_t*)data;
  uint32_t c = ~crc;
  while (n >= 8) {
    uint32_t a, b;
    a = (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
    b = (uint32_t)p[4] | (uint32_t)p[5] << 8 | (uint32_t)p[6] << 16 | (uint32_t)p[7] << 24;
    a ^= c;
    c = T.t[7][a & 0xff] ^ T.t[6][(a >> 8) & 0xff] ^ T.t[5][(a >> 16) & 0xff] ^ T.t[4][a >> 24] ^
        T.t[3][b & 0xff] ^ T.t[2][(b >> 8) & 0xff] ^ T.t[1][(b >> 16) & 0xff] ^ T.t[0][b >> 24];
    p += 8;
    n -= 8;
  }
  while (n--) c = T.t[0][(c ^ *p++) & 0xff] ^ (c >> 8);
  return ~c;
}
inline uint32_t of(const void* data, size_t n) { return update(0, data, n); }

// the CRC of A followed by B from crc(A), crc(B) and B's length (zlib's method: multiplying by x^(8 len) in GF(2))
inline uint32_t gf2_times(const uint32_t* m, uint32_t v) {
  uint32_t s = 0;
  for (; v; v >>= 1, ++m) if (v & 1) s ^= *m;
  return s;
}
inline void gf2_square(uint32_t* out, const uint32_t* m) { for (int i = 0; i < 32; ++i) out[i] = gf2_times(m, m[i]); }
inline uint32_t combine(uint32_t a, uint32_t b, uint64_t len_b) {
  if (!len_b) return a;
  uint32_t even[32], odd[32];
  odd[0] = 0xEDB88320u;
  for (int i = 1; i < 32; ++i) odd[i] = 1u << (i - 1);
  gf2_square(even, odd);
  gf2_square(odd, even);
  do {
    gf2_square(even, odd);
    if (len_b & 1) a = gf2_times(even, a);
    len_b >>= 1;
    if (!len_b) break;
    gf2_square(odd, even);
    if (len_b & 1) a = gf2_times(odd, a);
    len_b >>= 1;
  } while (len_b);
  return a ^ b;
}
}  // namespace crc32
