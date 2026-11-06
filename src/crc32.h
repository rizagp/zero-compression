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
}  // namespace crc32
