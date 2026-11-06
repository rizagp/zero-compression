// zero: the GPU kernels of the ZB1 block codec (see src/codec.h for the format). OpenCL C, compiled by the Intel
// OpenCL runtime and run through Level Zero. Built with -DBLOG=<log2 block size> -DG=<work-group size>.
//
// A batch holds nb blocks of B = 2^BLOG bytes (the last one nlast bytes) laid out one after the other; per-block
// arrays are indexed by block then position. One work-group (G work-items) handles one block in the per-block
// kernels; the per-position kernels run over every position of the batch.

#define B (1u << BLOG)
#define HBITS (BLOG - 3)
#define HB (1u << HBITS)
#define TCAP (B / 4u + 16u)                // tokens per block at most (a multiple of 16: aligned chunk loads)
#define ECAP (B / 4u + B / 128u + 16u)     // ext bytes per block at most (a multiple of 16)
#define OCAP (5u * B / 8u + 8u)            // offset-bit bytes per block at most (a multiple of 4)
#define OUTCAP (B + B / 8u + 8192u)        // payload slab per block
#define NL 64u                             // rANS lanes
#define LC0 (2u * (B / NL) + 16u)          // lane scratch, lit
#define LC1 (2u * (TCAP / NL + 1u) + 16u)  // tok, offsym
#define LC2 (2u * (ECAP / NL + 1u) + 16u)  // ext
#define LC3 LC1
#define LANES_PER_BLOCK (NL * (LC0 + LC1 + LC2 + LC3))
#define MINMATCH 4u
#define MAXMATCH 274u
#define SEG (B / G)
#define RANS_L (1u << 23)
#define SCALE 4096u

// 4 bytes at any byte position of a block, from the 4-byte-aligned words around it (blocks start 4-aligned and
// the input buffer has 16 bytes of slack past its end). One or two word loads instead of four byte loads.
inline uint load4u(__global const uchar* base, uint pos) {
  __global const uint* w = (__global const uint*)base;
  const uint i = pos >> 2, s = (pos & 3u) * 8u;
  const uint lo = w[i];
  return s == 0 ? lo : ((lo >> s) | (w[i + 1] << (32u - s)));
}
inline uint get32g(__global const uchar* p) { return (uint)p[0] | ((uint)p[1] << 8) | ((uint)p[2] << 16) | ((uint)p[3] << 24); }
inline uint get16g(__global const uchar* p) { return (uint)p[0] | ((uint)p[1] << 8); }
inline void put32g(__global uchar* p, uint v) { p[0] = (uchar)v; p[1] = (uchar)(v >> 8); p[2] = (uchar)(v >> 16); p[3] = (uchar)(v >> 24); }
inline void put16g(__global uchar* p, uint v) { p[0] = (uchar)v; p[1] = (uchar)(v >> 8); }
inline uint block_len(uint b, uint nb, uint nlast) { return b == nb - 1 ? nlast : B; }

// ============================================================ encoder

// 1. every position with 4 bytes left counts itself in its block's hash bucket
__kernel void k_hash_count(__global const uchar* in, __global uint* hcount, uint nb, uint nlast) {
  const uint gid = get_global_id(0);
  const uint b = gid >> BLOG;
  if (b >= nb) return;
  const uint p = gid & (B - 1);
  if (p + 4 > block_len(b, nb, nlast)) return;
  atomic_inc(&hcount[b * HB + ((load4u(in, gid) * 2654435761u) >> (32 - HBITS))]);
}

// 2. bucket starts: an exclusive scan of the block's bucket counts (one work-group per block); hfill zeroed
__kernel void k_bucket_scan(__global const uint* hcount, __global uint* bstart, __global uint* hfill, uint nb) {
  const uint b = get_group_id(0);
  const uint k = get_local_id(0);
  __global const uint* hc = hcount + (size_t)b * HB;
  __global uint* bs = bstart + (size_t)b * HB;
  __global uint* hf = hfill + (size_t)b * HB;
  __local uint sums[G];
  const uint per = HB / G;
  uint s = 0;
  for (uint i = k * per; i < (k + 1) * per; ++i) s += hc[i];
  sums[k] = s;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (k == 0) {
    uint a = 0;
    for (uint i = 0; i < G; ++i) { const uint x = sums[i]; sums[i] = a; a += x; }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint run = sums[k];
  for (uint i = k * per; i < (k + 1) * per; ++i) { bs[i] = run; run += hc[i]; hf[i] = 0; }
}

// 3. positions grouped by bucket: srt holds each bucket's positions one after the other, each with 12 more bits of
//    its hash in the top bits. The order inside a bucket is the order the lanes reach the counter, roughly position
//    order, and the closer the better: the neighbours are then the nearest earlier positions. 16 lanes and byte
//    loads keep it closer than 32 lanes and word loads, whatever the cause (measured: ratio 24.4 -> 23.7 %, the
//    kernel 6.5 -> 8.3 ms per 16 MiB; more match candidates buy less per ms), so this kernel reads bytes on purpose.
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void k_scatter(__global const uchar* in, __global const uint* bstart, __global uint* hfill, __global uint* srt,
                        uint nb, uint nlast) {
  const uint gid = get_global_id(0);
  const uint b = gid >> BLOG;
  if (b >= nb) return;
  const uint p = gid & (B - 1);
  if (p + 4 > block_len(b, nb, nlast)) return;
  const uint hv = get32g(in + gid) * 2654435761u;
  const uint h = b * HB + (hv >> (32 - HBITS));
  const uint slot = bstart[h] + atomic_inc(&hfill[h]);
  srt[((size_t)b << BLOG) + slot] = p | (((hv >> (32 - HBITS - 12)) & 0xFFFu) << 20);
}

// the common prefix of the bytes at a and at b, up to maxlen, 4 bytes a step
inline uint match_len(__global const uchar* base, uint a, uint b, uint maxlen) {
  uint l = 0;
  while (l + 4 <= maxlen) {
    const uint x = load4u(base, a + l) ^ load4u(base, b + l);
    if (x) return l + ((31u - clz(x & (0u - x))) >> 3);  // (the lowest differing byte: little endian)
    l += 4;
  }
  if (l < maxlen) {
    const uint x = load4u(base, a + l) ^ load4u(base, b + l);
    const uint tz = x ? ((31u - clz(x & (0u - x))) >> 3) : 4u;
    return l + min(tz, maxlen - l);
  }
  return l;
}

#ifndef NCANDL
#define NCANDL 8u  // bucket neighbours checked before the position in srt (mostly earlier positions)
#endif
#ifndef NCANDR
#define NCANDR 2u  // and after it
#endif
#ifndef LENCAP
#define LENCAP 16u  // the match finder compares this far; k_parse finishes the matches the chain really uses
#endif
#define NCAND (NCANDL + NCANDR)

// 4. the best match of every position among its neighbours in srt (the longest up to LENCAP, the nearest on a tie);
//    match[p] = (len << 20) | (dist - 1), or 0 when shorter than MINMATCH. It walks srt, not the positions: the
//    neighbours are then consecutive memory, and the bucket table and an index by position are not needed. A
//    neighbour with another tag, or at a later position, costs no read of the data, and every candidate's first 4
//    bytes are requested before any is looked at (loads in flight together, not one latency each; the compiler keeps
//    source order). The cost is what remains: comparing, so a long match is not compared from each of its positions.
__kernel void k_match(__global const uchar* in, __global const uint* srt, __global uint* match, uint nb, uint nlast) {
  const uint gid = get_global_id(0);
  const uint b = gid >> BLOG;
  if (b >= nb) return;
  const uint n = block_len(b, nb, nlast);
  const uint count = n >= MINMATCH ? n - (MINMATCH - 1) : 0u;  // positions in srt
  const uint i = gid & (B - 1);
  if (i >= n) return;
  if (i >= count) { match[gid] = 0; return; }  // (the last positions of a block are not in srt: exactly these slots)
  __global const uchar* base = in + ((size_t)b << BLOG);
  __global const uint* sb = srt + ((size_t)b << BLOG);
  const uint e = sb[i];
  const uint p = e & 0xFFFFFu;
  uint jc[NCAND], any = 0, nc = 0;
  for (uint k = 1; k <= max(NCANDL, NCANDR); ++k) {
    if (k <= NCANDL) {
      const uint c = i >= k ? sb[i - k] : ~e;
      const uint j = c & 0xFFFFFu;
      const uint ok = ((c ^ e) >> 20) == 0 && j < p;
      jc[nc++] = ok ? j : p;
      any |= ok;
    }
    if (k <= NCANDR) {
      const uint c = i + k < count ? sb[i + k] : ~e;
      const uint j = c & 0xFFFFFu;
      const uint ok = ((c ^ e) >> 20) == 0 && j < p;
      jc[nc++] = ok ? j : p;
      any |= ok;
    }
  }
  uint best = 0, bestd = 0;
  if (any) {
    __global const uint* w = (__global const uint*)base;
    uint w0[NCAND], w1[NCAND];
    const uint pw0 = w[p >> 2], pw1 = w[(p >> 2) + 1];
    for (uint m = 0; m < NCAND; ++m) w0[m] = w[jc[m] >> 2];
    for (uint m = 0; m < NCAND; ++m) w1[m] = w[(jc[m] >> 2) + 1];
    const uint ps = (p & 3u) * 8u;
    const uint pw = ps == 0 ? pw0 : ((pw0 >> ps) | (pw1 << (32u - ps)));
    const uint maxlen = min(LENCAP, n - p);
    if (p > 0 && load4u(base, p - 1) == pw) {  // the byte before: a run of one value costs no offset bits
      best = 4 + match_len(base, p + 3, p + 4, maxlen - 4);
      bestd = 1;
    }
    for (uint m = 0; m < NCAND && best < maxlen; ++m) {
      const uint j = jc[m];
      const uint sh = (j & 3u) * 8u;
      const uint x = sh == 0 ? w0[m] : ((w0[m] >> sh) | (w1[m] << (32u - sh)));
      if (j == p || x != pw) continue;
      if (best >= 4 && base[j + best] != base[p + best]) continue;
      const uint l = 4 + match_len(base, j + 4, p + 4, maxlen - 4);
      const uint d = p - j;
      if (l > best || (l == best && d < bestd)) { best = l; bestd = d; }
    }
  }
  match[((size_t)b << BLOG) + p] = best >= MINMATCH ? ((best << 20) | (bestd - 1)) : 0u;
}

// the position after the token at p. The match finder stops comparing at LENCAP bytes (a long match would otherwise
// be compared again from every one of its positions); the few positions the chain really visits finish it here.
inline uint nextpos(__global uint* m, __global const uchar* base, uint n, uint p) {
  const uint v = m[p];
  uint l = v >> 20;
  if (l == LENCAP) {
    const uint maxlen = min(MAXMATCH, n - p);
    if (maxlen > l) {
      const uint j = p - (v & 0xFFFFFu) - 1u;
      const uint x = l + match_len(base, j + l, p + l, maxlen - l);
      if (x != l) { m[p] = (x << 20) | (v & 0xFFFFFu); l = x; }
    }
  }
  return p + (l ? l : 1u);
}

// 5. the greedy parse: the chain 0 -> next(0) -> ... of positions where a token starts (mark != 0 at the end).
//    Each work-item walks its segment from the segment's start (tentative marks, 1); work-item 0 then follows the
//    true chain through the segments: where it meets a tentative walk the two coincide from there on, so only the
//    tentative prefix is undone; where it never meets one the tentative walk is undone whole. (Greedy parses
//    re-synchronize within a few tokens on ordinary data, so the sequential part is short.) `mark` starts zeroed.
__kernel void k_parse(__global uint* match, __global uchar* mark, uint nb, uint nlast, __global const uchar* in) {
  const uint b = get_group_id(0);
  const uint k = get_local_id(0);
  const uint n = block_len(b, nb, nlast);
  __global uint* m = match + ((size_t)b << BLOG);
  __global const uchar* base = in + ((size_t)b << BLOG);
  __global uchar* mk = mark + ((size_t)b << BLOG);
  __local uint ex[G];
  uint s = k * SEG;
  uint send = min(s + SEG, n);
  uint p = s;
  if (s < n)
    while (p < send) { mk[p] = 1; p = nextpos(m, base, n, p); }
  ex[k] = p;
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  if (k != 0) return;
  uint e = 0;  // where the true chain enters the segments still to do
  for (uint q = 0; q < G; ++q) {
    s = q * SEG;
    if (s >= n) break;
    send = min(s + SEG, n);
    if (e >= send) {  // the true chain jumps over this segment: its tentative walk is all wrong
      for (uint r = s; r < send; r = nextpos(m, base, n, r)) if (mk[r] == 1) mk[r] = 0;
      continue;
    }
    uint t = e, merged = 0, mpos = 0;
    while (t < send) {
      if (mk[t] == 1) { merged = 1; mpos = t; break; }
      mk[t] = 2;
      t = nextpos(m, base, n, t);
    }
    if (merged) {
      for (uint r = s; r != mpos; r = nextpos(m, base, n, r)) mk[r] = 0;
      e = ex[q];
    } else {
      for (uint r = s; r < send; r = nextpos(m, base, n, r)) if (mk[r] == 1) mk[r] = 0;
      e = t;
    }
  }
}

// 6. the chain -> literals and matches: lit (the literal bytes in order), per match the literals before it (tokLB),
//    its length and distance; cnt[b*8+0] tokens, cnt[b*8+1] literals
__kernel void k_tokens(__global const uchar* in, __global const uint* match, __global const uchar* mark, __global uchar* lit,
                       __global uint* tokLB, __global uint* tokM, __global uint* tokD, __global uint* cnt, uint nb, uint nlast) {
  const uint b = get_group_id(0);
  const uint k = get_local_id(0);
  const uint n = block_len(b, nb, nlast);
  __global const uchar* inb = in + ((size_t)b << BLOG);
  __global const uint* m = match + ((size_t)b << BLOG);
  __global const uchar* mk = mark + ((size_t)b << BLOG);
  __global uchar* litb = lit + ((size_t)b << BLOG);
  __global uint* LB = tokLB + (size_t)b * TCAP;
  __global uint* TM = tokM + (size_t)b * TCAP;
  __global uint* TD = tokD + (size_t)b * TCAP;
  __local uint ct[G], cl[G];
  const uint s = k * SEG;
  const uint send = min(s + SEG, n);
  uint nt = 0, nl = 0;
  for (uint p = s; p < send; ++p)
    if (mk[p]) { if (m[p] >> 20) ++nt; else ++nl; }
  ct[k] = nt;
  cl[k] = nl;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (k == 0) {
    uint a = 0, c = 0;
    for (uint i = 0; i < G; ++i) { const uint x = ct[i], y = cl[i]; ct[i] = a; cl[i] = c; a += x; c += y; }
    cnt[b * 8 + 0] = a;
    cnt[b * 8 + 1] = c;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint ti = ct[k], li = cl[k];
  for (uint p = s; p < send; ++p) {
    if (!mk[p]) continue;
    const uint mv = m[p];
    if (mv >> 20) { LB[ti] = li; TM[ti] = mv >> 20; TD[ti] = (mv & 0xFFFFFu) + 1u; ++ti; }
    else litb[li++] = inb[p];
  }
}

inline uint off_bits(uint d) { return 31u - clz(d); }

// 7. the token streams: tok, ext, offsym, and the offset bits (offw: uint words, zeroed before);
//    cnt[b*8+2] ext bytes, cnt[b*8+3] offset-bit bytes
__kernel void k_streams(__global const uint* tokLB, __global const uint* tokM, __global const uint* tokD, __global uint* cnt,
                        __global uchar* tok, __global uchar* ext, __global uchar* offsym, __global uint* offw, uint nb) {
  const uint b = get_group_id(0);
  const uint k = get_local_id(0);
  const uint nt = cnt[b * 8 + 0];
  __global const uint* LB = tokLB + (size_t)b * TCAP;
  __global const uint* TM = tokM + (size_t)b * TCAP;
  __global const uint* TD = tokD + (size_t)b * TCAP;
  __global uchar* tokb = tok + (size_t)b * TCAP;
  __global uchar* extb = ext + (size_t)b * ECAP;
  __global uchar* osb = offsym + (size_t)b * TCAP;
  __global uint* ow = offw + (size_t)b * (OCAP / 4u);
  __local uint ce[G], cb[G];
  const uint chunk = (nt + G - 1) / G;
  const uint t0 = min(k * chunk, nt), t1 = min(t0 + chunk, nt);
  uint ne = 0, nbits = 0;
  for (uint t = t0; t < t1; ++t) {
    const uint L = LB[t] - (t ? LB[t - 1] : 0u), M = TM[t];
    if (L >= 15) ne += 1 + (L - 15) / 255;
    if (M - 4 >= 15) ne += 1;
    nbits += off_bits(TD[t]);
  }
  ce[k] = ne;
  cb[k] = nbits;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (k == 0) {
    uint a = 0, c = 0;
    for (uint i = 0; i < G; ++i) { const uint x = ce[i], y = cb[i]; ce[i] = a; cb[i] = c; a += x; c += y; }
    cnt[b * 8 + 2] = a;
    cnt[b * 8 + 3] = (c + 7) / 8;
    cnt[b * 8 + 4] = c;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint ei = ce[k], bp = cb[k];
  for (uint t = t0; t < t1; ++t) {
    const uint L = LB[t] - (t ? LB[t - 1] : 0u), M = TM[t], D = TD[t];
    tokb[t] = (uchar)((min(L, 15u) << 4) | min(M - 4, 15u));
    if (L >= 15) {
      uint rem = L - 15;
      while (rem >= 255) { extb[ei++] = 255; rem -= 255; }
      extb[ei++] = (uchar)rem;
    }
    if (M - 4 >= 15) extb[ei++] = (uchar)(M - 4 - 15);
    const uint bb = off_bits(D);
    osb[t] = (uchar)bb;
    if (bb) {
      const uint extra = D - (1u << bb);
      const uint w = bp >> 5, sh = bp & 31;
      atomic_or(&ow[w], extra << sh);
      if (sh + bb > 32) atomic_or(&ow[w + 1], extra >> (32 - sh));
      bp += bb;
    }
  }
}

// the stream s of block b: its symbols, count and lane scratch
inline __global const uchar* stream_sym(uint s, uint b, __global const uchar* lit, __global const uchar* tok,
                                        __global const uchar* ext, __global const uchar* offsym) {
  if (s == 0) return lit + ((size_t)b << BLOG);
  if (s == 1) return tok + (size_t)b * TCAP;
  if (s == 2) return ext + (size_t)b * ECAP;
  return offsym + (size_t)b * TCAP;
}
inline uint stream_n(uint s, uint b, __global const uint* cnt) {
  if (s == 0) return cnt[b * 8 + 1];
  if (s == 2) return cnt[b * 8 + 2];
  return cnt[b * 8 + 0];
}
inline uint lane_cap(uint s) { return s == 0 ? LC0 : s == 2 ? LC2 : LC1; }
inline uint lane_off(uint s) { return s == 0 ? 0u : s == 1 ? NL * LC0 : s == 2 ? NL * (LC0 + LC1) : NL * (LC0 + LC1 + LC2); }

// 8. rANS: histogram, 12-bit frequencies (the same steps as the CPU), 64 lanes each encoding its symbols in reverse
//    into its scratch region (backwards). sinfo per (block, stream): kind, body length, present symbols, symbols.
__kernel void k_rans_enc(__global const uchar* lit, __global const uchar* tok, __global const uchar* ext, __global const uchar* offsym,
                         __global const uint* cnt, __global uchar* lanes, __global ushort* freqg, __global uint* lanelen,
                         __global uint* sinfo, uint nb) {
  const uint g = get_group_id(0);
  const uint b = g >> 2, s = g & 3;
  const uint k = get_local_id(0);
  __global const uchar* sym = stream_sym(s, b, lit, tok, ext, offsym);
  const uint n = stream_n(s, b, cnt);
  __global uchar* lane_base = lanes + (size_t)b * LANES_PER_BLOCK + lane_off(s);
  const uint LC = lane_cap(s);
  __local uint hist[256], freq[256], cum[256];
  __local uint np_l;
  for (uint i = k; i < 256; i += G) hist[i] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint i = k; i < n; i += G) atomic_inc(&hist[sym[i]]);
  barrier(CLK_LOCAL_MEM_FENCE);
  if (k == 0) {
    uint sum = 0, np = 0;
    for (uint c = 0; c < 256; ++c) {
      uint f = 0;
      if (hist[c]) {
        ++np;
        f = (uint)(((ulong)hist[c] << 12) / (ulong)n);
        if (f == 0) f = 1;
      }
      freq[c] = f;
      sum += f;
    }
    while (sum > SCALE) {
      int best = -1;
      for (int c = 0; c < 256; ++c)
        if (freq[c] > 1 && (best < 0 || freq[c] > freq[best])) best = c;
      --freq[best];
      --sum;
    }
    if (sum < SCALE) {
      int best = 0;
      for (int c = 1; c < 256; ++c)
        if (freq[c] > freq[best]) best = c;
      freq[best] += SCALE - sum;
    }
    uint c0 = 0;
    for (uint c = 0; c < 256; ++c) { cum[c] = c0; c0 += freq[c]; }
    np_l = np;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint i = k; i < 256; i += G) freqg[(size_t)g * 256 + i] = (ushort)freq[i];
  if (n == 0) {
    if (k == 0) { sinfo[g * 4 + 0] = 0; sinfo[g * 4 + 1] = 0; sinfo[g * 4 + 2] = 0; sinfo[g * 4 + 3] = 0; }
    return;
  }
  if (k < NL) {
    __global uchar* end = lane_base + (size_t)(k + 1) * LC;
    __global uchar* ptr = end;
    uint x = RANS_L;
    if (k < n) {
      uint i = k + ((n - 1 - k) / NL) * NL;  // the lane's last symbol
      for (;;) {
        const uint c = sym[i];
        const uint f = freq[c];
        const uint xmax = f << 19;
        while (x >= xmax) { *--ptr = (uchar)x; x >>= 8; }
        x = ((x / f) << 12) + (x % f) + cum[c];
        if (i < NL) break;
        i -= NL;
      }
    }
    ptr -= 4;
    ptr[0] = (uchar)x; ptr[1] = (uchar)(x >> 8); ptr[2] = (uchar)(x >> 16); ptr[3] = (uchar)(x >> 24);
    lanelen[(size_t)g * NL + k] = (uint)(end - ptr);
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  if (k == 0) {
    uint total = 32 + 2 * np_l + 4 * NL;
    for (uint j = 0; j < NL; ++j) total += lanelen[(size_t)g * NL + j];
    const uint kind = total >= n ? 0u : 1u;
    sinfo[g * 4 + 0] = kind;
    sinfo[g * 4 + 1] = kind ? total : n;
    sinfo[g * 4 + 2] = np_l;
    sinfo[g * 4 + 3] = n;
  }
}

// 9. the block payload, as src/codec.h lays it out; outlen[b] = its size, or 0xFFFFFFFF when not smaller than the block
__kernel void k_pack(__global const uchar* lit, __global const uchar* tok, __global const uchar* ext, __global const uchar* offsym,
                     __global const uchar* offb, __global const uint* cnt, __global const uchar* lanes, __global const ushort* freqg,
                     __global const uint* lanelen, __global const uint* sinfo, __global uchar* out, __global uint* outlen, uint nb,
                     uint nlast) {
  const uint b = get_group_id(0);
  const uint k = get_local_id(0);
  const uint n = block_len(b, nb, nlast);
  const uint nob = cnt[b * 8 + 3];
  __global uchar* o = out + (size_t)b * OUTCAP;
  uint off[5];
  off[0] = 16;
  for (uint s = 0; s < 4; ++s) off[s + 1] = off[s] + 5 + sinfo[(b * 4 + s) * 4 + 1];
  const uint total = off[4] + nob;
  if (total >= n) {
    if (k == 0) outlen[b] = 0xFFFFFFFFu;
    return;
  }
  if (k == 0) {
    put32g(o, cnt[b * 8 + 0]);
    put32g(o + 4, cnt[b * 8 + 1]);
    put32g(o + 8, cnt[b * 8 + 2]);
    put32g(o + 12, nob);
    outlen[b] = total;
  }
  for (uint s = 0; s < 4; ++s) {
    const uint g = b * 4 + s;
    const uint kind = sinfo[g * 4 + 0], len = sinfo[g * 4 + 1], np = sinfo[g * 4 + 2], ns = sinfo[g * 4 + 3];
    __global uchar* d = o + off[s];
    if (k == 0) { d[0] = (uchar)kind; put32g(d + 1, len); }
    __global const uchar* sym = stream_sym(s, b, lit, tok, ext, offsym);
    if (kind == 0) {
      for (uint i = k; i < ns; i += G) d[5 + i] = sym[i];
      continue;
    }
    if (k < 32) {
      uint bits = 0;
      for (uint j = 0; j < 8; ++j) bits |= (freqg[(size_t)g * 256 + k * 8 + j] != 0 ? 1u : 0u) << j;
      d[5 + k] = (uchar)bits;
    }
    if (k == 0) {
      uint idx = 0;
      for (uint c = 0; c < 256; ++c) {
        const uint f = freqg[(size_t)g * 256 + c];
        if (f) { put16g(d + 5 + 32 + 2 * idx, f); ++idx; }
      }
    }
    if (k < NL) put32g(d + 5 + 32 + 2 * np + 4 * k, lanelen[(size_t)g * NL + k]);
    uint doff = 5 + 32 + 2 * np + 4 * NL;
    __global const uchar* lane_base = lanes + (size_t)b * LANES_PER_BLOCK + lane_off(s);
    const uint LC = lane_cap(s);
    for (uint j = 0; j < NL; ++j) {
      const uint ll = lanelen[(size_t)g * NL + j];
      __global const uchar* src = lane_base + (size_t)(j + 1) * LC - ll;
      for (uint i = k; i < ll; i += G) d[doff + i] = src[i];
      doff += ll;
    }
  }
  __global const uchar* ob = offb + (size_t)b * OCAP;
  for (uint i = k; i < nob; i += G) o[off[4] + i] = ob[i];
}

// ============================================================ decoder

// error codes (err[b]); the host turns them into "block damaged"
#define E_HDR 1u
#define E_COUNTS 2u
#define E_STREAM 3u
#define E_TABLE 4u
#define E_LANE 5u
#define E_TOKEN 6u

// 1. the four symbol streams of a block (one work-group of 64 per stream): raw copied, rANS decoded by 64 lanes
__kernel void k_dec_streams(__global const uchar* comp, __global const uint* boff, __global const uint* binfo, __global uchar* lit,
                            __global uchar* tok, __global uchar* ext, __global uchar* offsym, __global uint* dcnt, __global uint* err,
                            uint nb) {
  const uint g = get_group_id(0);
  const uint b = g >> 2, s = g & 3;
  const uint k = get_local_id(0);
  const uint GS = get_local_size(0);  // (launched with 64 work-items: the rANS lanes)
  const uint raw = binfo[b * 4 + 0], clen = binfo[b * 4 + 1], method = binfo[b * 4 + 2];
  if (method == 0) return;
  __global const uchar* p = comp + boff[b];
  __local uint freq[256], cum[256], lanestart[NL + 1];
  __local uchar slot[SCALE];
  __local uint fail;
  if (k == 0) fail = 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (clen < 16) { if (k == 0) err[b] = E_HDR; return; }
  const uint nt = get32g(p), nl = get32g(p + 4), ne = get32g(p + 8), nob = get32g(p + 12);
  if (nt > TCAP || nl > raw || ne > ECAP || nob > OCAP) { if (k == 0) err[b] = E_COUNTS; return; }
  uint pos = 16;
  for (uint t = 0; t < s; ++t) {
    if (pos + 5 > clen) { if (k == 0) err[b] = E_STREAM; return; }
    pos += 5 + get32g(p + pos + 1);
  }
  if (pos + 5 > clen) { if (k == 0) err[b] = E_STREAM; return; }
  const uint kind = p[pos], len = get32g(p + pos + 1);
  if (pos + 5 + len > clen) { if (k == 0) err[b] = E_STREAM; return; }
  __global const uchar* body = p + pos + 5;
  if (s == 3 && k == 0) {
    dcnt[b * 8 + 0] = nt; dcnt[b * 8 + 1] = nl; dcnt[b * 8 + 2] = ne; dcnt[b * 8 + 3] = nob;
    dcnt[b * 8 + 4] = pos + 5 + len;
    if (pos + 5 + len + nob > clen) err[b] = E_STREAM;
  }
  __global uchar* dest;
  uint ns;
  if (s == 0) { dest = lit + ((size_t)b << BLOG); ns = nl; }
  else if (s == 1) { dest = tok + (size_t)b * TCAP; ns = nt; }
  else if (s == 2) { dest = ext + (size_t)b * ECAP; ns = ne; }
  else { dest = offsym + (size_t)b * TCAP; ns = nt; }
  if (kind == 0) {
    if (len != ns) { if (k == 0) err[b] = E_STREAM; return; }
    for (uint i = k; i < ns; i += GS) dest[i] = body[i];
    return;
  }
  if (kind != 1 || ns == 0 || len < 32 + 4 * NL) { if (k == 0) err[b] = E_STREAM; return; }
  // the table (bitmap, frequencies, lane lengths: at most 32 + 512 + 256 bytes) into local memory first
  __local uchar hdr[800];
  const uint hn = min(len, 800u);
  for (uint i = k; i < hn; i += GS) hdr[i] = body[i];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (k == 0) {
    uint idx = 0, c0 = 0;
    for (uint c = 0; c < 256; ++c) {
      uint f = 0;
      if ((hdr[c >> 3] >> (c & 7)) & 1) {
        if (32 + 2 * (idx + 1) + 4 * NL > len) { fail = E_TABLE; break; }
        f = (uint)hdr[32 + 2 * idx] | ((uint)hdr[33 + 2 * idx] << 8);
        ++idx;
      }
      freq[c] = f;
      cum[c] = c0;
      c0 += f;
    }
    if (!fail && c0 != SCALE) fail = E_TABLE;
    if (!fail) {
      uint ls = 32 + 2 * idx + 4 * NL;
      const uint lp = 32 + 2 * idx;
      for (uint j = 0; j < NL; ++j) {
        lanestart[j] = ls;
        const uint q = lp + 4 * j;
        ls += (uint)hdr[q] | ((uint)hdr[q + 1] << 8) | ((uint)hdr[q + 2] << 16) | ((uint)hdr[q + 3] << 24);
        if (ls > len) { fail = E_LANE; break; }
      }
      lanestart[NL] = ls;
    }
    if (fail) err[b] = fail;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (fail) return;
  for (uint c = k; c < 256; c += GS)
    for (uint q = 0; q < freq[c]; ++q) slot[cum[c] + q] = (uchar)c;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (k < NL) {
    __global const uchar* in = body + lanestart[k];
    __global const uchar* in_end = body + lanestart[k + 1];
    if (in_end - in < 4) { err[b] = E_LANE; return; }
    uint x = get32g(in);
    in += 4;
    for (uint i = k; i < ns; i += NL) {
      const uint sl = x & (SCALE - 1);
      const uint c = slot[sl];
      dest[i] = (uchar)c;
      x = freq[c] * (x >> 12) + sl - cum[c];
      while (x < RANS_L) {
        if (in >= in_end) { err[b] = E_LANE; return; }
        x = (x << 8) | *in++;
      }
    }
  }
}

// 2. the LZ replay: one sub-group of 16 lanes per block (16 blocks per work-group), the lanes walking the tokens
//    together and each copying a share of every token's bytes. The token streams are staged in local memory in
//    chunks of DCH bytes (the per-token reads are then local, not global); long literal runs and non-overlapping
//    long matches are copied 4 bytes a lane. An overlapping match (distance < length) repeats the distance-long
//    pattern before it, so every byte of a token is independent of the token's other bytes. A match that reads
//    bytes written since the last fence takes a global memory fence first (the lanes are one hardware thread, so
//    nothing else is needed); matches that reach further back, and literals, need none. Stored blocks are copied
//    256 bytes a step.
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#define SGW 16u
#define NSG (G / SGW)
#define DCH 256u
// a DCH-byte chunk at src (4-aligned) into the sub-group's local chunk
inline void stage4(__local uint* dst, __global const uchar* src, uint k) {
  __global const uint* s = (__global const uint*)src;
  for (uint q = 0; q < DCH / 4 / SGW; ++q) dst[k + SGW * q] = s[k + SGW * q];
  sub_group_barrier(CLK_LOCAL_MEM_FENCE);
}
// the same from an unaligned source
inline void stage1(__local uint* dst, __global const uchar* src, uint k) {
  __local uchar* d = (__local uchar*)dst;
  for (uint q = 0; q < DCH / 4 / SGW; ++q) vstore4(vload4(0, src + 4 * (k + SGW * q)), 0, d + 4 * (k + SGW * q));
  sub_group_barrier(CLK_LOCAL_MEM_FENCE);
}
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void k_dec_lz(__global const uchar* comp, __global const uint* boff, __global const uint* binfo, __global const uchar* lit,
                       __global const uchar* tok, __global const uchar* ext, __global const uchar* offsym, __global const uint* dcnt,
                       __global uchar* out, __global uint* err, uint nb) {
  const uint sg = get_sub_group_id();
  const uint b = get_group_id(0) * NSG + sg;
  __local uint l_tok[NSG][DCH / 4], l_os[NSG][DCH / 4], l_ext[NSG][DCH / 4], l_ob[NSG][DCH / 4];
  if (b >= nb) return;
  const uint k = get_sub_group_local_id();
  const uint raw = binfo[b * 4 + 0], method = binfo[b * 4 + 2];
  __global uchar* o = out + ((size_t)b << BLOG);
  __global const uchar* p = comp + boff[b];
  if (method == 0) {
    uint i = k * 16;
    for (; i + 16 <= raw; i += SGW * 16) vstore16(vload16(0, p + i), 0, o + i);
    for (i = (raw & ~255u) + k; i < raw; i += SGW) o[i] = p[i];
    return;
  }
  if (err[b]) return;
  const uint nt = dcnt[b * 8 + 0], nl = dcnt[b * 8 + 1], ne = dcnt[b * 8 + 2], nob = dcnt[b * 8 + 3];
  __global const uchar* offb = p + dcnt[b * 8 + 4];
  __global const uchar* litb = lit + ((size_t)b << BLOG);
  __global const uchar* tokb = tok + (size_t)b * TCAP;
  __global const uchar* extb = ext + (size_t)b * ECAP;
  __global const uchar* osb = offsym + (size_t)b * TCAP;
  __local const uchar* s_tok = (__local const uchar*)l_tok[sg];
  __local const uchar* s_os = (__local const uchar*)l_os[sg];
  __local const uchar* s_ext = (__local const uchar*)l_ext[sg];
  __local const uchar* s_ob = (__local const uchar*)l_ob[sg];
  uint tc0 = 0, ec0 = 0, ob0 = 0;
  stage4(l_tok[sg], tokb, k);
  stage4(l_os[sg], osb, k);
  stage4(l_ext[sg], extb, k);
  stage1(l_ob[sg], offb, k);
  uint op = 0, li = 0, ei = 0, bp = 0;
  uint fenced = 0;  // the output before this point is visible to every lane
  for (uint t = 0; t < nt; ++t) {
    if (t - tc0 >= DCH) {
      tc0 = t & ~(DCH - 1);
      stage4(l_tok[sg], tokb + tc0, k);
      stage4(l_os[sg], osb + tc0, k);
    }
    const uint tv = s_tok[t - tc0];
    uint L = tv >> 4, M = (tv & 15) + 4;
    if (L == 15) {
      uint x;
      do {
        if (ei >= ne) { if (k == 0) err[b] = E_TOKEN; return; }
        if (ei - ec0 >= DCH) { ec0 = ei & ~(DCH - 1); stage4(l_ext[sg], extb + ec0, k); }
        x = s_ext[ei - ec0];
        ++ei;
        L += x;
      } while (x == 255);
    }
    if (M - 4 == 15) {
      if (ei >= ne) { if (k == 0) err[b] = E_TOKEN; return; }
      if (ei - ec0 >= DCH) { ec0 = ei & ~(DCH - 1); stage4(l_ext[sg], extb + ec0, k); }
      M += s_ext[ei - ec0];
      ++ei;
    }
    const uint bb = s_os[t - tc0];
    if (bb > 20) { if (k == 0) err[b] = E_TOKEN; return; }
    uint extra = 0;
    if (bb) {
      if ((bp + bb + 7) / 8 > nob) { if (k == 0) err[b] = E_TOKEN; return; }
      const uint byte = bp >> 3;
      // (the 4 bytes read must lie inside the staged chunk: a new chunk starts at this byte, any alignment)
      if (byte + 4 > ob0 + DCH || byte < ob0) { ob0 = byte; stage1(l_ob[sg], offb + ob0, k); }
      const uint r = byte - ob0;
      const uint w = (uint)s_ob[r] | ((uint)s_ob[r + 1] << 8) | ((uint)s_ob[r + 2] << 16) | ((uint)s_ob[r + 3] << 24);
      extra = (w >> (bp & 7)) & ((1u << bb) - 1);
      bp += bb;
    }
    const uint D = (1u << bb) + extra;
    if (li + L > nl || op + L + M > raw || D > op + L) { if (k == 0) err[b] = E_TOKEN; return; }
    if (L >= 64) {
      for (uint i = k * 4; i + 4 <= L; i += SGW * 4) vstore4(vload4(0, litb + li + i), 0, o + op + i);
      for (uint i = (L & ~3u) + k; i < L; i += SGW) o[op + i] = litb[li + i];
    } else {
      for (uint i = k; i < L; i += SGW) o[op + i] = litb[li + i];
    }
    const uint mstart = op + L;
    // the match reads o[mstart - D .. mstart - D + min(M, D)) (its literal part comes from litb): fence if any of
    // that was written since the last fence
    if (mstart - D + min(M, D) > fenced) {
      sub_group_barrier(CLK_GLOBAL_MEM_FENCE);
      fenced = mstart;
    }
    if (M >= 64 && D >= M && mstart - D + M <= op) {  // the source lies wholly in earlier tokens' output
      for (uint i = k * 4; i + 4 <= M; i += SGW * 4) vstore4(vload4(0, o + mstart - D + i), 0, o + mstart + i);
      for (uint i = (M & ~3u) + k; i < M; i += SGW) o[mstart + i] = o[mstart - D + i];
    } else {
      for (uint i = k; i < M; i += SGW) {
        const uint src = mstart - D + (D >= M ? i : (i % D));
        o[mstart + i] = src >= op ? litb[li + (src - op)] : o[src];
      }
    }
    op += L + M;
    li += L;
  }
  const uint rem = nl - li;
  if (op + rem != raw) { if (k == 0) err[b] = E_TOKEN; return; }
  for (uint i = k; i < rem; i += SGW) o[op + i] = litb[li + i];
}
