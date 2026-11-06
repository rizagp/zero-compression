// zero: the GPU kernels of the ZB2 block codec (see src/codec2.h for the format). OpenCL C, compiled by the Intel
// OpenCL runtime and run through Level Zero. Built with -DBLOG=<log2 block size> -DG=<work-group size>.
//
// A batch holds nb blocks of B = 2^BLOG bytes (the last one nlast bytes) laid out one after the other; per-block
// arrays are indexed by block then position. One work-group (G work-items) handles one block in the per-block
// kernels; the per-position kernels run over every position of the batch.

#define B (1u << BLOG)
#ifndef HBITS
#define HBITS (BLOG - 3)
#endif
#define HB (1u << HBITS)
#define TCAP (B / 4u + 16u)                // tokens per block at most (a multiple of 16: aligned chunk loads)
#define ECAP (B / 5u + 64u)                // ext bytes per block at most: 3 for every 19 bytes of output
#define OCAP (5u * B / 8u + 8u)            // offset-bit bytes per block at most (a multiple of 4)
#define OUTCAP (B + B / 8u + 8192u)        // payload slab per block
#define MASK (B - 1u)
#define NL 64u                             // rANS lanes
#define LC0 (2u * (B / NL) + 16u)          // lane scratch, lit
#define LC1 (2u * (TCAP / NL + 1u) + 16u)  // tok, offsym
#define LC2 (2u * (ECAP / NL + 1u) + 16u)  // ext
#define LANES_PER_BLOCK (NL * (LC0 + 3u * LC1 + LC2))   // lit, tok, ext, sym, groups
#define MINMATCH 4u
#define MAXMATCH 4095u  // (12 bits in match[]; the format allows more)
#define SEG (B / G)
#define RANS_L (1u << 23)
#define SCALE 4096u

inline uint off_bits_of(uint d) { return 31u - clz(d); }
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

#ifndef HASHLEN
#define HASHLEN 5   // (measured: 4 and 6 lose more than they gain, see docs/how-it-works.md)
#endif
inline uint hash_mul(__global const uchar* base, uint pos) {
  const uint v = load4u(base, pos);
#if HASHLEN == 4
  return v * 2654435761u;
#else
  // (the bytes past the 4th, as far as the block's slack allows: the input buffer has 16 bytes after its end)
  const uint w = load4u(base, pos + 4u) & ((1u << (8 * (HASHLEN - 4))) - 1u);
  return (v * 2654435761u) ^ (w * 0x9E3779B1u + (w >> 3));
#endif
}
// 1. every position with 4 bytes left counts itself in its block's hash bucket
__kernel void k_hash_count(__global const uchar* in, __global uint* hcount, uint nb, uint nlast) {
  const uint gid = get_global_id(0);
  const uint b = gid >> BLOG;
  if (b >= nb) return;
  const uint p = gid & (B - 1);
  if (p + 4 > block_len(b, nb, nlast)) return;
  atomic_inc(&hcount[b * HB + (hash_mul(in, gid) >> (32 - HBITS))]);
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

// 3. positions grouped by bucket, each with 12 more bits of its hash in the top bits
__kernel void k_scatter(__global const uchar* in, __global const uint* bstart, __global uint* hfill, __global uint* srt,
                        uint nb, uint nlast) {
  const uint gid = get_global_id(0);
  const uint b = gid >> BLOG;
  if (b >= nb) return;
  const uint p = gid & (B - 1);
  if (p + 4 > block_len(b, nb, nlast)) return;
  const uint hv = hash_mul(in, gid);
  const uint h = b * HB + (hv >> (32 - HBITS));
  const uint slot = bstart[h] + atomic_inc(&hfill[h]);
  srt[((size_t)b << BLOG) + slot] = p | (((hv >> (32 - HBITS - 12)) & 0xFFFu) << 20);
}

// 3b. each bucket sorted by its tag, then by position, in registers: one read and one write per bucket. A position's left
//     neighbours are then the nearest earlier positions with its tag: real candidates, with none wasted. A bucket longer than
//     16 is sorted in overlapping windows, which is nearly as good: its neighbours are what counts.
#define CE(i, j) { const uint lo_ = min(a[i], a[j]), hi_ = max(a[i], a[j]); a[i] = lo_; a[j] = hi_; }
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void k_sort(__global const uint* bstart, __global const uint* hcount, __global uint* srt, uint nb) {
  const uint gid = get_global_id(0);
  const uint b = gid >> HBITS;
  if (b >= nb) return;
  const uint hc = hcount[gid];
  if (hc < 2u) return;
  const uint lo = bstart[gid];
  __global uint* sb = srt + ((size_t)b << BLOG) + lo;
  for (uint at = 0; at + 1u < hc; at += 12u) {
    const uint n = min(hc - at, 16u);
    __global uint* w = sb + at;
    const uint16 v = vload16(0, w);
    uint a[16];
    a[0] = v.s0; a[1] = v.s1; a[2] = v.s2; a[3] = v.s3; a[4] = v.s4; a[5] = v.s5; a[6] = v.s6; a[7] = v.s7;
    a[8] = v.s8; a[9] = v.s9; a[10] = v.sa; a[11] = v.sb; a[12] = v.sc; a[13] = v.sd; a[14] = v.se; a[15] = v.sf;
    for (uint i = n; i < 16u; ++i) a[i] = 0xFFFFFFFFu;  // (what follows belongs to the next bucket: sorted past the end)
    CE(0, 1); CE(2, 3); CE(0, 2); CE(1, 3); CE(1, 2); CE(4, 5);
    CE(6, 7); CE(4, 6); CE(5, 7); CE(5, 6); CE(0, 4); CE(2, 6);
    CE(2, 4); CE(1, 5); CE(3, 7); CE(3, 5); CE(1, 2); CE(3, 4);
    CE(5, 6); CE(8, 9); CE(10, 11); CE(8, 10); CE(9, 11); CE(9, 10);
    CE(12, 13); CE(14, 15); CE(12, 14); CE(13, 15); CE(13, 14); CE(8, 12);
    CE(10, 14); CE(10, 12); CE(9, 13); CE(11, 15); CE(11, 13); CE(9, 10);
    CE(11, 12); CE(13, 14); CE(0, 8); CE(4, 12); CE(4, 8); CE(2, 10);
    CE(6, 14); CE(6, 10); CE(2, 4); CE(6, 8); CE(10, 12); CE(1, 9);
    CE(5, 13); CE(5, 9); CE(3, 11); CE(7, 15); CE(7, 11); CE(3, 5);
    CE(7, 9); CE(11, 13); CE(1, 2); CE(3, 4); CE(5, 6); CE(7, 8);
    CE(9, 10); CE(11, 12); CE(13, 14);
    uint q = 0;
    if (n & 8u) { vstore8((uint8)(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]), 0, w); q = 8u; }
    if (n & 4u) { vstore4((uint4)(a[q], a[q + 1u], a[q + 2u], a[q + 3u]), 0, w + q); q += 4u; }
    if (n & 2u) { vstore2((uint2)(a[q], a[q + 1u]), 0, w + q); q += 2u; }
    if (n & 1u) { w[q] = a[q]; q += 1u; }
    if (n == 16u) { vstore8((uint8)(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]), 0, w); vstore8((uint8)(a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15]), 0, w + 8); }
  }
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
#define NCANDL 4u  // bucket neighbours checked before the position in srt: the nearest earlier ones with its tag
#endif
#ifndef NCANDR
#define NCANDR 0u  // and after it (a sorted bucket has nothing useful there)
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
  const uint p = e & MASK;
  uint jc[NCAND], any = 0, nc = 0;
  for (uint k = 1; k <= max(NCANDL, NCANDR); ++k) {
    if (k <= NCANDL) {
      const uint c = i >= k ? sb[i - k] : ~e;
      const uint j = c & MASK;
      const uint ok = ((c ^ e) >> 20) == 0 && j < p;
      jc[nc++] = ok ? j : p;
      any |= ok;
    }
    if (k <= NCANDR) {
      const uint c = i + k < count ? sb[i + k] : ~e;
      const uint j = c & MASK;
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

// 4b. lazy matching: a match gives way to a better one a byte later. A pure function of the match finder's results, in a
//     pass of its own: the parse's walkers overlap, and must decide alike wherever they meet.
__kernel void k_lazy(__global const uint* match, __global uint* out, uint nb, uint nlast) {
  const uint gid = get_global_id(0);
  const uint b = gid >> BLOG;
  if (b >= nb) return;
  const uint p = gid & MASK;
  const uint n = block_len(b, nb, nlast);
  if (p >= n) return;
  const uint v = match[gid];
  const uint v2 = p + 1u < n ? match[gid + 1u] : 0u;
  const int l = (int)(v >> 20), l2 = (int)(v2 >> 20);
  uint r = v;
  if (l && l2) {
    const int cost = (int)off_bits_of((v & 0xFFFFFu) + 1u), cost2 = (int)off_bits_of((v2 & 0xFFFFFu) + 1u);
    if (8 * l2 - cost2 - 8 > 8 * l - cost) r = 0;
  }
  out[gid] = r;
}

// the position after the token at p. The match finder stops comparing at LENCAP bytes (a long match would otherwise
// be compared again from every one of its positions); the few positions the chain really visits finish it here.
inline uint nextpos(__global uint* m, __global const uchar* base, uint n, uint p) {
  const uint v = m[p];
  uint l = v >> 20;
  if (l == LENCAP) {
    const uint maxlen = min(MAXMATCH, n - p);
    if (maxlen > l) {
      const uint j = (p - (v & 0xFFFFFu) - 1u) & MASK;
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
//    its length and distance; cnt[b*8+0] tokens, cnt[b*8+1] literals. The lanes of a sub-group take consecutive positions,
//    so a load touches one or two cache lines, not sixteen.
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void k_tokens(__global const uchar* in, __global const uint* match, __global const uchar* mark, __global uchar* lit,
                       __global uint* tokLB, __global uint* tokM, __global uint* tokD, __global uint* cnt, uint nb, uint nlast) {
  const uint b = get_group_id(0);
  const uint sg = get_sub_group_id();
  const uint k = get_sub_group_local_id();
  const uint n = block_len(b, nb, nlast);
  __global const uchar* inb = in + ((size_t)b << BLOG);
  __global const uint* m = match + ((size_t)b << BLOG);
  __global const uchar* mk = mark + ((size_t)b << BLOG);
  __global uchar* litb = lit + ((size_t)b << BLOG);
  __global uint* LB = tokLB + (size_t)b * TCAP;
  __global uint* TM = tokM + (size_t)b * TCAP;
  __global uint* TD = tokD + (size_t)b * TCAP;
  __local uint ct[16], cl[16];
  const uint S = B / 16u;
  const uint s0 = sg * S, send = min(s0 + S, n);
  uint nt = 0, nl = 0;
  for (uint p0 = s0; p0 < send; p0 += 16u) {
    const uint p = p0 + k;
    const uint on = p < send && mk[p] != 0;
    const uint tk = on && (m[p] >> 20) != 0;
    nt += sub_group_reduce_add(tk);
    nl += sub_group_reduce_add(on - tk);
  }
  if (k == 0) { ct[sg] = nt; cl[sg] = nl; }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (sg == 0 && k == 0) {
    uint a = 0, c = 0;
    for (uint i = 0; i < 16u; ++i) { const uint x = ct[i], y = cl[i]; ct[i] = a; cl[i] = c; a += x; c += y; }
    cnt[b * 8 + 0] = a;
    cnt[b * 8 + 1] = c;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint ti = sub_group_broadcast(ct[sg], 0u), li = sub_group_broadcast(cl[sg], 0u);
  for (uint p0 = s0; p0 < send; p0 += 16u) {
    const uint p = p0 + k;
    const uint on = p < send && mk[p] != 0;
    const uint mv = on ? m[p] : 0u;
    const uint tk = on && (mv >> 20) != 0;
    const uint lt = on - tk;
    const uint tq = ti + sub_group_scan_exclusive_add(tk), lq = li + sub_group_scan_exclusive_add(lt);
    if (tk && tq < TCAP) { LB[tq] = lq; TM[tq] = mv >> 20; TD[tq] = (mv & 0xFFFFFu) + 1u; }
    else if (lt) litb[lq & MASK] = inb[p];
    ti += sub_group_reduce_add(tk);
    li += sub_group_reduce_add(lt);
  }
}

// 7. the groups and the token streams, one sub-group per block, 16 tokens at a time: tok, ext, sym, grp and the offset bits
//    (offw: uint words, zeroed before). A match that reads what its own group writes closes the group, unless it reads
//    only its own literals. An offset equal to one of the last three of the group before becomes that symbol.
//    cnt[b*8+2] ext bytes, cnt[b*8+3] offset-bit bytes, cnt[b*8+5] groups
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void k_streams(__global const uint* tokLB, __global const uint* tokM, __global const uint* tokD, __global uint* cnt,
                        __global uchar* tok, __global uchar* ext, __global uchar* sym, __global uchar* grp, __global uint* offw, uint nb) {
  const uint b = get_group_id(0) * (G / 16u) + get_sub_group_id();
  if (b >= nb) return;
  const uint k = get_sub_group_local_id();
  const uint nt = min(cnt[b * 8 + 0], TCAP);
  __global const uint* LB = tokLB + (size_t)b * TCAP;
  __global const uint* TM = tokM + (size_t)b * TCAP;
  __global const uint* TD = tokD + (size_t)b * TCAP;
  __global uchar* tokb = tok + (size_t)b * TCAP;
  __global uchar* extb = ext + (size_t)b * ECAP;
  __global uchar* symb = sym + (size_t)b * TCAP;
  __global uchar* grpb = grp + (size_t)b * TCAP;
  __global uint* ow = offw + (size_t)b * (OCAP / 4u);
  uint op = 0, ei = 0, bp = 0, ng = 0, gstart = 0, gt = 0;
  uint r0 = 1, r1 = 4, r2 = 8, l0 = 1, l1 = 4, l2 = 8;
  for (uint t = 0; t < nt; t += 16u) {
    const uint c = min(16u, nt - t);
    const uint on = k < c;
    const uint q = t + k;
    const uint lb = on ? LB[q] : 0u;
    const uint lprev = on && q ? LB[q - 1u] : 0u;
    const uint L = lb - lprev;
    const uint M = on ? TM[q] : 0u;
    const uint D = on ? TD[q] : 1u;
    const uint opk = op + sub_group_scan_exclusive_add(L + M);
    const uint i = opk + L, j = i - D;
    const uint need = j < opk ? j + min(M, D) : 0u;  // how far this token's source reaches, when not its own literals
    uint sv = 0;
    uint lo = 0;
    while (lo < c) {
      // the first token from lo on that cannot join the open group
      const uint viol = on && k >= lo && ((need > gstart && !(k == lo && gt == 0u)) || gt + (k - lo) >= 255u);
      const uint first = min(sub_group_reduce_min(viol ? k : 16u), c);
      if (k >= lo && k < first) sv = D == r0 ? 250u : D == r1 ? 251u : D == r2 ? 252u : 31u - clz(D);
      for (uint e = first > lo + 3u ? first - 3u : lo; e < first; ++e) {
        const uint v = sub_group_broadcast(D, e);
        l2 = l1; l1 = l0; l0 = v;
      }
      gt += first - lo;
      if (first < c) {
        if (gt && k == 0) grpb[ng] = (uchar)gt;
        if (gt) ++ng;
        gt = 0;
        gstart = sub_group_broadcast(opk, first);
        r0 = l0; r1 = l1; r2 = l2;
      }
      lo = first;
    }
    if (on) {
      symb[q] = (uchar)sv;
      tokb[q] = (uchar)((min(L, 15u) << 4) | min(M - 4u, 15u));
    }
    const uint fl = on && L >= 15u, fm = on && M >= 19u;
    const uint xq = ei + sub_group_scan_exclusive_add(3u * fl + 2u * fm);
    if (fl) { const uint x = L - 15u; extb[xq] = (uchar)x; extb[xq + 1u] = (uchar)(x >> 8); extb[xq + 2u] = (uchar)(x >> 16); }
    if (fm) { const uint x = M - 19u, y = xq + 3u * fl; extb[y] = (uchar)x; extb[y + 1u] = (uchar)(x >> 8); }
    const uint nbb = on && sv < 250u ? sv : 0u;
    const uint bq = bp + sub_group_scan_exclusive_add(nbb);
    if (nbb) {
      const uint extra = D - (1u << nbb);
      const uint w = bq >> 5, sh = bq & 31u;
      atomic_or(&ow[w], extra << sh);
      if (sh + nbb > 32u) atomic_or(&ow[w + 1u], extra >> (32u - sh));
    }
    op += sub_group_reduce_add(L + M);
    ei += sub_group_reduce_add(3u * fl + 2u * fm);
    bp += sub_group_reduce_add(nbb);
  }
  if (gt) { if (k == 0) grpb[ng] = (uchar)gt; ++ng; }
  if (k == 0) {
    cnt[b * 8 + 2] = ei;
    cnt[b * 8 + 3] = (bp + 7u) / 8u;
    cnt[b * 8 + 5] = ng;
  }
}

// the stream s of block b: its symbols, count and lane scratch
inline __global const uchar* stream_sym(uint s, uint b, __global const uchar* lit, __global const uchar* tok,
                                        __global const uchar* ext, __global const uchar* sym, __global const uchar* grp) {
  if (s == 0) return lit + ((size_t)b << BLOG);
  if (s == 1) return tok + (size_t)b * TCAP;
  if (s == 2) return ext + (size_t)b * ECAP;
  if (s == 3) return sym + (size_t)b * TCAP;
  return grp + (size_t)b * TCAP;
}
inline uint stream_n(uint s, uint b, __global const uint* cnt) {
  if (s == 0) return cnt[b * 8 + 1];
  if (s == 2) return cnt[b * 8 + 2];
  if (s == 4) return cnt[b * 8 + 5];
  return cnt[b * 8 + 0];
}
inline uint lane_cap(uint s) { return s == 0 ? LC0 : s == 2 ? LC2 : LC1; }
inline uint lane_off(uint s) { return s == 0 ? 0u : s == 1 ? NL * LC0 : s == 2 ? NL * (LC0 + LC1) : s == 3 ? NL * (LC0 + LC1 + LC2) : NL * (LC0 + 2u * LC1 + LC2); }

// 9. rANS: histogram, 12-bit frequencies (the same steps as the CPU), 64 lanes each encoding its symbols in reverse
//    into its scratch region (backwards). sinfo per (block, stream): kind, body length, present symbols, symbols.
__kernel void k_rans_enc(__global const uchar* lit, __global const uchar* tok, __global const uchar* ext, __global const uchar* symst,
                         __global const uchar* grp, __global const uint* cnt, __global uchar* lanes, __global ushort* freqg, __global uint* lanelen,
                         __global uint* sinfo, uint nb) {
  const uint g = get_group_id(0);
  const uint b = g / 5u, s = g % 5u;
  const uint k = get_local_id(0);
  __global const uchar* sym = stream_sym(s, b, lit, tok, ext, symst, grp);
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

// 10. the block payload, as src/codec2.h lays it out; outlen[b] = its size, or 0xFFFFFFFF when not smaller than the block
__kernel void k_pack(__global const uchar* lit, __global const uchar* tok, __global const uchar* ext, __global const uchar* symst,
                     __global const uchar* grp,
                     __global const uchar* offb, __global const uint* cnt, __global const uchar* lanes, __global const ushort* freqg,
                     __global const uint* lanelen, __global const uint* sinfo, __global uchar* out, __global uint* outlen, uint nb,
                     uint nlast) {
  const uint b = get_group_id(0);
  const uint k = get_local_id(0);
  const uint n = block_len(b, nb, nlast);
  const uint nob = cnt[b * 8 + 3];
  __global uchar* o = out + (size_t)b * OUTCAP;
  uint off[6];
  off[0] = 20;
  for (uint s = 0; s < 5; ++s) off[s + 1] = off[s] + 5 + sinfo[(b * 5 + s) * 4 + 1];
  const uint total = off[5] + nob;
  if (total >= n) {
    if (k == 0) outlen[b] = 0xFFFFFFFFu;
    return;
  }
  if (k == 0) {
    put32g(o, cnt[b * 8 + 0]);
    put32g(o + 4, cnt[b * 8 + 1]);
    put32g(o + 8, cnt[b * 8 + 2]);
    put32g(o + 12, nob);
    put32g(o + 16, cnt[b * 8 + 5]);
    outlen[b] = total;
  }
  for (uint s = 0; s < 5; ++s) {
    const uint g = b * 5 + s;
    const uint kind = sinfo[g * 4 + 0], len = sinfo[g * 4 + 1], np = sinfo[g * 4 + 2], ns = sinfo[g * 4 + 3];
    __global uchar* d = o + off[s];
    if (k == 0) { d[0] = (uchar)kind; put32g(d + 1, len); }
    __global const uchar* sym = stream_sym(s, b, lit, tok, ext, symst, grp);
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
  for (uint i = k; i < nob; i += G) o[off[5] + i] = ob[i];
}

// ============================================================ decoder
//
// No memory access here can leave its block's buffers, whatever the data: every index is masked to the block, and every loop
// is bounded by a length that fits 16 bits. A bug or a damaged archive gives wrong bytes, which the CRC catches; it cannot
// fault the GPU, because a fault here would also take down other programs using it.
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#define SGW 16u
#define NSG (G / SGW)
#define SYM_REP 250u
#ifndef CHUNK
#define CHUNK 8u   // bytes a lane copies at a time: 4, 8 or 16
#endif
// error codes (err[b]); the host turns them into "block damaged"
#define E_HDR 1u
#define E_COUNTS 2u
#define E_STREAM 3u
#define E_TABLE 4u
#define E_LANE 5u
#define E_TOKEN 6u

// 1. the five symbol streams of a block (one work-group of 64 per stream): raw copied, rANS decoded by 64 lanes
__kernel void k_dec_streams(__global const uchar* comp, __global const uint* boff, __global const uint* binfo, __global uchar* lit,
                            __global uchar* tok, __global uchar* ext, __global uchar* sym, __global uchar* grp, __global uint* dcnt,
                            __global uint* err, uint nb) {
  const uint g = get_group_id(0);
  const uint b = g / 5u, s = g % 5u;
  const uint k = get_local_id(0);
  const uint GS = get_local_size(0);
  const uint raw = binfo[b * 4 + 0], clen = binfo[b * 4 + 1], method = binfo[b * 4 + 2];
  if (method == 0) return;
  __global const uchar* p = comp + boff[b];
  __local uint freq[256], cum[256], lanestart[NL + 1];
  __local uchar slot[SCALE];
  __local uint fail;
  if (k == 0) fail = 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (clen < 20) { if (k == 0) err[b] = E_HDR; return; }
  const uint nt = get32g(p), nl = get32g(p + 4), ne = get32g(p + 8), nob = get32g(p + 12), ng = get32g(p + 16);
  if (nt > TCAP || nl > raw || ne > ECAP || ng > nt || nob > B) { if (k == 0) err[b] = E_COUNTS; return; }
  uint pos = 20;
  for (uint t = 0; t < s; ++t) {
    if (pos + 5 > clen) { if (k == 0) err[b] = E_STREAM; return; }
    pos += 5 + min(get32g(p + pos + 1), clen);
  }
  if (pos + 5 > clen) { if (k == 0) err[b] = E_STREAM; return; }
  const uint kind = p[pos], len = get32g(p + pos + 1);
  if (len > clen || pos + 5 + len > clen) { if (k == 0) err[b] = E_STREAM; return; }
  __global const uchar* body = p + pos + 5;
  if (s == 4 && k == 0) {
    dcnt[b * 8 + 0] = nt; dcnt[b * 8 + 1] = nl; dcnt[b * 8 + 2] = ne; dcnt[b * 8 + 3] = nob; dcnt[b * 8 + 5] = ng;
    dcnt[b * 8 + 4] = pos + 5 + len;
    if (pos + 5 + len + nob > clen) err[b] = E_STREAM;
  }
  __global uchar* dest;
  uint ns;
  if (s == 0) { dest = lit + ((size_t)b << BLOG); ns = nl; }
  else if (s == 1) { dest = tok + (size_t)b * TCAP; ns = nt; }
  else if (s == 2) { dest = ext + (size_t)b * ECAP; ns = ne; }
  else if (s == 3) { dest = sym + (size_t)b * TCAP; ns = nt; }
  else { dest = grp + (size_t)b * TCAP; ns = ng; }
  if (kind == 0) {
    if (len != ns) { if (k == 0) err[b] = E_STREAM; return; }
    for (uint i = k; i < ns; i += GS) dest[i] = body[i];
    return;
  }
  if (kind != 1 || ns == 0 || len < 32 + 4 * NL) { if (k == 0) err[b] = E_STREAM; return; }
  __local uchar hdr[800];
  const uint hn = min(len, 800u);
  for (uint i = k; i < hn; i += GS) hdr[i] = body[i];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (k == 0) {
    uint idx = 0, c0 = 0;
    for (uint c = 0; c < 256; ++c) {
      uint f = 0;
      if ((hdr[c >> 3] >> (c & 7)) & 1) {
        if (32 + 2 * (idx + 1) + 4 * NL > min(len, 800u)) { fail = E_TABLE; break; }
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
    for (uint q = 0; q < freq[c]; ++q) slot[(cum[c] + q) & (SCALE - 1)] = (uchar)c;
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

// n bytes from s to d, exactly: a lane must not touch what the next lane writes
inline void copy_exact(__global uchar* d, __global const uchar* s, uint n) {
  uint i = 0;
#if CHUNK == 16
  for (; i + 16 <= n; i += 16) vstore16(vload16(0, s + i), 0, d + i);
  if (n & 8) { vstore8(vload8(0, s + i), 0, d + i); i += 8; }
#elif CHUNK == 8
  for (; i + 8 <= n; i += 8) vstore8(vload8(0, s + i), 0, d + i);
#endif
  if (n & 4) { vstore4(vload4(0, s + i), 0, d + i); i += 4; }
#if CHUNK == 4
  for (; i + 4 <= n; i += 4) vstore4(vload4(0, s + i), 0, d + i);
#endif
  if (n & 2) { vstore2(vload2(0, s + i), 0, d + i); i += 2; }
  if (n & 1) d[i] = s[i];
}

// 2. the LZ replay: one sub-group of 16 lanes per block, one token per lane, a global fence only between groups
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void k_dec_lz(__global const uchar* comp, __global const uint* boff, __global const uint* binfo, __global const uchar* lit,
                       __global const uchar* tok, __global const uchar* ext, __global const uchar* sym, __global const uchar* grp,
                       __global const uint* dcnt, __global uchar* out, __global uint* err, uint nb) {
  const uint sgid = get_sub_group_id();
  const uint b = get_group_id(0) * NSG + sgid;
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
  const uint nt = dcnt[b * 8 + 0], nl = dcnt[b * 8 + 1], ne = dcnt[b * 8 + 2], nob = dcnt[b * 8 + 3], ng = dcnt[b * 8 + 5];
  __global const uchar* offb = p + dcnt[b * 8 + 4];
  __global const uchar* litb = lit + ((size_t)b << BLOG);
  __global const uchar* tokb = tok + (size_t)b * TCAP;
  __global const uchar* extb = ext + (size_t)b * ECAP;
  __global const uchar* symb = sym + (size_t)b * TCAP;
  __global const uchar* grpb = grp + (size_t)b * TCAP;
  uint t = 0, op = 0, li = 0, ei = 0, bp = 0;
  uint r0 = 1, r1 = 4, r2 = 8, l0 = 1, l1 = 4, l2 = 8;
  uint bad = 0;
  for (uint g = 0; g < ng; ++g) {
    const uint gn = grpb[g];
    if (gn == 0 || t + gn > nt) { if (k == 0) err[b] = E_TOKEN; return; }
    const uint tend = t + gn;
    while (t < tend) {
      const uint c = min(SGW, tend - t);
      const uint on = k < c;
      const uint tv = on ? tokb[t + k] : 0u;
      const uint sv = on ? symb[t + k] : 0u;
      const uint fl = (tv >> 4) == 15u, fm = (tv & 15u) == 15u;
      const uint nx = 3u * fl + 2u * fm;
      const uint xq = ei + sub_group_scan_exclusive_add(nx);
      uint L = tv >> 4, M = on ? (tv & 15u) + 4u : 0u;
      if (fl) L += (uint)extb[xq & MASK] | ((uint)extb[(xq + 1u) & MASK] << 8) | ((uint)extb[(xq + 2u) & MASK] << 16);
      if (fm) { const uint q = xq + 3u * fl; M += (uint)extb[q & MASK] | ((uint)extb[(q + 1u) & MASK] << 8); }
      const uint nb_ = sv < SYM_REP ? min(sv, 19u) : 0u;
      const uint bq = bp + sub_group_scan_exclusive_add(nb_);
      uint D = 0;
      if (nb_) {
        const uint at = min(bq >> 3, nob);
        const uint w = get32g(offb + at);
        D = (1u << nb_) + ((w >> (bq & 7u)) & ((1u << nb_) - 1u));
      } else if (on && sv == 0u) {
        D = 1u;
      } else if (sv == SYM_REP) {
        D = r0;
      } else if (sv == SYM_REP + 1u) {
        D = r1;
      } else if (sv == SYM_REP + 2u) {
        D = r2;
      }
      const uint tot = L + M;
      const uint opk = op + sub_group_scan_exclusive_add(tot);
      const uint lik = li + sub_group_scan_exclusive_add(L);
      const uint ms = opk + L;
      if (on && ((uint)(opk + tot) > raw || lik + L > nl || D == 0u || D > ms)) { bad = 1; M = 0; }
      // literals come from their own stream, so they need no order
      if (L) copy_exact(o + (opk & MASK), litb + (lik & MASK), L);
      if (M) {
        const uint src = ms - D;
        const uint own = src >= opk;  // (the source is this token's literals, which are not in the output for certain yet)
        __global const uchar* sp = own ? litb + ((lik + (src - opk)) & MASK) : o + (src & MASK);
        if (D >= M) {
          copy_exact(o + (ms & MASK), sp, M);
        } else {  // a pattern D bytes long, repeated: read it once, write it round and round
          for (uint i = 0; i < M; ++i) o[(ms + i) & MASK] = sp[i % D];
        }
      }
      // the next chunk and the next group
      op += sub_group_reduce_add(tot);
      li += sub_group_reduce_add(L);
      ei += sub_group_reduce_add(nx);
      bp += sub_group_reduce_add(nb_);
      for (uint q = c > 3u ? c - 3u : 0u; q < c; ++q) {
        const uint v = sub_group_broadcast(D, q);
        l2 = l1; l1 = l0; l0 = v;
      }
      t += c;
    }
    r0 = l0; r1 = l1; r2 = l2;
    sub_group_barrier(CLK_GLOBAL_MEM_FENCE);
  }
  const uint rem = nl - min(li, nl);
  if (t != nt || op + rem != raw || sub_group_reduce_max(bad)) { if (k == 0) err[b] = E_TOKEN; return; }
  uint i = k * 8u;
  for (; i + 8u <= rem; i += SGW * 8u) vstore8(vload8(0, litb + ((li + i) & MASK)), 0, o + ((op + i) & MASK));
  for (i = (rem & ~7u) + k; i < rem; i += SGW) o[(op + i) & MASK] = litb[(li + i) & MASK];
}
