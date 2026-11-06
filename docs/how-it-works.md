# How zero works

zero splits its input into independent blocks, so a GPU can work on many at once. Each block is compressed with LZ77 and rANS.
The GPU kernels (`kernels/zero.cl`) and the CPU codec (`src/codec2.h`) read and write the same format, so an archive made by
either can be read by the other.

## The archive

A `.zero` file holds any number of files and folders. Their bytes form one solid stream, cut into blocks of one size (64 KiB to
1 MiB, 256 KiB by default). All numbers are little endian.

| Part | Contents |
|---|---|
| Header, 24 bytes | `ZERO`, version (2), log2 of the block size, 2 reserved bytes, total raw bytes (u64), number of blocks (u64) |
| Each block | raw length (u32), stored length (u32), method (u8: 0 stored, 1 ZB2), 3 reserved bytes, CRC-32 of the raw bytes (u32), payload |
| Entry table | u32 count, then per entry: type (u8: 1 folder), 3 reserved bytes, name length (u32), UTF-8 name with `/`, size (u64), modified time (u64), attributes (u32), CRC-32 of the file (u32) |
| Footer | table offset (u64), table length (u64), CRC-32 of the table (u32), `ZEND` |

A block that would not shrink is stored as it is, so incompressible data costs 16 bytes per block.

## A block: ZB2

ZB2 is built around how a GPU decodes. Tokens gather in **groups** that decode side by side, one token per lane:

- A match may read only output from before its group, or the literals of its own token. Those come from the literal stream, not
  from the output, so a run such as `AAAA` is fine there.
- Lengths past 15 have a fixed size, so a lane finds its own by counting flags, with no walking through the stream.
- An offset equal to one of the last three offsets of the group before costs no bits. Lanes need nothing from each other.

The streams:

| Stream | Contents |
|---|---|
| `lit` | the literal bytes |
| `tok` | one byte per token: literal run and match length, 4 bits each |
| `ext` | literal runs past 15 in 3 bytes, match lengths past 18 in 2 |
| `sym` | per token the bit length of its offset, or which recent offset it repeats |
| `groups` | tokens per group |
| offset bits | the rest of each offset, raw |

All but the last are entropy coded with rANS: 12-bit frequencies and 64 interleaved lanes, so 64 work-items decode a stream side
by side. A stream that would not shrink stays raw. `src/codec2.h` and `src/codec.h` document the exact bytes.

The parse knows nothing of groups, so segments of a block can be parsed in parallel. One pass afterwards makes the groups: a
match that reads its own group closes it. This costs almost nothing in size, because no match changes.

## Compressing on the GPU

Per batch of 16 MiB:

1. **Grouping positions.** `k_hash_count` counts each position's 5-byte hash per block, `k_bucket_scan` lays out the buckets, and
   `k_scatter` places each position in its bucket with 12 more bits of its hash as a tag.
2. **Sorting buckets.** `k_sort` sorts each bucket by tag, then position, with a sorting network in registers: one read and one
   write per bucket. A position's left neighbours are then the nearest earlier positions with its tag.
3. **Finding matches.** `k_match` walks that order and compares against a few left neighbours, up to a length set by the level.
   Neighbours with another tag cost no read of the data.
4. **Lazy matching.** `k_lazy` lets a match give way to a better one a byte later. It is a pure function of the previous step, so
   every walker in the next step decides alike.
5. **Parsing.** `k_parse` walks the block's segments in parallel, then one lane follows the true chain, keeping each walk from
   where the two meet. A match that reached the comparison limit is finished here, only where the chain uses it.
6. **Tokens and streams.** `k_tokens` collects literals and matches. `k_streams` makes the groups, the recent offsets and the
   streams, 16 tokens at a time.
7. **Entropy coding.** `k_rans_enc` builds a histogram and codes 64 lanes per stream.
8. **Packing.** `k_pack` lays out the payload.

## Decompressing on the GPU

`k_dec_streams` decodes the rANS lanes, one work-group of 64 per stream. `k_dec_lz` then replays the tokens with one 16-lane
sub-group per block, one token per lane, with a global memory fence only between groups. The compressed input and the output are
in host memory, because an integrated GPU's RAM is the CPU's.

While the GPU decodes a batch, a thread reads the next one, and another checks and writes the one before.

## When the GPU fails

A GPU can fail in the middle of a job: a lost device, from a fault here or in another program using the same GPU, or a driver
reset. `src/safe_codec.h` then does that batch and the rest on the CPU. Both codecs share one format, so the job finishes and the
archive is right. A lost device stays lost for the process, so there is no going back to the GPU until the program starts again.

The kernels are also written not to fault: every index in the decoder is masked to its block, and every loop is bounded by a
length that fits 16 bits. A damaged archive gives wrong bytes, which the CRC catches. `tests/stress.py` checks this: 732 runs, 300
of them on damaged archives, with no failure and no GPU reset.

Run one thing at a time on the GPU. A lost device in one process took down another process's GPU work in testing, so
`tools/gpu_gate.ps1` checks the GPU is free before test runs.

## The runtime

`src/gpu.h` and `src/ze_min.h` are a few hundred lines on Level Zero with no SDK. The loader library is opened at run time. The
driver's OpenCL runtime compiles the kernels to the device's native binary, which is cached on disk and loaded as a Level Zero
module. OpenCL is loaded only when something must be compiled, which took opening the GPU from 0.67 s to 0.09 s. This is the
approach of [arcinferzero](https://github.com/rizagp/arcinferzero).

## What was tried

Every step was measured with alternated runs, because single runs of one binary moved by up to 2x. `tools/lab/` has the tools.

**What helped:**

| Change | Effect |
|---|---|
| Tokens decoding side by side, in groups | decoding 2.0x to 2.3x faster, with the same or a smaller archive |
| Comparing only a few bytes, and finishing long matches in the parse | `k_match` 37 to 20 ms per 16 MiB, with the same matches |
| Sorted buckets, with fewer candidates | `k_match` 16 to 9 ms, and a smaller archive |
| Recent offsets and lazy matching | archives 5 to 10 % smaller on binaries |
| Lanes taking consecutive tokens, not scattered ones | `k_tokens` and `k_streams` 28 to 9 ms |
| One token stage per sub-group, with no fences inside a group | the fence stopped being a cost: removing it entirely changed nothing |
| Reading, checking and writing while the GPU works | 6.46 GiB checked in 4.0 s, from 7.0 s |
| Parallel checksums | 0.16 s of a 0.5 s extraction gone |
| Host memory for the decoder's input and output | copies gone, kernels no slower |
| Smaller blocks | decoder threads are the main lever: 1 MiB to 512 KiB nearly doubled LZ speed |

**What was measured and dropped:**

| Idea | Result |
|---|---|
| The GPU and the CPU sharing a batch | see [the benchmarks](benchmarks.md#why-not-the-cpu-and-the-gpu-together) |
| Coding the top or low bits of offsets | under 1 %: they are close to random |
| Literal and match lengths in separate streams | larger |
| Skipping rANS on tokens or symbols | 2 to 17 % larger |
| A fixed fence every G bytes | costs more than groups for less |
| Cutting matches short to make groups larger | 4 to 7 % larger for up to twice the group size, and 4 % more speed at most |
| Scatter launched in windows of positions | slower than sorting, and no better |
| Scatter in position order, one sub-group per block | good ratio, but 30 to 50 ms against 8 |
| Odd-even passes over the buckets | less per ms than sorting them |
| A 6-byte hash | binaries 1.3 points larger |
| More or fewer buckets | no gain worth its cost |
| The 256-register kernel build | no kernel spills (`ZERO_KPROPS=1` shows it) |
| The NPU | not tried: it runs matrix graphs, and LZ and rANS are branches and bytes |

## Finding your way around

| Path | Contents |
|---|---|
| `src/codec.h` | the entropy coder |
| `src/codec2.h` | the ZB2 format and the CPU encoder and decoder |
| `src/archive.h` | the `.zero` container: walking folders, compressing, listing, extracting, testing, the CPU codec |
| `src/gpu.h`, `src/ze_min.h`, `src/platform.h` | the Level Zero runtime and the Windows / Linux shim |
| `src/gpu_codec.h` | the GPU codec: buffers, kernel launches, batches |
| `src/safe_codec.h` | the CPU taking over when the GPU fails |
| `kernels/zero.cl` | the kernels, embedded into the exe at build time |
| `src/main.cpp` | the command line |
| `src/gui_win.h` | the Windows window, the installer and the Explorer entries |
| `src/gui_gtk.h` | the Linux window |
| `tools/lab/` | the format experiments and decoder measurements |

Two switches help with work on the kernels: `ZERO_PROFILE=1` prints the time of each stage per batch, and `ZERO_KPROPS=1` prints
each kernel's register spill and memory use.
