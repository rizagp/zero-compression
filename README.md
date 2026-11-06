# zero

**Lossless compression of files and folders on an Intel GPU.** A from-scratch block codec (LZ77 + rANS) whose
encoder and decoder run as OpenCL C kernels through **Level Zero** on Intel Arc / Iris graphics, with a CPU
reference implementation of the same format, a `.zero` archive container for many files and folders, a command
line, and a window (Win32 on Windows, GTK 3 on Linux). One C++20 code base, no dependencies beyond the GPU driver.

Built and measured on a Core Ultra 5 125H laptop (Arc iGPU, Xe-LPG, 112 EUs, no discrete GPU), Windows 11.

## What it does

- `zero.exe` / `zero`: the command line. `c` compresses files and folders into one archive, `x` extracts, `t` tests
  every checksum without writing, `l` lists, `bench` measures one file, `info` shows the GPU.
- `zero-gui.exe` / `zero-gui`: a window. Drop files and folders on it (or Add files... / Add folder...), choose the
  archive path, Compress. Drop a `.zero` archive to Extract... or Test... it. Progress, a log, Cancel.
- A `.zero` archive holds any number of files and folders. Their bytes form one solid stream cut into blocks
  (64 KiB .. 1 MiB, default 512 KiB); every block compresses and decompresses on its own, which is what the GPU
  parallelizes. Every block and every file carries a CRC-32; extraction and `t` verify them all.
- Without a usable GPU (no Intel driver, another vendor's GPU) everything runs on the CPU codec, which reads and
  writes the same format. `--cpu` forces it.

## Measured (2026-11-06)

186.7 MiB of mixed binaries (`.exe` and `.obj` files), 512 KiB blocks, warm kernels, same laptop:

| | GPU (Arc iGPU) | CPU (18 threads, reference) | zstd -1, one thread | zstd -3, all threads |
|---|---|---|---|---|
| compress, level 5 (default) | 0.85 s, 231 MB/s, 24.1 % | 0.40 s, 486 MB/s, 21.5 % | 0.30 s, 643 MB/s, 21.2 % | 0.13 s, 1.5 GB/s, 17.7 % |
| compress, level 1 / 9 | 266 MB/s, 25.3 % / 198 MB/s, 23.5 % | | | |
| decompress | 0.14 s, 1.35 GB/s | 0.07-0.13 s, 1.5-2.8 GB/s | 0.17 s, 1.2 GB/s (one thread) | 0.18 s, 1.1 GB/s (one thread) |

(zstd 1.5 through the `zstandard` Python package, the same file in memory.) Text (Markdown, 551 KiB): GPU 39.3 %,
CPU 37.5 %. Random data is stored as is. A 2 MiB run of zeros: 6.6 KiB. Every archive made by one codec was
extracted bit for bit by the other (tests/roundtrip.ps1: 250 files, four codec combinations), and an archive
made on Windows extracted identically on Linux.

So, honestly: on this integrated GPU the codec is correct and complete, its ratio is near zstd -1, and the GPU is
**not yet faster** than the CPU cores of the same chip, nor than zstd. Where the GPU time goes (ZERO_PROFILE=1
prints it per batch): the match finder (about half of compression) and the LZ replay (about 70 % of
decompression). The GPU path leaves the CPU free meanwhile.

The first version did 175 MB/s and 876 MB/s. What the optimization pass found, on this GPU:
- The match finder is bound by random memory accesses per candidate, not by byte compares: reading 4 bytes a
  step through aligned words changed nothing (and unaligned `uchar4` loads were 2x slower). Fewer candidates and
  stopping at a "good enough" length did: that is what the levels set (1: 4+2 neighbours, stop at 32 bytes;
  5: 8+2, 64; 9: 8+8, never).
- The LZ decoder's cost per token was the global memory fence between tokens, not the token-stream reads (staging
  them in local memory changed nothing). A fence is now taken only when a match reads bytes written since the
  last one; distant matches and literals need none. One 16-lane sub-group per block instead of a 256-item
  work-group had earlier taken 529 -> 876 MB/s.
- The decoder writes straight into host memory (an iGPU's RAM is the CPU's): the LZ kernel got faster, not
  slower, and the copy back disappeared.
- A 64 MiB encoder batch (more blocks in flight) was slower end to end than 16 MiB and took 1.3 GB.

## Build

Windows: Visual Studio Build Tools with "Desktop development with C++" (the script finds the compiler), then

```
.\build.ps1
```

gives `bin\zero.exe` and `bin\zero-gui.exe` (statically linked, UTF-8 paths, no runtime to install).

Linux: g++ 13+ or clang++ 17+, then `./build.sh` gives `bin/zero` and `bin/zero-gui`. The GUI loads GTK 3 at run
time (`libgtk-3.so.0`), so no GTK development packages are needed to build. The GPU needs Intel's compute runtime
(`libze_loader.so.1`, `libOpenCL.so.1`, the `intel-opencl-icd` / `intel-level-zero-gpu` packages). The Linux build
was compiled and its CLI run in a container; the GTK window has not yet been opened on a Linux desktop.

## Use

```
zero c photos docs\report.docx -o backup.zero        # files and folders into one archive
zero c project -o project.zero --block 20            # 1 MiB blocks (a little smaller, less parallel)
zero l backup.zero                                   # list
zero t backup.zero                                   # verify every checksum
zero x backup.zero -o restored                       # extract into a folder
zero x backup.zero --cpu                             # with the CPU codec
zero bench big.bin                                   # compress in memory, decompress with both codecs, compare
zero info                                            # the GPU, its driver, the kernel cache
```

The compiled kernels are cached in `%LOCALAPPDATA%\zero\kernels` (Linux: `~/.cache/zero/kernels`), keyed by
source, options, GPU and driver; the first run on a GPU takes about two seconds longer.

## How it works

`src/codec.h` defines the block format ("ZB1") and holds the CPU reference encoder and decoder; `kernels/zero.cl`
produces and consumes the same bytes on the GPU.

Encoder, per batch of blocks (16 MiB):
1. **Match finding**: every position's 4-byte hash is counted per block (`k_hash_count`), the buckets are laid out by
   a scan (`k_bucket_scan`) and the positions scattered into them (`k_scatter`), so each position sees its whole
   bucket; `k_match` takes the longest match among 8 bucket neighbours on each side (the nearest on a tie), plus
   the byte before (runs), matches of 4..274 bytes, offsets anywhere in the block.
2. **Greedy parse in parallel** (`k_parse`): each work-item walks its segment of the block from the segment start;
   one lane then follows the true chain through the segments, keeping each tentative walk from where the two
   coincide and undoing only its prefix. Greedy parses re-synchronize within a few tokens on ordinary data.
3. **Tokens and streams** (`k_tokens`, `k_streams`): literals, one token byte per match (LZ4-style nibbles with
   extension bytes), offset bit lengths as symbols and the extra bits packed with atomics.
4. **rANS** (`k_rans_enc`): per stream a histogram, 12-bit frequencies (the same integer steps as the CPU), 64
   interleaved lanes each encoding its symbols in reverse; a stream that would not shrink stays raw.
5. **Pack** (`k_pack`): the payload laid out as `codec.h` describes; a block that would not shrink is stored.

Decoder: `k_dec_streams` (one work-group of 64 per stream, the rANS lanes) and `k_dec_lz` (one 16-lane sub-group
per block walking the tokens, every lane copying a share of each token, 4 bytes a lane for long ones; an
overlapping match repeats the pattern before it, so a token's bytes are independent of each other; a memory fence
only before a match that reads bytes written since the last fence). The output goes straight to host memory.

The runtime (`src/gpu.h`, `src/ze_min.h`) is a few hundred lines on Level Zero with no SDK: the loader library is
opened at run time, kernels are compiled by the driver's OpenCL runtime into the device's native binary and loaded
as a Level Zero module (the approach of [arcinferzero](https://github.com/rizagp/arcinferzero)).

## Layout

| | |
|---|---|
| `src/codec.h` | the ZB1 block format, shared constants, CPU reference encoder / decoder |
| `src/archive.h` | the `.zero` container, walking folders, compressing / listing / extracting / testing, the CPU codec |
| `src/gpu.h`, `src/ze_min.h`, `src/platform.h` | Level Zero runtime, loader, Windows / Linux shim |
| `src/gpu_codec.h` | the GPU codec: buffers, kernel launches, batches |
| `kernels/zero.cl` | the kernels (embedded into the exe at build time) |
| `src/main.cpp` | the command line; `src/gui_win.h`, `src/gui_gtk.h` the windows |
| `tests/roundtrip.ps1` | compress / test / extract a folder with both codecs in all four combinations, compare hashes |

## Roadmap

- Decoder: its parallelism is the number of blocks in a batch (one sub-group each); a file of 100 MB keeps only
  200 of the GPU's ~900 hardware threads busy. Smaller blocks for small files, or two sub-groups sharing a block's
  literal copies, would help. The rANS stage (15 % of decompression) decodes 64 lanes per stream with a serial
  table parse per work-group.
- Encoder: the match finder is bound by random reads of the candidates' bytes; a longer hash (5-6 bytes) would
  make the candidates better and fewer. Lazy matching for a better ratio.
- Format: Huffman-free offset coding is already compact; an order-1 context for literals would help text.
- Archive: extract single entries (the table already gives each file's place in the stream), symbolic links,
  Windows attributes.
- Linux: run the GTK window; the Level Zero path on Linux has only been compiled, not run.
- Overlap file reading with GPU work (today each batch is read, compressed, written in turn).

## License

Apache License 2.0.
