# zero

**Lossless compression of files and folders on an Intel GPU.** A from-scratch block codec (LZ77 + rANS) whose
encoder and decoder run as OpenCL C kernels through **Level Zero** on Intel Arc / Iris graphics, with a CPU
reference implementation of the same format, a `.zero` archive container for many files and folders, a command
line, and a window (Win32 on Windows, GTK 3 on Linux). One C++20 code base, no dependencies beyond the GPU driver.

Built and measured on a Core Ultra 5 125H laptop (Arc iGPU, Xe-LPG, 112 EUs, no discrete GPU), Windows 11.

## What it does

- `zero.exe` / `zero`: the command line. `c` compresses files and folders into one archive, `x` extracts, `t` tests
  every checksum without writing, `l` lists, `bench` measures one file, `info` shows the GPU.
- `zero-gui.exe` / `zero-gui`: a window. Drop files and folders on it and press Compress; drop a `.zero` archive and
  press Extract. The archive's name and place are suggested, and the block size and processor stay under Advanced. On
  Windows it follows Windows 11: Mica, the system's light or dark theme and accent color, painted with Direct2D with no
  framework (it is not read by screen readers yet). The Linux window is still the plain GTK one.
- In Explorer, after "Add zero to the Explorer menu" on the window's start screen (or `zero-gui --install-shell`):
  a double-click opens a `.zero` archive, and the right-click menu has Compress with zero and Extract here. This is for
  the current user only and needs no administrator; the same button, or `--uninstall-shell`, removes it. Windows 11
  puts these commands under "Show more options": its first menu takes only signed packages. If the exe moves, add it again.
- A `.zero` archive holds any number of files and folders. Their bytes form one solid stream cut into blocks
  (64 KiB .. 1 MiB, default 512 KiB); every block compresses and decompresses on its own, which is what the GPU
  parallelizes. Every block and every file carries a CRC-32; extraction and `t` verify them all.
- Without a usable GPU (no Intel driver, another vendor's GPU) everything runs on the CPU codec, which reads and
  writes the same format. `--cpu` forces it.

## Measured (2026-11-06)

186.7 MiB of mixed binaries (`.exe` and `.obj` files), 512 KiB blocks, warm kernels, same laptop, each number the
median of runs alternated with idle gaps:

| | GPU (Arc iGPU) | CPU (18 threads, reference) | zstd -1, one thread | zstd -3, all threads |
|---|---|---|---|---|
| compress, level 5 (default) | 0.67 s, 290 MB/s, 23.7 % | 0.7 s, 275 MB/s, 21.5 % (486 MB/s on a cold machine) | 0.30 s, 643 MB/s, 21.2 % | 0.13 s, 1.5 GB/s, 17.7 % |
| compress, level 1 / 9 | 344 MB/s, 25.0 % / 161 MB/s, 22.8 % | | | |
| decompress | 0.14 s, 1.4 GB/s | 0.09 s, 2.2 GB/s | 0.17 s, 1.2 GB/s (one thread) | 0.18 s, 1.1 GB/s (one thread) |

(zstd 1.5 through the `zstandard` Python package, the same file in memory.) Text (Markdown, 551 KiB): GPU 39.1 %,
CPU 37.5 %. Random data is stored as is. A 2 MiB run of zeros: 9.4 KiB. Every archive made by one codec was
extracted bit for bit by the other (tests/roundtrip.ps1: 250 files, four codec combinations), and an archive
made on Windows extracted identically on Linux.

The whole program on the same file, process start to exit (about 0.2 s of it is opening the GPU): `zero c` 0.94 s,
`zero x` 0.61 s, against 1.31 s and 0.74 s before host work was moved out of the GPU's way. 250 files, 22.4 MiB:
0.36 s and 0.64 s, against 0.40 s and 0.83 s.

Data that hardly compresses is where the GPU wins. A quantized model and its mmproj (GGUF, 12.15 GiB, 1 MiB blocks,
97.5 %): GPU 80 s, CPU 631 s.

So, honestly: on this integrated GPU the codec is correct and complete and its ratio is near zstd -1. On ordinary
files the GPU is about level with the CPU cores of the same chip when compressing, slower when decompressing, and not
faster than zstd. Where the GPU time goes (ZERO_PROFILE=1
prints it per batch): the match finder (about a third of compression) and the LZ replay (about 70 % of
decompression). The GPU path leaves the CPU free meanwhile.

The first version did 175 MB/s and 876 MB/s. What the optimization passes found, on this GPU:
- A long match was compared from every one of its positions: stopping at a few bytes and letting the parse finish
  only the positions the chain really visits took the match finder from 37 to 20 ms per 16 MiB with the same matches
  (`LENCAP`, by far the largest gain, found by capping lengths as an ablation).
- A candidate's cost is the compare, not finding it: with no candidates at all the kernel takes 1 ms of 37. Fewer and
  better candidates are what the levels set (1: 4+2 neighbours, compare 8 bytes; 5: 8+2, 16; 9: 16+4, 32).
- Walking srt instead of the positions removes the bucket tables and the 64 MiB index from the kernel; a 12-bit tag
  beside each position rejects most neighbours without reading the data. Requesting all candidates' words before
  looking at any gave 6 %.
- The order inside a bucket decides the ratio, because the neighbours should be the nearest earlier positions: a
  16-lane scatter reading bytes gives 23.7 %, a 32-lane one reading words 24.4 %, for 2 ms per 16 MiB. Odd-even
  passes to sort the buckets bought less per ms than more candidates (2 passes 3.6 ms, 0.2 points).
- The LZ decoder's cost per token was the global memory fence between tokens, not the token-stream reads (staging
  them in local memory changed nothing). A fence is now taken only when a match reads bytes written since the
  last one; distant matches and literals need none. One 16-lane sub-group per block instead of a 256-item
  work-group had earlier taken 529 -> 876 MB/s.
- The decoder reads and writes host memory directly (an iGPU's RAM is the CPU's): the kernels got no slower and two
  copies disappeared. For the encoder's input and output the same change gained nothing.
- The GPU was waiting for one CPU thread's checksums: 0.16 s of a 0.5 s extraction. They now run on all threads
  (a file's CRC by pieces, combined), and `zero c` reads and checksums the next batch while the GPU works.
- The GPU and the CPU sharing a batch's blocks were dropped: on mixed files 390 MB/s and 22.6 %, but on the 12 GiB model
  128 s against the GPU's 80 s, and still 96 s when batches the CPU slowed down were left to the GPU alone: the GPU
  waits for the CPU's unfinished blocks. Decompressing together was slower than either alone. The NPU was not tried:
  it runs matrix graphs, and LZ and rANS are branches and bytes.
- No gain, measured: copying 4 bytes a lane from shorter tokens than 64 bytes, the 256-register build (no kernel
  spills: ZERO_KPROPS=1 prints it), a 64 MiB encoder batch (slower end to end and 1.3 GB).
- Smaller blocks mean more hardware threads in the decoder: `--block 18` decompresses at 1.53 GB/s for 24.2 %,
  `--block 17` at about 1.5-1.65 GB/s for 25.5 %. The default stays 512 KiB for its ratio.
- Compare alternated runs only: single runs of the same binary moved by 2x here (other programs' memory traffic,
  Windows), and a wall time 10x the codec's was the host and the file system, not the GPU.

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
   a scan (`k_bucket_scan`) and the positions scattered into them with a 12-bit tag (`k_scatter`); `k_match` walks
   that order and takes the longest match among the neighbours with the same tag (8 before and 2 after by default,
   the nearest on a tie), plus the byte before (runs), comparing up to 16 bytes; offsets anywhere in the block.
2. **Greedy parse in parallel** (`k_parse`): each work-item walks its segment of the block from the segment start;
   one lane then follows the true chain through the segments, keeping each tentative walk from where the two
   coincide and undoing only its prefix. Greedy parses re-synchronize within a few tokens on ordinary data. A match
   that reached the comparison limit is finished here, up to 274 bytes, only where the chain uses it.
3. **Tokens and streams** (`k_tokens`, `k_streams`): literals, one token byte per match (LZ4-style nibbles with
   extension bytes), offset bit lengths as symbols and the extra bits packed with atomics.
4. **rANS** (`k_rans_enc`): per stream a histogram, 12-bit frequencies (the same integer steps as the CPU), 64
   interleaved lanes each encoding its symbols in reverse; a stream that would not shrink stays raw.
5. **Pack** (`k_pack`): the payload laid out as `codec.h` describes; a block that would not shrink is stored.

Decoder: `k_dec_streams` (one work-group of 64 per stream, the rANS lanes) and `k_dec_lz` (one 16-lane sub-group
per block walking the tokens, every lane copying a share of each token, 4 bytes a lane for long ones; an
overlapping match repeats the pattern before it, so a token's bytes are independent of each other; a memory fence
only before a match that reads bytes written since the last fence). The compressed input and the output are in
host memory.

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
| `tests/gui_block_switch.ps1` | drives the window through several archives with different block sizes in one process, compare hashes |
| `tests/explorer_menu.ps1` | the Explorer commands, opening an archive, several files into one window, removing the entries |
| `tools/zero.ico`, `tools/zero.rc`, `tools/make_icon.py` | the icon of the window and of `.zero` files |

## Roadmap

- Decoder: its parallelism is the number of blocks in a batch (one sub-group each); a file of 100 MB keeps only
  200 of the GPU's ~900 hardware threads busy. Smaller blocks for small files, or two sub-groups sharing a block's
  literal copies, would help. The rANS stage (15 % of decompression) decodes 64 lanes per stream with a serial
  table parse per work-group.
- Encoder: a truly position-ordered bucket would give the same ratio with fewer candidates; a longer hash (5-6 bytes)
  would make them better. Lazy matching for a better ratio.
- Format: Huffman-free offset coding is already compact; an order-1 context for literals would help text.
- Archive: extract single entries (the table already gives each file's place in the stream), symbolic links,
  Windows attributes.
- Linux: run the GTK window; the Level Zero path on Linux has only been compiled, not run.
- Overlap writing with GPU work too: `zero c` now reads ahead, but each batch is still written in turn, and `zero x`
  writes its files between batches. Opening the GPU costs about 0.2 s per process.

## License

Apache License 2.0.
