# CPU against GPU

All numbers are from one laptop: Core Ultra 5 125H, Arc iGPU (Xe-LPG, 112 EUs, no discrete GPU), 32 GB of shared RAM, an NVMe
SSD, Windows 11, on mains power. Block size 256 KiB, level 5. Measured on 2026-11-07. Treat them as a guide for machines like
this one: a discrete Arc card or another CPU would give different results.

## The short version

- **On large ordinary files the two take about the same time, but the GPU leaves the CPU free.** Compressing 6.46 GiB used 35 s
  of CPU time with the GPU and 323 s with the CPU codec.
- **On ordinary files the CPU makes archives 5 to 15 % smaller,** because its search finds better matches. On a quantized model
  the two come out the same.
- **On small files the CPU is faster,** because the GPU has to be opened and its memory set up first.
- **On data that hardly compresses the GPU compresses faster:** 35 s against 57 s on an 11.29 GiB model.

So use the GPU, the default, when the machine has other work to do or the data hardly compresses. Use `--cpu` for the smallest
archives, and for small files when you want the fastest result.

![GPU against CPU on 6.46 GiB of DLLs](images/cpu-gpu.svg)

## The whole program

Process start to exit, read from and written to the SSD, median of two or three runs alternated with idle gaps.

| Data | | Compress | Size | CPU time to compress | Check archive | Extract |
|---|---|---|---|---|---|---|
| Text, 0.34 GiB | GPU | 1.2 s | 11.0 % | 2 s | 0.5 s | 0.8 s |
| | CPU | 0.6 s | 9.4 % | 6 s | 0.2 s | 0.5 s |
| Binaries, 1 GiB | GPU | 4.2 s | 40.0 % | 5 s | 1.4 s | 2.1 s |
| | CPU | 3.4 s | 37.9 % | 47 s | 0.7 s | 1.6 s |
| Binaries, 6.46 GiB | GPU | 24.6 s | 43.4 % | 35 s | 4.0 s | 35.9 s |
| | CPU | 22.6 s | 41.5 % | 323 s | 3.9 s | 19.0 s |
| Random, 2 GiB | GPU | 5.5 s | 100 % | 9 s | 2.3 s | 3.2 s |
| | CPU | 8.5 s | 100 % | 117 s | 0.9 s | 3.2 s |

Every extracted file was compared by SHA-256, and every archive was also read by the other codec.

**Extract is the least certain column.** It writes to disk, and repeated runs of one binary differed by up to 3x here, which
is file cache and antivirus as much as the codec. Check archive does the same decoding and writes nothing.

**Memory.** Compressing, the program itself peaks near 570 MiB with the GPU and 270 MiB with the CPU, not counting the GPU's
own buffers. Decoding with the GPU holds up to about 4 GiB, because it decodes up to 512 MiB at a time.

### A large model file

A quantized model hardly compresses at all. These are the two files of one model, read from and written to the SSD:

| File | | Compress | Size | CPU time to compress | Check archive | Extract |
|---|---|---|---|---|---|---|
| Quantized weights, 11.29 GiB | GPU | 35.4 s | 99.2 % | 56 s | 7.2 s | 111.5 s |
| | CPU | 56.6 s | 99.2 % | 754 s | 5.4 s | 82.4 s |
| Multimodal projector (bfloat16), 0.87 GiB | GPU | 2.8 s | 78.2 % | 4 s | 1.3 s | 1.8 s |
| | CPU | 3.8 s | 78.7 % | 52 s | 0.9 s | 1.3 s |

Together, 12.15 GiB compress in 38 s on the GPU and 60 s on the CPU, and the CPU spends about 60 s against 805 s of processor time.

The extracted files matched by SHA-256, and each archive was also read by the other codec. The weights ran once, GPU first, so
that row is a single sample, and the GPU may have found less of the file in the cache. The projector is the median of three
rounds. Extracting the weights is slow because of what lies outside the codec, whose own share was about 6.5 s on either.

On the previous format (ZB1), with 1 MiB blocks and reading from a USB disk, the same 12.15 GiB took 80 s on the GPU and 631 s on
the CPU. Disk, format and block size all changed at once, so that gap of 8x cannot be split between them, and it should not be
quoted for the current format.

To measure only models, keeping the other results: `python tests/compare_cpu_gpu.py --work <folder> --model <file.gguf> --models-only`.

## The GPU codec by itself

Kernels only, no disk, 512 MiB per data set (text 343 MiB). This is what the GPU can do when nothing else holds it back, and it
is why the two formats below were compared this way. Decoding after the first pass, with the same blocks:

| Decode | ZB1 (before) | ZB2 (now) | |
|---|---|---|---|
| Binaries | 2,012 MB/s | 4,116 MB/s | 2.0x |
| Text | 4,108 MB/s | 9,358 MB/s | 2.3x |
| Libraries | 2,426 MB/s | 5,220 MB/s | 2.2x |

In these three rows ZB1 uses 512 KiB blocks and tokens from the GPU encoder, and ZB2 uses 256 KiB blocks and tokens from the CPU
encoder, chosen because the two sizes come out about equal. A first pass is slower, about 1.3x to 1.4x better than ZB1, because
memory is touched for the first time.

Compression by level, in memory, the same data:

| Level 1 / 5 / 9 | GPU | Size |
|---|---|---|
| Binaries | 324 / 287 / 181 MB/s | 44.0 / 43.0 / 42.3 % |
| Libraries | 336 / 286 / 161 MB/s | 26.5 / 25.0 / 23.9 % |
| Text | 384 / 302 / 157 MB/s | 12.4 / 11.0 / 10.1 % |

Those were taken before the 5-byte hash and sorted buckets, which moved level 5 to about 283, 298 and 306 MB/s at 42.7, 25.6 and
10.1 % on 128 MiB samples of the same data.

### Why not the CPU and the GPU together

It was built and measured, on the previous format. On mixed files it compressed at 390 MB/s against about 280 for either alone,
but on a 12 GiB model read from a USB disk it took 128 s against the GPU's 80 s, because the GPU waited for blocks the CPU had not finished. Leaving the CPU out of batches it
slowed still took 96 s. Decoding together was slower than either alone. It was dropped.

## How to measure

```bash
python tests/compare_cpu_gpu.py --work <folder>
```

```bash
python tools/make_charts.py
```

The first writes `docs/data/benchmarks.json` and removes its test data at the end. The second redraws the chart from it.

Numbers on this machine move a lot, so:

- Compare only runs that alternate. Single runs of one binary differed by up to 2x.
- Leave idle gaps, and nothing else on the GPU. `tools/gpu_gate.ps1` checks that before every GPU run.
- Other programs' memory traffic slows the GPU, because an iGPU shares the RAM.
- Separate first passes from later ones.
