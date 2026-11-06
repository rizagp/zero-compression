"""The whole codec, in memory, on each corpus: GPU levels and the CPU, alternated, with the time of each GPU stage. Decoding is
reported cold (the first pass) and warm (the median of the others). Stops if the GPU gets busy.
usage: bench_levels.py <folder of *.bin> <rounds> [block] [extra zero bench args]"""
import collections, glob, os, re, statistics, subprocess, sys, time
folder, rounds = sys.argv[1], int(sys.argv[2])
block = sys.argv[3] if len(sys.argv) > 3 else '18'
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
zero, gate = os.path.join(root, 'bin', 'zero.exe'), os.path.join(root, 'tools', 'gpu_gate.ps1')
variants = [('gpu L1', ['--level', '1']), ('gpu L5', []), ('gpu L9', ['--level', '9']), ('cpu L5', ['--cpu'])]


def free():
    r = subprocess.run(['pwsh', '-NoProfile', '-File', gate, '-Stable', '20', '-Wait', '3000'], capture_output=True, text=True)
    if r.returncode:
        print('STOPPED:', r.stdout.strip(), flush=True)
    return r.returncode == 0


for path in sorted(glob.glob(os.path.join(folder, '*.bin'))):
    size = os.path.getsize(path)
    print('==', os.path.basename(path), '%d MiB' % (size >> 20), 'block', block, flush=True)
    res = collections.defaultdict(list)
    stages = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in range(rounds):
        for name, args in variants:
            if not free():
                sys.exit(4)
            e = dict(os.environ, ZERO_PROFILE='1', ZERO_BENCH_REPEAT='4')
            t = subprocess.run([zero, 'bench', path, '--block', block] + args, env=e, capture_output=True, text=True, errors='replace')
            out = t.stdout + t.stderr
            if t.returncode or 'MISMATCH' in out:
                raise SystemExit('failed ' + name + ':\n' + out[-500:])
            m = re.search(r'compress:\s+([\d.]+) s, .*\(([\d.]+) %\)', out)
            dec = [float(x) for x in re.findall(r'decompress:\s+([\d.]+) s, \d+ MB/s with ' + ('CPU' if 'cpu' in name else 'GPU'), out)]
            res[name].append((float(m.group(1)), float(m.group(2)), dec[0], statistics.median(dec[1:]) if len(dec) > 1 else dec[0], 'GPU stopped' in out))
            for line in out.splitlines():
                q = re.match(r'\[profile\] compress ([\d.]+) MB \(ms\):(.*)', line)
                if q and float(q.group(1)) > 15:
                    for s, v in re.findall(r'([a-z]+) ([\d.]+)', q.group(2)):
                        stages[name][s].append(float(v))
            time.sleep(2)
    for name, _ in variants:
        x = res[name]
        med = lambda i: statistics.median(y[i] for y in x)
        print('  %-7s compress %5.0f MB/s  %6.2f %%  decompress cold %5.0f warm %5.0f MB/s%s' % (name, size / med(0) / 1e6, x[0][1], size / med(2) / 1e6, size / med(3) / 1e6, '  GPU STOPPED' if any(y[4] for y in x) else ''), flush=True)
    if stages['gpu L5']:
        print('  L5 ms per 16 MiB: ' + ' '.join('%s %.1f' % (k, statistics.median(v)) for k, v in stages['gpu L5'].items()), flush=True)
