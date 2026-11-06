"""The ZB2 decoder against the ZB1 one on the same data, alternated, stages only (no disk). Stops if the GPU gets busy.
usage: compare_decoders.py <folder with zdec.exe and c\\*.bin> <rounds> name=<zdec args> ..."""
import os, re, shlex, statistics, subprocess, sys, time
lab, rounds = sys.argv[1], int(sys.argv[2])
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
zero, zdec, cl = os.path.join(root, 'bin', 'zero.exe'), os.path.join(lab, 'zdec.exe'), os.path.join(root, 'kernels', 'zero.cl')
gate = os.path.join(root, 'tools', 'gpu_gate.ps1')
variants = [a.split('=', 1) for a in sys.argv[3:]]
corpora = ['dll', 'text', 'lib']


def free():
    r = subprocess.run(['pwsh', '-NoProfile', '-File', gate, '-Stable', '20', '-Wait', '3000'], capture_output=True, text=True)
    if r.returncode:
        print('STOPPED:', r.stdout.strip(), flush=True)
    return r.returncode == 0


def zb1(path, block):
    e = dict(os.environ, ZERO_PROFILE='1', ZERO_BENCH_REPEAT='6')
    t = subprocess.run([zero, 'bench', path, '--block', str(block)], env=e, capture_output=True, text=True, errors='replace')
    out = t.stdout + t.stderr
    lines = [(float(mb), float(x), float(y)) for mb, x, y in re.findall(r'decompress (\S+) MB \(ms\):.*?rans ([\d.]+) lz ([\d.]+)', out)]
    # a pass over a large file is several batches: add them up, or the speed comes out too high
    runs, cur, got = [], [0.0, 0.0], 0.0
    size = os.path.getsize(path) / 1e6
    for mb, x, y in lines:
        cur[0] += x; cur[1] += y; got += mb
        if got >= size - 1:
            runs.append(tuple(cur)); cur, got = [0.0, 0.0], 0.0
    r = re.search(r'\(([\d.]+) %\)', out)
    return runs[0][0], runs[0][1], statistics.median(x[0] for x in runs[1:]), statistics.median(x[1] for x in runs[1:]), float(r.group(1))


def zb2(path, args):
    t = subprocess.run([zdec, cl, path, '--rounds', '6'] + shlex.split(args), capture_output=True, text=True, errors='replace')
    out = t.stdout + t.stderr
    if t.returncode or ('identical' not in out and 'nocheck' not in args):
        raise SystemExit('zdec failed, code %d, %s:\n' % (t.returncode, args) + out[-600:])
    c = re.search(r'round 0: streams ([\d.]+) ms, lz ([\d.]+) ms', out)
    m = re.search(r'median: streams ([\d.]+) ms, lz ([\d.]+) ms', out)
    r = re.search(r'MiB, ([\d.]+) %', out)
    return float(c.group(1)), float(c.group(2)), float(m.group(1)), float(m.group(2)), float(r.group(1))


res = {}
for c in corpora:
    path = os.path.join(lab, 'c', c + '.bin')
    size = os.path.getsize(path)
    for rnd in range(rounds):
        for block in (19,):
            if not free():
                sys.exit(4)
            res.setdefault((c, 'ZB1 b%d' % block), []).append(zb1(path, block))
            time.sleep(1)
        for name, args in variants:
            if not free():
                sys.exit(4)
            res.setdefault((c, name), []).append(zb2(path, args))
            time.sleep(1)
    print('==', c)
    for (cc, name), rows in res.items():
        if cc != c:
            continue
        med = lambda i: statistics.median(r[i] for r in rows)
        print('  %-14s %6.2f %% | cold streams %5.1f lz %5.1f = %5.0f MB/s | warm streams %5.1f lz %5.1f = %5.0f MB/s, lz alone %5.0f' % (
            name, rows[0][4], med(0), med(1), size / (med(0) + med(1)) / 1e3, med(2), med(3), size / (med(2) + med(3)) / 1e3, size / med(3) / 1e3), flush=True)
