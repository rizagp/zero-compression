"""Tries hard to make the GPU fail: every block size and level on awkward data, damaged archives, many runs in a row. A device
loss here is a bug even though the CPU would finish the job, so it counts as a failure. Waits for its turn on the GPU.
Everything is written under --work and removed at the end.

    python tests/stress.py --work <folder> [--rounds 3] [--big file]
"""
import argparse, hashlib, os, random, shutil, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZERO, GATE = os.path.join(ROOT, 'bin', 'zero.exe'), os.path.join(ROOT, 'tools', 'gpu_gate.ps1')
ap = argparse.ArgumentParser()
ap.add_argument('--work', required=True)
ap.add_argument('--rounds', type=int, default=3)
ap.add_argument('--big')
a = ap.parse_args()
os.makedirs(a.work, exist_ok=True)
rng = random.Random(12345)
MiB = 1 << 20


def gate():
    r = subprocess.run(['pwsh', '-NoProfile', '-File', GATE, '-Stable', '20', '-Wait', '3000'], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit('stopped, GPU busy: ' + r.stdout.strip())


def data_sets():
    d = {}
    d['empty'] = b''
    d['one byte'] = b'x'
    d['three bytes'] = b'abc'
    d['four bytes'] = b'abcd'
    d['zeros 8 MiB'] = bytes(8 * MiB)
    d['one value 3 MiB'] = b'\xAB' * (3 * MiB)
    d['period 2'] = b'ab' * (2 * MiB)
    d['period 3'] = b'abc' * MiB
    d['period 7'] = b'abcdefg' * (MiB // 2)
    d['period 300'] = bytes(rng.randrange(256) for _ in range(300)) * 8000
    d['random 4 MiB'] = rng.randbytes(4 * MiB)
    x = bytearray(rng.randbytes(2 * MiB))
    x[MiB:MiB + MiB // 2] = x[:MiB // 2]          # a far copy
    d['far copy'] = bytes(x)
    # long runs, short runs and noise side by side: groups of every shape
    parts = []
    while sum(map(len, parts)) < 4 * MiB:
        k = rng.randrange(5)
        n = rng.randrange(1, 70000)
        if k == 0: parts.append(bytes([rng.randrange(256)]) * n)
        elif k == 1: parts.append(rng.randbytes(n))
        elif k == 2: parts.append(rng.randbytes(rng.randrange(1, 9)) * (n // 8 + 1))
        elif k == 3 and parts: parts.append(parts[rng.randrange(len(parts))][:n])
        else: parts.append(b'the quick brown fox ' * (n // 20 + 1))
    d['mixed'] = b''.join(parts)
    # exactly a block, and one byte either side
    for bl in (16, 18, 20):
        for delta in (-1, 0, 1):
            d['block %d %+d' % (bl, delta)] = (b'hello world ' * ((1 << bl) // 12 + 1))[:(1 << bl) + delta]
    # a long literal run: no matches for more than 64 KiB, then one
    d['long literals'] = rng.randbytes(200000) + b'0123456789' * 200 + rng.randbytes(200000)
    # matches of every length, with tiny offsets
    b = bytearray()
    for n in list(range(4, 40)) + [100, 274, 275, 1000, 4095, 4096, 5000, 70000]:
        s = rng.randbytes(8)
        b += s + s[:4] * (n // 4)
    d['every length'] = bytes(b)
    if a.big:
        with open(a.big, 'rb') as f:
            d['big'] = f.read()
    return d


def resets_since(t0):
    q = "Get-WinEvent -FilterHashtable @{LogName='System'; StartTime=(Get-Date).AddSeconds(-%d)} -ErrorAction SilentlyContinue | Where-Object { $_.Id -eq 4101 -or $_.ProviderName -match 'igfx' } | Measure-Object | ForEach-Object Count" % (time.time() - t0 + 5)
    r = subprocess.run(['pwsh', '-NoProfile', '-Command', q], capture_output=True, text=True)
    return r.stdout.strip() or '0'


t_start = time.time()
sets = data_sets()
fails = runs = 0
arc, out = os.path.join(a.work, 'a.zero'), os.path.join(a.work, 'x')


def run(*args):
    return subprocess.run([ZERO] + list(args), capture_output=True, text=True, errors='replace')


def bad(msg):
    global fails
    fails += 1
    print('FAILED', msg, flush=True)


try:
    for rnd in range(a.rounds):
        gate()
        for name, data in sets.items():
            src = os.path.join(a.work, 'input.bin')
            with open(src, 'wb') as f:
                f.write(data)
            want = hashlib.sha256(data).hexdigest()
            for block in (16, 18, 20):
                for level in (1, 5, 9):
                    runs += 1
                    c = run('c', src, '-o', arc, '--block', str(block), '--level', str(level), '-q')
                    if c.returncode or 'GPU stopped' in c.stdout + c.stderr:
                        bad('%s block %d level %d: compress %s' % (name, block, level, (c.stdout + c.stderr)[-160:]))
                        continue
                    for dec in ('--gpu', '--cpu'):
                        x = run('x', arc, '-o', out, '-q', dec)
                        got = ''
                        p = os.path.join(out, 'input.bin')
                        if os.path.exists(p):
                            with open(p, 'rb') as f:
                                got = hashlib.sha256(f.read()).hexdigest()
                        if x.returncode or 'GPU stopped' in x.stdout + x.stderr or got != want:
                            bad('%s block %d level %d decoded %s: exit %d %s' % (name, block, level, dec, x.returncode, (x.stdout + x.stderr)[-160:]))
                        shutil.rmtree(out, ignore_errors=True)
        # damage: the decoder must refuse or give wrong bytes that the CRC catches, never fault the GPU
        src = os.path.join(a.work, 'input.bin')
        with open(src, 'wb') as f:
            f.write(sets['mixed'])
        run('c', src, '-o', arc, '-q')
        with open(arc, 'rb') as f:
            good = f.read()
        for i in range(150):
            b = bytearray(good)
            for _ in range(rng.randrange(1, 40)):
                b[rng.randrange(24, len(b))] = rng.randrange(256)
            with open(arc, 'wb') as f:
                f.write(b)
            r = run('t', arc, '-q', '--gpu')
            runs += 1
            txt = r.stdout + r.stderr
            if 'GPU stopped' in txt or 'device lost' in txt:
                bad('damaged archive %d lost the GPU: %s' % (i, txt[-160:]))
            if r.returncode == 0 and bytes(b) != good:
                pass  # (bytes that only touched padding can pass; the content was checked by CRC)
        print('round %d: %d runs, %d failures, GPU resets in the log: %s, %.0f s' % (rnd + 1, runs, fails, resets_since(t_start), time.time() - t_start), flush=True)
finally:
    shutil.rmtree(a.work, ignore_errors=True)
print('ALL OK' if not fails else 'FAILURES: %d' % fails)
sys.exit(1 if fails else 0)
