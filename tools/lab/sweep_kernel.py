"""One kernel option at a time: compress in memory, ratio and the time of each GPU stage, alternated. Stops if the GPU gets busy.
usage: sweep_kernel.py <folder of *.bin> <rounds> <block> name=<ZERO_KOPTS>[;ENV=VAL] ..."""
import collections, glob, os, re, statistics, subprocess, sys, time
folder, rounds, block = sys.argv[1], int(sys.argv[2]), sys.argv[3]
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
zero, gate = os.path.join(root, 'bin', 'zero.exe'), os.path.join(root, 'tools', 'gpu_gate.ps1')
variants = []
for a in sys.argv[4:]:
    name, _, rest = a.partition('=')
    parts = rest.split(';')
    variants.append((name, parts[0], dict(p.split('=', 1) for p in parts[1:] if p)))


def free():
    r = subprocess.run(['pwsh', '-NoProfile', '-File', gate, '-Stable', '20', '-Wait', '3000'], capture_output=True, text=True)
    if r.returncode:
        print('STOPPED:', r.stdout.strip(), flush=True)
    return r.returncode == 0


for path in sorted(glob.glob(os.path.join(folder, '*.bin'))):
    print('==', os.path.basename(path), flush=True)
    data = collections.defaultdict(lambda: collections.defaultdict(list))
    ratio, secs = {}, collections.defaultdict(list)
    for r in range(rounds):
        for name, opts, env in variants:
            if not free():
                sys.exit(4)
            e = dict(os.environ, ZERO_PROFILE='1')
            if opts:
                e['ZERO_KOPTS'] = opts
            e.update(env)
            t = subprocess.run([zero, 'bench', path, '--block', block], env=e, capture_output=True, text=True, errors='replace')
            out = t.stdout + t.stderr
            if t.returncode or 'MISMATCH' in out or 'GPU stopped' in out:
                raise SystemExit('failed ' + name + ':\n' + out[-500:])
            m = re.search(r'compress:\s+([\d.]+) s, .*\(([\d.]+) %\)', out)
            ratio[name] = float(m.group(2))
            secs[name].append(float(m.group(1)))
            for line in out.splitlines():
                q = re.match(r'\[profile\] compress ([\d.]+) MB \(ms\):(.*)', line)
                if q and float(q.group(1)) > 15:
                    for s, v in re.findall(r'([a-z]+) ([\d.]+)', q.group(2)):
                        data[name][s].append(float(v))
            time.sleep(1)
    size = os.path.getsize(path)
    for name, _, _ in variants:
        d = data[name]
        med = lambda s: statistics.median(d[s]) if d[s] else 0
        print('  %-14s %6.2f %%  %5.0f MB/s | hcount %4.1f scatter %4.1f sort %4.1f match %5.1f lazy %3.1f parse %4.1f tokens %4.1f streams %4.1f rans %4.1f' % (
            name, ratio[name], size / statistics.median(secs[name]) / 1e6, med('hcount'), med('scatter'), med('sort'), med('match'), med('lazy'), med('parse'), med('tokens'), med('streams'), med('rans')), flush=True)
