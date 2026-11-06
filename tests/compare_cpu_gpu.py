"""Compares the GPU and CPU codecs on large inputs, the whole program, as docs/benchmarks.md reports it.

    python tests/compare_cpu_gpu.py --work D:/temp/zero-bench --model path/to/model.gguf [--model2 path/to/other.gguf]

Inputs: Windows SDK headers (text) and System32 DLLs (binaries) concatenated into single files, plus any model files.
The codecs alternate in each round, with idle gaps. For each it records compress, test (decompress and verify, nothing
written) and extract (written to disk): wall time, the codec's own time, CPU time of the process, peak memory, and the
ratio. Extracted files are compared by SHA-256. Needs psutil. Everything under --work is deleted at the end; the results
go to docs/data/benchmarks.json.
"""
import argparse, glob, hashlib, json, os, platform, re, shutil, statistics, subprocess, sys, time
import psutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GIB = 1 << 30


def concat(out, paths, limit):
    """Up to limit bytes of paths, in sorted order, into one file."""
    done = 0
    with open(out, 'wb') as o:
        for p in paths:
            if done >= limit:
                break
            try:
                with open(p, 'rb') as f:
                    while done < limit:
                        c = f.read(min(1 << 22, limit - done))
                        if not c:
                            break
                        o.write(c)
                        done += len(c)
            except OSError:
                pass  # (in use or not readable)
    return done


def tree(top, patterns):
    r = []
    for pat in patterns:
        r += glob.glob(os.path.join(top, '**', pat), recursive=True)
    return sorted(set(r))


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 22), b''):
            h.update(c)
    return h.hexdigest()


def run(args):
    """Wall time, CPU seconds and peak working set of one process, and its output."""
    t = time.perf_counter()
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='replace')
    ps = psutil.Process(p.pid)
    cpu, peak = 0.0, 0
    while p.poll() is None:
        try:
            c = ps.cpu_times()
            cpu = c.user + c.system
            peak = max(peak, ps.memory_info().peak_wset if hasattr(ps.memory_info(), 'peak_wset') else ps.memory_info().rss)
        except psutil.Error:
            pass
        time.sleep(0.05)
    wall = time.perf_counter() - t
    out = p.stdout.read()
    return dict(code=p.returncode, wall=wall, cpu=cpu, peak=peak), out


def wait_for_gpu():
    """Takes turns with other projects on the GPU."""
    gate = os.path.join(ROOT, 'tools', 'gpu_gate.ps1')
    t = subprocess.run(['pwsh', '-NoProfile', '-File', gate, '-Stable', '20', '-Wait', '3000'], capture_output=True, text=True)
    if t.returncode:
        raise SystemExit('stopped, GPU busy: ' + t.stdout.strip())


def one(exe, flag, src, work, block, level, gap, verify):
    arc, xdir = os.path.join(work, 'a.zero'), os.path.join(work, 'x')
    r = {}
    wait_for_gpu()
    time.sleep(gap)
    c, out = run([exe, 'c', src, '-o', arc, '--block', str(block), '--level', str(level), '-q', flag])
    if c['code']:
        raise SystemExit('compress failed: ' + out[-400:])
    m = re.search(r'\(([\d.]+) %\).*codec ([\d.]+) s', out)
    c['ratio'], c['codec'] = float(m.group(1)), float(m.group(2))
    c['bytes'] = os.path.getsize(arc)
    r['compress'] = c
    wait_for_gpu()
    time.sleep(gap)
    t, out = run([exe, 't', arc, '-q', flag])
    if t['code']:
        raise SystemExit('test failed: ' + out[-400:])
    r['test'] = t
    wait_for_gpu()
    time.sleep(gap)
    x, out = run([exe, 'x', arc, '-o', xdir, '-q', flag])
    if x['code']:
        raise SystemExit('extract failed: ' + out[-400:])
    m = re.search(r'codec ([\d.]+) s', out)
    x['codec'] = float(m.group(1)) if m else None
    r['extract'] = x
    if verify:
        got = os.path.join(xdir, os.path.basename(src))
        r['identical'] = sha(got) == sha(src)
        other = '--cpu' if flag == '--gpu' else '--gpu'
        c2, out = run([exe, 't', arc, '-q', other])  # (made by one codec, read by the other)
        r['cross_decoded'] = c2['code'] == 0
    shutil.rmtree(xdir, ignore_errors=True)
    os.remove(arc)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', required=True)
    ap.add_argument('--model', action='append', default=[])
    ap.add_argument('--gap', type=float, default=8)
    ap.add_argument('--binaries-gib', type=float, default=8)
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    exe = os.path.join(a.work, 'zero.exe')
    shutil.copy(os.path.join(ROOT, 'bin', 'zero.exe'), exe)  # (a build meanwhile cannot change what is measured)
    data = []
    text = os.path.join(a.work, 'text.bin')
    n = concat(text, tree(r'C:\Program Files (x86)\Windows Kits\10\Include', ['*.h', '*.idl', '*.inl', '*.hpp']), 2 * GIB)
    data.append(dict(name='Text (Windows SDK headers)', path=text, bytes=n, rounds=3))
    dlls = tree(r'C:\Windows\System32', ['*.dll'])
    bin8 = os.path.join(a.work, 'bin8.bin')
    n = concat(bin8, dlls, int(a.binaries_gib * GIB))
    bin1 = os.path.join(a.work, 'bin1.bin')
    n1 = concat(bin1, [bin8], GIB)
    data.append(dict(name='Binaries, 1 GiB (System32 DLLs)', path=bin1, bytes=n1, rounds=3))
    data.append(dict(name='Binaries, %.1f GiB (System32 DLLs)' % (n / GIB), path=bin8, bytes=n, rounds=2))
    rnd = os.path.join(a.work, 'random.bin')
    with open(rnd, 'wb') as f:
        for _ in range(2 * 1024 // 16):
            f.write(os.urandom(16 << 20))
    data.append(dict(name='Random, 2 GiB (cannot be compressed)', path=rnd, bytes=os.path.getsize(rnd), rounds=2))
    for m in a.model:
        big = os.path.getsize(m) > 4 * GIB
        data.append(dict(name=os.path.basename(m), path=m, bytes=os.path.getsize(m), rounds=1 if big else 3, on_usb=True))
    bat = psutil.sensors_battery()
    results = dict(date=time.strftime('%Y-%m-%d'), machine=dict(cpu='Intel Core Ultra 5 125H', threads=psutil.cpu_count(), ram_gib=round(psutil.virtual_memory().total / GIB),
                   gpu='Arc iGPU (Xe-LPG, 112 EUs)', os=platform.platform(), plugged_in=bat.power_plugged if bat else None), block_log=18, level=5, datasets=[])
    out_json = os.path.join(ROOT, 'docs', 'data', 'benchmarks.json')
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    try:
        for d in data:
            runs = dict(gpu=[], cpu=[])
            print('==', d['name'], '%.2f GiB' % (d['bytes'] / GIB), flush=True)
            for r in range(d['rounds']):
                for label in ('gpu', 'cpu'):
                    x = one(exe, '--' + label, d['path'], a.work, 18, 5, a.gap, r == 0)
                    runs[label].append(x)
                    print('  %s round %d: compress %.1f s (codec %.1f) %.2f %%, test %.1f s, extract %.1f s, identical %s' % (
                        label, r + 1, x['compress']['wall'], x['compress']['codec'], x['compress']['ratio'], x['test']['wall'], x['extract']['wall'], x.get('identical', '-')), flush=True)
            results['datasets'].append(dict(name=d['name'], bytes=d['bytes'], on_usb=d.get('on_usb', False), runs=runs))
            with open(out_json, 'w') as f:
                json.dump(results, f, indent=1)
    finally:
        shutil.rmtree(a.work, ignore_errors=True)
    print('written', out_json, 'work folder left:', os.path.exists(a.work))


if __name__ == '__main__':
    main()
