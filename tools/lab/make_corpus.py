"""Test data for zlab, from files every Windows machine with the SDK has, so no personal data is involved. Read only.
usage: make_corpus.py <folder> [MiB per corpus]. A model file can be added: make_corpus.py <folder> 128 model=<file>"""
import glob, os, sys

out = sys.argv[1]
limit = int(sys.argv[2]) << 20 if len(sys.argv) > 2 else 128 << 20
os.makedirs(out, exist_ok=True)


def take(name, paths, step=1):
    done = 0
    with open(os.path.join(out, name + '.bin'), 'wb') as o:
        for p in sorted(paths)[::step]:
            if done >= limit:
                break
            try:
                with open(p, 'rb') as f:
                    c = f.read(limit - done)
                o.write(c)
                done += len(c)
            except OSError:
                pass
    print(name, done >> 20, 'MiB')


def tree(top, *patterns):
    r = []
    for pat in patterns:
        r += glob.glob(os.path.join(top, '**', pat), recursive=True)
    return r


dlls = tree(r'C:\Windows\System32', '*.dll')
take('dll', dlls, step=max(1, len(dlls) // 1500))   # (across the alphabet, not only the first letters)
take('text', tree(r'C:\Program Files (x86)\Windows Kits\10\Include', '*.h', '*.idl', '*.hpp', '*.inl'))
take('lib', tree(r'C:\Program Files (x86)\Windows Kits\10\Lib', '*.lib') + tree(r'C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC', '*.lib', '*.pdb'), step=3)
for a in sys.argv[3:]:
    name, path = a.split('=', 1)
    take(name, [path])
