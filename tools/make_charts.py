"""Draws docs/images/cpu-gpu.svg from docs/data/benchmarks.json: the whole program on the largest data set, GPU against CPU.
Two panels, each with its own scale: the time a person waits, and the processor time the CPU spends. Light and dark."""
import json, os, statistics
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(root, 'docs', 'data', 'benchmarks.json')))
x = max((s for s in d['datasets'] if 'Binaries' in s['name']), key=lambda s: s['bytes'])
med = lambda who, what, field: statistics.median(r[what][field] for r in x['runs'][who])
GPU, CPU = 'gpu', 'cpu'
panels = [
    ('Time you wait (seconds)', [('Compress', med(GPU, 'compress', 'wall'), med(CPU, 'compress', 'wall')),
                                 ('Check archive', med(GPU, 'test', 'wall'), med(CPU, 'test', 'wall')),
                                 ('Extract', med(GPU, 'extract', 'wall'), med(CPU, 'extract', 'wall'))]),
    ('Processor time the CPU spends (seconds)', [('Compress', med(GPU, 'compress', 'cpu'), med(CPU, 'compress', 'cpu')),
                                                 ('Check archive', med(GPU, 'test', 'cpu'), med(CPU, 'test', 'cpu'))]),
]
W, left, right = 720, 118, 56
bar, gap, group = 16, 2, 16
out = []
y = 66
for title, rows in panels:
    out.append(f'<text class="h" x="16" y="{y}">{title}</text>')
    y += 14
    top = max(max(a, b) for _, a, b in rows)
    scale = (W - left - right) / top
    for label, g, c in rows:
        out.append(f'<text class="t" x="{left - 10}" y="{y + bar + 5}" text-anchor="end">{label}</text>')
        for cls, v in (('g', g), ('c', c)):
            w = max(3, v * scale)
            out.append(f'<rect class="{cls}" x="{left}" y="{y}" width="{w:.1f}" height="{bar}" rx="4"><title>{"GPU" if cls == "g" else "CPU"}, {label}: {v:.1f} s</title></rect>')
            out.append(f'<text class="v" x="{left + w + 6:.1f}" y="{y + bar - 3}">{v:.1f}</text>')
            y += bar + gap
        y += group
    y += 16
H = y - 8
svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" role="img" aria-labelledby="t">
<title id="t">zero on {x['bytes'] / 2**30:.2f} GiB of Windows DLLs: GPU against CPU. The numbers are in the table in docs/benchmarks.md.</title>
<style>
  svg {{ --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e; --gpu: #2a78d6; --cpu: #c98500; font-family: "Segoe UI", system-ui, -apple-system, sans-serif; }}
  @media (prefers-color-scheme: dark) {{ svg {{ --surface: #1a1a19; --ink: #f0efec; --ink2: #c3c2b7; --gpu: #3987e5; --cpu: #d9981a; }} }}
  .bg {{ fill: var(--surface); }} .h {{ fill: var(--ink); font-size: 14px; font-weight: 600; }} .t {{ fill: var(--ink2); font-size: 13px; }}
  .v {{ fill: var(--ink); font-size: 12px; }} .g {{ fill: var(--gpu); }} .c {{ fill: var(--cpu); }}
</style>
<rect class="bg" width="{W}" height="{H}" rx="8"/>
<text class="h" x="16" y="26" style="font-size:15px">{x['bytes'] / 2**30:.2f} GiB of Windows DLLs, block 256 KiB, level 5</text>
<rect class="g" x="16" y="38" width="12" height="12" rx="3"/><text class="t" x="33" y="49">GPU (Arc iGPU)</text>
<rect class="c" x="150" y="38" width="12" height="12" rx="3"/><text class="t" x="167" y="49">CPU (18 threads)</text>
{chr(10).join(out)}
</svg>
'''
path = os.path.join(root, 'docs', 'images', 'cpu-gpu.svg')
open(path, 'w', encoding='utf-8', newline='\n').write(svg)
print(path, len(svg))
