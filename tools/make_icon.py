"""Draws tools/zero.ico (needs Pillow; the .ico is committed, so a build does not): a white zero on the Windows accent blue."""
import os
from PIL import Image, ImageDraw

S = 1024
img = Image.new('RGBA', (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
d.rounded_rectangle((40, 40, S - 40, S - 40), radius=200, fill=(0, 103, 192, 255))
d.ellipse((262, 170, S - 262, S - 170), fill=(255, 255, 255, 255))
d.ellipse((392, 300, S - 392, S - 300), fill=(0, 103, 192, 255))
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'zero.ico')
img.resize((256, 256), Image.LANCZOS).save(out, sizes=[(16, 16), (20, 20), (24, 24), (32, 32), (40, 40), (48, 48), (64, 64), (128, 128), (256, 256)])
print(out, os.path.getsize(out))
