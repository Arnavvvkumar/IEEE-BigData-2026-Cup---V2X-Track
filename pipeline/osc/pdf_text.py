"""Minimal PDF text extraction without extra packages (Flate streams + Tj/TJ operators). usage: pdf_text.py <pdf>"""
import re
import sys
import zlib

raw = open(sys.argv[1], 'rb').read()
out = []
for m in re.finditer(rb'stream\r?\n(.*?)\r?\nendstream', raw, re.S):
    data = m.group(1)
    try:
        data = zlib.decompress(data)
    except Exception:
        pass
    if b'BT' not in data:
        continue
    for blk in re.findall(rb'BT(.*?)ET', data, re.S):
        line = []
        for a, b in re.findall(rb'\[(.*?)\]\s*TJ|\((.*?)(?<!\\)\)\s*Tj', blk, re.S):
            if a:
                line.append(b''.join(re.findall(rb'\((.*?)(?<!\\)\)', a, re.S)))
            else:
                line.append(b)
        if line:
            out.append(b''.join(line).decode('latin-1', 'replace'))
text = '\n'.join(out)
print(text[:6000] if text.strip() else 'NO_TEXT_FOUND (pages may be images)')
print('--- streams parsed:', len(out), '| pages:', len(re.findall(rb'/Type\s*/Page\b', raw)))
