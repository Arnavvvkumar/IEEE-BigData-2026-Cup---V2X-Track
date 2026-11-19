"""List and print (as text, never executed) the contents of the Codabench result zip. Read-only."""
import os
import json
import re
import zipfile

z = zipfile.ZipFile(os.environ['BDC_ROOT'] + '/prediction_result.zip')
for i in z.infolist():
    print(f'{i.file_size:10d}  {i.filename}')
for i in z.infolist():
    if i.file_size < 400_000 and i.filename.lower().endswith(('.json', '.txt', '.html', '.csv', '.md')):
        t = z.read(i.filename).decode('utf-8', 'replace')
        if i.filename.lower().endswith('.html'):
            t = re.sub(r'<[^>]+>', ' ', t); t = re.sub(r'\s+', ' ', t)
        print(f'\n===== {i.filename} ({len(t)} chars)'); print(t[:3500])
