"""Print the declaration inside a submission zip and run the scorer's phrase check. usage: verify_zip.py <zip>"""
import sys
import zipfile

with zipfile.ZipFile(sys.argv[1]) as z:
    names = z.namelist()
    decl = z.read('declaration.txt').decode('utf-8', errors='replace')
print(decl)
print('scorer phrase check:', 'I confirm that I did not train on any frame listed in'.lower() in decl.lower())
print('entries', len(names), '| non-scan entries', [n for n in names if not n.startswith('synthetic/')], '| scans', sum(n.startswith('synthetic/') for n in names))
