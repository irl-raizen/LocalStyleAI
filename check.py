import sys, traceback, json, os
from pathlib import Path
from PIL import Image

MANIFEST = Path('./exports/manifest.jsonl')
style = 'lineart_clean'

entries = []
with open(MANIFEST, 'r') as f:
    for line in f:
        entry = json.loads(line)
        if entry['style'] == style:
            entries.append(entry)

print(f"Found {len(entries)} entries for {style}")
bad = []
for e in entries:
    fpath = e['file']
    try:
        img = Image.open(fpath).convert('RGB')
        img.load()  # Force full load
        img.close()
    except Exception as ex:
        print(f"BAD: {fpath} -> {ex}")
        bad.append(fpath)

print(f"\n{len(bad)} bad files found")
for b in bad:
    print(f"  Removing: {b}")
    try:
        os.remove(b)
    except:
        pass
