import os
from pathlib import Path
from PIL import Image
import json

DATA_DIR = './data'
OUT_MANIFEST = './exports/manifest.jsonl'
RES = 512

# Supported image extensions (including webp)
SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp')

os.makedirs(os.path.dirname(OUT_MANIFEST), exist_ok=True)

count_by_style = {}
skipped = 0

with open(OUT_MANIFEST, 'w', encoding='utf-8') as fout:
    for style in sorted(os.listdir(DATA_DIR)):
        style_dir = os.path.join(DATA_DIR, style)
        if not os.path.isdir(style_dir):
            continue

        style_count = 0
        for fname in sorted(os.listdir(style_dir)):
            if not fname.lower().endswith(SUPPORTED_EXTENSIONS):
                continue

            fpath = os.path.join(style_dir, fname)
            try:
                img = Image.open(fpath).convert('RGB')
                if img.width < 200 or img.height < 200:
                    print(f"SKIP (too small: {img.width}x{img.height}): {fpath}")
                    skipped += 1
                    img.close()
                    continue
            except Exception as e:
                print(f"SKIP (corrupt): {fpath}: {e}")
                skipped += 1
                continue

            # Resize and save back as PNG for consistency
            img = img.resize((RES, RES), Image.LANCZOS)
            # Save as .png to avoid lossy re-encoding issues
            # Use the same filename stem but always output .png
            stem = Path(fname).stem
            out_fname = f"{stem}.png"
            out_path = os.path.join(style_dir, out_fname)
            img.save(out_path, format='PNG')
            img.close()

            # Use forward slashes for cross-platform path consistency in manifest
            manifest_path = out_path.replace('\\', '/')

            entry = {
                'file': manifest_path,
                'style': style,
            }

            fout.write(json.dumps(entry) + '\n')
            style_count += 1

        count_by_style[style] = style_count

print(f"\n{'='*50}")
print(f"Manifest written to: {OUT_MANIFEST}")
print(f"Skipped: {skipped}")
for s, c in count_by_style.items():
    print(f"  {s}: {c} images")
print(f"  Total: {sum(count_by_style.values())} images")
print(f"{'='*50}")
