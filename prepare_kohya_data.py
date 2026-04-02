import os
from pathlib import Path

# Must match train_lora.py / server.py
STYLE_PROMPTS = {
    "anime_clean": "anime style, cel shaded, vibrant colors, clean lines, anime art",
    "ghibli_clean": "studio ghibli style, hand painted, watercolor, soft lighting, miyazaki art",
    "lineart_clean": "line art, ink drawing, black and white, clean outlines, sketch art, no color",
}

DATA_DIR = Path('data')
EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp')

print("Generating caption files for kohya-ss...")

for style, prompt in STYLE_PROMPTS.items():
    style_dir = DATA_DIR / style
    if not style_dir.exists():
        print(f"Skipping {style} (folder not found)")
        continue
    
    count = 0
    for file in style_dir.iterdir():
        if file.suffix.lower() in EXTENSIONS:
            caption_file = file.with_suffix('.txt')
            with open(caption_file, 'w', encoding='utf-8') as f:
                f.write(prompt)
            count += 1
    
    print(f"  {style}: {count} captions created.")

print("Done.")
