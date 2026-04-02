"""
LocalStyleAI - Dataset Preparation

Scans the data/ directory for style subdirectories, validates and resizes
images to a uniform resolution, and writes a JSONL manifest file used by
the training pipeline.
"""

import os
import json
from pathlib import Path
from PIL import Image

from src.utils.helpers import DATA_DIR, EXPORTS_DIR, MANIFEST_PATH, get_logger

logger = get_logger("dataset_prep")

SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
DEFAULT_RESOLUTION = 512


def prepare_dataset(resolution: int = DEFAULT_RESOLUTION):
    """
    Scan data/<style>/ folders, resize images, and write a JSONL manifest.

    Args:
        resolution: Target size (width & height) for all training images.
    """
    os.makedirs(EXPORTS_DIR, exist_ok=True)

    count_by_style: dict[str, int] = {}
    skipped = 0

    with open(MANIFEST_PATH, "w", encoding="utf-8") as fout:
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
                    img = Image.open(fpath).convert("RGB")
                    if img.width < 200 or img.height < 200:
                        logger.warning("SKIP (too small %dx%d): %s", img.width, img.height, fpath)
                        skipped += 1
                        img.close()
                        continue
                except Exception as e:
                    logger.warning("SKIP (corrupt): %s — %s", fpath, e)
                    skipped += 1
                    continue

                # Resize and save as PNG
                img = img.resize((resolution, resolution), Image.LANCZOS)
                stem = Path(fname).stem
                out_path = os.path.join(style_dir, f"{stem}.png")
                img.save(out_path, format="PNG")
                img.close()

                entry = {
                    "file": out_path.replace("\\", "/"),
                    "style": style,
                }
                fout.write(json.dumps(entry) + "\n")
                style_count += 1

            count_by_style[style] = style_count

    # Report
    logger.info("=" * 50)
    logger.info("Manifest written to: %s", MANIFEST_PATH)
    logger.info("Skipped: %d", skipped)
    for s, c in count_by_style.items():
        logger.info("  %s: %d images", s, c)
    logger.info("  Total: %d images", sum(count_by_style.values()))
    logger.info("=" * 50)


if __name__ == "__main__":
    prepare_dataset()
