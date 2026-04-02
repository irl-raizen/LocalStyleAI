"""
Kohya-ss Caption File Generator

Generates .txt caption files alongside training images for use with
the kohya-ss / sd-scripts LoRA training framework.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.helpers import STYLE_PROMPTS, DATA_DIR, get_logger

logger = get_logger("kohya_prep")

EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


def main():
    logger.info("Generating caption files for kohya-ss...")

    for style, prompt in STYLE_PROMPTS.items():
        style_dir = Path(DATA_DIR) / style
        if not style_dir.exists():
            logger.info("Skipping %s (folder not found)", style)
            continue

        count = 0
        for file in style_dir.iterdir():
            if file.suffix.lower() in EXTENSIONS:
                caption_file = file.with_suffix(".txt")
                with open(caption_file, "w", encoding="utf-8") as f:
                    f.write(prompt)
                count += 1

        logger.info("  %s: %d captions created.", style, count)

    logger.info("Done.")


if __name__ == "__main__":
    main()
