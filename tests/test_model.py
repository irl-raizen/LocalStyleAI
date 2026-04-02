"""
LocalStyleAI - Model loading & generation test.

Verifies that the Stable Diffusion pipeline loads correctly
and can produce images for each style.
"""

import os
import sys
import torch

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.helpers import (
    STYLE_PROMPTS,
    STYLE_NEGATIVE_PROMPTS,
    EXPORTS_DIR,
    get_device,
    get_logger,
    load_pipeline,
)

logger = get_logger("test_model")

TEST_SUBJECTS = {
    "anime_clean": "a beautiful young girl with flowing silver hair, gazing at the stars",
    "ghibli_clean": "a cozy magical cottage hidden in a lush green forest",
    "lineart_clean": "a ferocious dragon perched on a mountain",
}


def test_pipeline_loads():
    """Smoke test: verify the pipeline loads without crashing."""
    device = get_device()
    logger.info("Device: %s", device)

    pipe = load_pipeline(device=device)
    assert pipe is not None, "Pipeline failed to load"
    logger.info("✓ Pipeline loaded successfully.")
    return pipe


def test_generation(pipe):
    """Generate one image per style to verify end-to-end."""
    os.makedirs(EXPORTS_DIR, exist_ok=True)

    for style_name in STYLE_PROMPTS:
        logger.info("=" * 50)
        logger.info("Testing style: %s", style_name)

        style_prefix = STYLE_PROMPTS[style_name]
        subject = TEST_SUBJECTS.get(style_name, "a fantasy landscape")
        full_prompt = f"{style_prefix}, {subject}, high quality, detailed, sharp focus"
        negative_prompt = STYLE_NEGATIVE_PROMPTS[style_name]

        logger.info("Prompt: %s", full_prompt)

        image = pipe(
            full_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=20,   # Fewer steps for a quick test
            guidance_scale=7.5,
            height=512,
            width=512,
        ).images[0]

        out_path = os.path.join(EXPORTS_DIR, f"test_{style_name}.png")
        image.save(out_path)
        logger.info("✓ Saved: %s", out_path)

    logger.info("=" * 50)
    logger.info("All style tests passed!")


if __name__ == "__main__":
    pipe = test_pipeline_loads()
    test_generation(pipe)
