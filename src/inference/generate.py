"""
LocalStyleAI - Image Generation (Inference)

Provides a clean API for generating styled images using Stable Diffusion
with optional LoRA adapters.
"""

import os
import torch
from PIL import Image

from src.utils.helpers import (
    STYLE_PROMPTS,
    STYLE_NEGATIVE_PROMPTS,
    LORA_DIR,
    EXPORTS_DIR,
    get_device,
    get_logger,
    load_pipeline,
)

logger = get_logger("inference")


def generate_image(
    prompt: str,
    style: str = "default",
    steps: int = 30,
    guidance_scale: float = 7.5,
    width: int = 512,
    height: int = 512,
    output_path: str | None = None,
    pipe=None,
) -> Image.Image:
    """
    Generate a single image from a text prompt with optional style LoRA.

    Args:
        prompt:         Text description of the desired image.
        style:          One of 'anime_clean', 'ghibli_clean', 'lineart_clean', or 'default'.
        steps:          Number of denoising steps (higher = better quality, slower).
        guidance_scale: CFG scale — how strictly to follow the prompt.
        width:          Output width in pixels (must be multiple of 8).
        height:         Output height in pixels (must be multiple of 8).
        output_path:    If provided, save the image to this path.
        pipe:           Optionally pass a pre-loaded pipeline to avoid reloading.

    Returns:
        PIL.Image — the generated image.
    """
    device = get_device()

    # Load pipeline if not provided
    if pipe is None:
        pipe = load_pipeline(device=device)

    # Build full prompt with style prefix
    if style in STYLE_PROMPTS:
        style_prefix = STYLE_PROMPTS[style]
        full_prompt = f"{style_prefix}, {prompt}, high quality, best quality, detailed"
        negative_prompt = STYLE_NEGATIVE_PROMPTS.get(style, "")
    else:
        full_prompt = f"{prompt}, high quality, best quality, detailed"
        negative_prompt = "blurry, low quality, deformed, ugly"

    # Load LoRA weights if available
    lora_path = os.path.join(LORA_DIR, style)
    if style in STYLE_PROMPTS and os.path.isdir(lora_path):
        try:
            pipe.load_lora_weights(lora_path)
            logger.info("LoRA loaded: %s", style)
        except Exception as e:
            logger.warning("Failed to load LoRA '%s': %s", style, e)

    logger.info("Generating — prompt: '%s...'", full_prompt[:60])
    logger.info("Negative: '%s'", negative_prompt)

    result = pipe(
        prompt=full_prompt,
        negative_prompt=negative_prompt if negative_prompt else None,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
    )
    image = result.images[0]

    # Save if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        image.save(output_path)
        logger.info("Saved: %s", output_path)

    return image


def generate_style_samples(styles: list[str] | None = None, images_per_style: int = 1):
    """
    Generate sample images for each style. Useful for testing & demo purposes.

    Args:
        styles:           List of style keys to test.  Defaults to all available.
        images_per_style: How many images to generate per style.
    """
    if styles is None:
        styles = list(STYLE_PROMPTS.keys())

    test_subjects = {
        "anime_clean": "a beautiful young girl with flowing silver hair, gazing at the stars",
        "ghibli_clean": "a cozy magical cottage hidden in a lush green forest",
        "lineart_clean": "a ferocious dragon perched on a mountain",
    }

    device = get_device()
    pipe = load_pipeline(device=device)
    os.makedirs(EXPORTS_DIR, exist_ok=True)

    for style_name in styles:
        logger.info("=" * 50)
        logger.info("Generating style: %s", style_name)

        subject = test_subjects.get(style_name, "a fantasy landscape")

        for idx in range(1, images_per_style + 1):
            out_path = os.path.join(EXPORTS_DIR, f"sample_{style_name}_{idx}.png")
            generate_image(
                prompt=subject,
                style=style_name,
                output_path=out_path,
                pipe=pipe,
            )
            logger.info("  [%d/%d] Saved: %s", idx, images_per_style, out_path)

    logger.info("=" * 50)
    logger.info("All style samples complete — check %s", EXPORTS_DIR)
