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
)
from src.models.model_router import ModelRouter
from src.ai.prompt_enhancer import enhance_prompt, DEFAULT_NEGATIVE_PROMPT
from src.ai.prompt_structurer import structure_prompt
from src.ai.prompt_composer import compose_prompt
from src.memory.memory_engine import MemoryEngine
from src.critic.regeneration_manager import RegenerationManager
import json

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
) -> tuple[Image.Image, dict]:
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
        router:         Optionally pass a ModelRouter to avoid reloading.

    Returns:
        (PIL.Image, dict) — the generated image and evaluation metadata.
    """
    device = get_device()

    # Load router if not provided
    if pipe is None:
        router = ModelRouter()
        backend = router.get_active_backend()
    else:
        router = pipe
        backend = router.get_active_backend()

    # Phase 4: Memory Injection
    memory_engine = MemoryEngine()
    prompt, style = memory_engine.inject_memory(prompt, style)

    # Phase 2: Structured Prompt Extraction
    structured = structure_prompt(prompt=prompt, style=style)
    
    if structured is not None:
        logger.info("Structured prompt extracted: %s", structured.model_dump())
        full_prompt = compose_prompt(structured)
        negative_prompt = STYLE_NEGATIVE_PROMPTS.get(style, DEFAULT_NEGATIVE_PROMPT)
    else:
        # Fall back to Phase 1 enhanced prompt system
        logger.info("Structured prompt extraction failed, falling back to Phase 1 prompt enhancer.")
        enhanced = enhance_prompt(prompt=prompt, style=style)
        
        # Keep original style prefixing fallback logic if Phase 1 also fell back
        if enhanced["enhanced_prompt"] == prompt:
            if style in STYLE_PROMPTS:
                style_prefix = STYLE_PROMPTS[style]
                full_prompt = f"{style_prefix}, {prompt}, high quality, best quality, detailed"
                negative_prompt = STYLE_NEGATIVE_PROMPTS.get(style, "")
            else:
                full_prompt = f"{prompt}, high quality, best quality, detailed"
                negative_prompt = enhanced["negative_prompt"]
        else:
            full_prompt = enhanced["enhanced_prompt"]
            negative_prompt = enhanced["negative_prompt"]

    # Load LoRA weights if available
    lora_path = os.path.join(LORA_DIR, style)
    if style in STYLE_PROMPTS and os.path.isdir(lora_path):
        try:
            backend.pipe.load_lora_weights(lora_path)
            logger.info("LoRA loaded: %s", style)
        except Exception as e:
            logger.warning("Failed to load LoRA '%s': %s", style, e)

    logger.info("Generating — prompt: '%s...'", full_prompt[:60])
    logger.info("Negative: '%s'", negative_prompt)

    logger.info("Generation started")
    
    # Phase 5: Vision Critic Auto-Regeneration
    regen_manager = RegenerationManager()
    
    def _do_generate(current_prompt, **kwargs):
        return router.generate(
            prompt=current_prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            steps=steps,
            width=width,
            height=height,
        )
        
    image, metadata = regen_manager.run_generation_loop(_do_generate, full_prompt)
    
    # Add router info to metadata
    metadata["model"] = router.active_model_name
    from src.critic.critic_models import CriticFactory
    critic = CriticFactory.get_critic()
    metadata["critic_model"] = critic.model_name if critic else "none"
    metadata["prompt"] = full_prompt

    logger.info("Generation completed")

    # Save if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        image.save(output_path)
        
        # Save metadata alongside image
        meta_path = os.path.splitext(output_path)[0] + ".json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
            
        logger.info("Saved: %s (and metadata)", output_path)

    return image, metadata


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
    router = ModelRouter()
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
                pipe=router,
            )
            logger.info("  [%d/%d] Saved: %s", idx, images_per_style, out_path)

    logger.info("=" * 50)
    logger.info("All style samples complete — check %s", EXPORTS_DIR)
