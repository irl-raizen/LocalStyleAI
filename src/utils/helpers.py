"""
LocalStyleAI - Shared Helpers & Constants

Central module for style definitions, model loading, and common utilities
used across training, inference, and the API server.
"""

import os
import sys
import logging
import torch

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a consistently-formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ---------------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
LORA_DIR = os.path.join(PROJECT_ROOT, "loras")
EXPORTS_DIR = os.path.join(PROJECT_ROOT, "exports")
CONFIGS_DIR = os.path.join(PROJECT_ROOT, "configs")
MANIFEST_PATH = os.path.join(EXPORTS_DIR, "manifest.jsonl")

# ---------------------------------------------------------------------------
# Style Definitions
# ---------------------------------------------------------------------------
STYLE_PROMPTS: dict[str, str] = {
    "anime_clean": "anime style, cel shaded, vibrant colors, clean lines, anime art",
    "ghibli_clean": "studio ghibli style, hand painted, watercolor, soft lighting, miyazaki art",
    "lineart_clean": "line art, ink drawing, black and white, clean outlines, sketch art, no color",
}

STYLE_NEGATIVE_PROMPTS: dict[str, str] = {
    "anime_clean": "photograph, realistic, 3d render, line art, sketch, ghibli",
    "ghibli_clean": "photograph, realistic, 3d render, anime, cel shaded, line art, sketch",
    "lineart_clean": "photograph, realistic, color, painting, anime, cel shaded, ghibli, vibrant",
}

AVAILABLE_STYLES = list(STYLE_PROMPTS.keys())

# ---------------------------------------------------------------------------
# Base Model Priority List
# ---------------------------------------------------------------------------
BASE_MODELS = [
    "Lykon/dreamshaper-8",
    "SG161222/Realistic_Vision_V5.1_noVAE",
    "runwayml/stable-diffusion-v1-5",
]

# ---------------------------------------------------------------------------
# Device / dtype helpers
# ---------------------------------------------------------------------------

def get_device() -> str:
    """Return 'cuda' if a GPU is available, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_dtype(device: str | None = None) -> torch.dtype:
    """Return float16 for CUDA, float32 for CPU."""
    if device is None:
        device = get_device()
    return torch.float16 if device == "cuda" else torch.float32


# ---------------------------------------------------------------------------
# Pipeline loader (shared by inference & API)
# ---------------------------------------------------------------------------

def load_pipeline(device: str | None = None, low_vram: bool = True):
    """
    Load a Stable Diffusion pipeline, trying models in priority order.

    Args:
        device:   'cuda' or 'cpu'.  Auto-detected if None.
        low_vram: If True, enable attention slicing & CPU offload (for ≤6 GB).

    Returns:
        StableDiffusionPipeline instance, or raises RuntimeError.
    """
    from diffusers import StableDiffusionPipeline

    logger = get_logger("pipeline")

    if device is None:
        device = get_device()
    dtype = get_dtype(device)

    logger.info("Loading Stable Diffusion pipeline  (device=%s, dtype=%s)", device, dtype)

    pipe = None
    for model_id in BASE_MODELS:
        try:
            logger.info("  Trying: %s ...", model_id)
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id, torch_dtype=dtype, low_cpu_mem_usage=True
            )
            logger.info("  ✓ Loaded: %s", model_id)
            break
        except Exception as exc:
            logger.warning("  ✗ Failed: %s — %s", model_id, exc)

    if pipe is None:
        raise RuntimeError(
            "Could not load any base model. Check your internet connection "
            "and Hugging Face cache."
        )

    pipe.safety_checker = None

    if device == "cuda" and low_vram:
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
        logger.info("  Low-VRAM optimizations enabled (CPU offload + attention slicing).")
    else:
        pipe = pipe.to(device)
        if device == "cuda":
            pipe.enable_attention_slicing()

    return pipe
