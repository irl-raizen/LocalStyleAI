import torch
from typing import Optional
from src.models.backends.base import BaseImageBackend
from src.utils.helpers import get_logger, get_device, get_dtype

logger = get_logger("sdxl_backend")

class SDXLBackend(BaseImageBackend):
    def __init__(self):
        super().__init__()
        self.model_id = "stabilityai/stable-diffusion-xl-base-1.0"

    def load_model(self):
        if self.is_loaded and self.pipe is not None:
            return

        from diffusers import StableDiffusionXLPipeline
        self.device = get_device()
        dtype = get_dtype(self.device)

        logger.info("Loading SDXL pipeline: %s", self.model_id)
        try:
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                self.model_id, torch_dtype=dtype, low_cpu_mem_usage=True
            )
            logger.info("  ✓ Loaded SDXL")
        except Exception as exc:
            logger.error("  ✗ Failed to load SDXL: %s", exc)
            raise RuntimeError(f"Could not load SDXL model: {exc}")

        if self.device == "cuda":
            self.pipe.enable_model_cpu_offload()
            self.pipe.enable_attention_slicing()
            logger.info("  Optimizations enabled (CPU offload + attention slicing).")
        else:
            self.pipe = self.pipe.to(self.device)

        self.is_loaded = True

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        steps: int = 30,
        seed: Optional[int] = None
    ):
        if not self.is_loaded:
            self.load_model()

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            width=width,
            height=height,
            generator=generator
        )
        return result.images[0]
