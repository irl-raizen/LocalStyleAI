import torch
from typing import Optional
from src.models.backends.base import BaseImageBackend
from src.utils.helpers import get_logger, get_device

logger = get_logger("flux_dev_backend")

class FluxDevBackend(BaseImageBackend):
    def __init__(self):
        super().__init__()
        self.model_id = "black-forest-labs/FLUX.1-dev"

    def load_model(self):
        if self.is_loaded and self.pipe is not None:
            return

        from diffusers import FluxPipeline
        self.device = get_device()
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        logger.info("Loading FLUX.1-dev pipeline: %s", self.model_id)
        try:
            self.pipe = FluxPipeline.from_pretrained(
                self.model_id, torch_dtype=dtype, low_cpu_mem_usage=True
            )
            logger.info("  ✓ Loaded FLUX.1-dev")
        except Exception as exc:
            logger.error("  ✗ Failed to load FLUX.1-dev: %s", exc)
            raise RuntimeError(f"Could not load FLUX.1-dev model: {exc}")

        if self.device == "cuda":
            self.pipe.enable_model_cpu_offload()
            try:
                self.pipe.vae.enable_slicing()
                self.pipe.vae.enable_tiling()
            except AttributeError:
                pass
            logger.info("  Optimizations enabled (CPU offload + VAE slicing/tiling).")
        else:
            self.pipe = self.pipe.to(self.device)

        self.is_loaded = True

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        steps: int = 28,
        seed: Optional[int] = None
    ):
        if not self.is_loaded:
            self.load_model()

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        result = self.pipe(
            prompt=prompt,
            num_inference_steps=steps,
            width=width,
            height=height,
            generator=generator,
            guidance_scale=3.5 # Dev uses CFG, usually around 3.5
        )
        return result.images[0]
