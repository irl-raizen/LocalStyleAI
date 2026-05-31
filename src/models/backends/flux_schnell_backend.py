import torch
from typing import Optional
from src.models.backends.base import BaseImageBackend
from src.utils.helpers import get_logger, get_device

logger = get_logger("flux_schnell_backend")

class FluxSchnellBackend(BaseImageBackend):
    def __init__(self):
        super().__init__()
        self.model_id = "black-forest-labs/FLUX.1-schnell"

    def load_model(self):
        if self.is_loaded and self.pipe is not None:
            return

        from diffusers import FluxPipeline
        self.device = get_device()
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        logger.info("Loading FLUX.1-schnell pipeline: %s", self.model_id)
        try:
            self.pipe = FluxPipeline.from_pretrained(
                self.model_id, torch_dtype=dtype, low_cpu_mem_usage=True
            )
            logger.info("  ✓ Loaded FLUX.1-schnell")
        except Exception as exc:
            logger.error("  ✗ Failed to load FLUX.1-schnell: %s", exc)
            raise RuntimeError(f"Could not load FLUX.1-schnell model: {exc}")

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
        steps: int = 4,
        seed: Optional[int] = None
    ):
        if not self.is_loaded:
            self.load_model()

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # FLUX models generally don't use negative_prompt or guidance_scale in the same way,
        # but Schnell specifically has a low step count and fixed guidance.
        # We pass only what is strictly required or allowed by the pipeline.
        result = self.pipe(
            prompt=prompt,
            num_inference_steps=steps,
            width=width,
            height=height,
            generator=generator,
            guidance_scale=0.0 # Schnell does not use CFG
        )
        return result.images[0]
