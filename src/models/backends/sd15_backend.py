import torch
from typing import Optional
from src.models.backends.base import BaseImageBackend
from src.utils.helpers import get_logger, get_device, get_dtype

logger = get_logger("sd15_backend")

class SD15Backend(BaseImageBackend):
    def __init__(self):
        super().__init__()
        self.model_ids = [
            "Lykon/dreamshaper-8",
            "SG161222/Realistic_Vision_V5.1_noVAE",
            "runwayml/stable-diffusion-v1-5"
        ]

    def load_model(self):
        if self.is_loaded and self.pipe is not None:
            return

        from diffusers import StableDiffusionPipeline
        self.device = get_device()
        dtype = get_dtype(self.device)

        logger.info("Loading SD1.5 pipeline...")
        for model_id in self.model_ids:
            try:
                logger.info("  Trying: %s ...", model_id)
                self.pipe = StableDiffusionPipeline.from_pretrained(
                    model_id, torch_dtype=dtype, low_cpu_mem_usage=True
                )
                logger.info("  ✓ Loaded: %s", model_id)
                break
            except Exception as exc:
                logger.warning("  ✗ Failed: %s — %s", model_id, exc)

        if self.pipe is None:
            raise RuntimeError("Could not load any SD1.5 base model.")

        self.pipe.safety_checker = None

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
        width: int = 512,
        height: int = 512,
        steps: int = 20,
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
