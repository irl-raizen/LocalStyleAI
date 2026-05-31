import torch
from PIL import Image
from typing import Optional, Dict, Any

from src.models.model_router import ModelRouter
from src.utils.helpers import get_logger, get_device

logger = get_logger("inpainting_engine")

class InpaintingEngine:
    def __init__(self):
        self.device = get_device()
        self.router = ModelRouter()

    def apply_edit(
        self, 
        image: Image.Image, 
        mask: Optional[Image.Image], 
        prompt: str, 
        negative_prompt: str = ""
    ) -> Image.Image:
        """
        Attempts to perform inpainting using the current backend.
        If inpainting is unsupported by the current pipeline, falls back to img2img.
        """
        backend = self.router.get_active_backend()
        
        # Determine the base model ID to load the correct pipeline component
        # We can use AutoPipelineForImage2Image or AutoPipelineForInpainting
        # To avoid reloading massive models continuously, we'll try to reuse the backend's pipe components
        # Or simply run img2img if the pipe supports it.
        
        try:
            logger.info("Attempting inpainting/img2img with active model: %s", self.router.active_model_name)
            
            # Most pipelines from diffusers can be cast to img2img using the from_pipe method
            if "flux" in self.router.active_model_name:
                # FLUX supports img2img natively in its pipeline via strength
                # FLUX Fill (inpainting) is a separate model. We will use img2img as fallback.
                logger.info("Using FLUX img2img fallback.")
                return backend.pipe(
                    prompt=prompt,
                    image=image,
                    strength=0.6,
                    guidance_scale=3.5 if "dev" in self.router.active_model_name else 0.0,
                    num_inference_steps=20 if "dev" in self.router.active_model_name else 4,
                    max_sequence_length=512,
                ).images[0]
            elif "sdxl" in self.router.active_model_name:
                from diffusers import StableDiffusionXLImg2ImgPipeline
                img2img_pipe = StableDiffusionXLImg2ImgPipeline.from_pipe(backend.pipe).to(self.device)
                return img2img_pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=image,
                    strength=0.6,
                    num_inference_steps=30,
                ).images[0]
            else: # SD1.5
                from diffusers import StableDiffusionImg2ImgPipeline
                img2img_pipe = StableDiffusionImg2ImgPipeline.from_pipe(backend.pipe).to(self.device)
                return img2img_pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=image,
                    strength=0.6,
                    num_inference_steps=30,
                ).images[0]
                
        except Exception as e:
            logger.error("Inpainting / Img2Img failed: %s", e)
            logger.warning("Returning original image to prevent crash.")
            return image
