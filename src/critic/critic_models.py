import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image

import torch
from src.utils.helpers import get_device, get_logger

logger = get_logger("critic_models")

class BaseVisionCritic(ABC):
    def __init__(self):
        self.device = get_device()
        self.is_loaded = False
        self.model_name = "unknown"

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def calculate_similarity(self, image: Image.Image, texts: List[str]) -> List[float]:
        """Returns similarity scores between 0.0 and 1.0 for each text string."""
        pass


class ClipCritic(BaseVisionCritic):
    def __init__(self):
        super().__init__()
        self.model_name = "clip"
        self.model_id = "openai/clip-vit-base-patch32"

    def load(self):
        if self.is_loaded: return
        from transformers import CLIPProcessor, CLIPModel
        logger.info("Loading CLIP model: %s", self.model_id)
        try:
            self.processor = CLIPProcessor.from_pretrained(self.model_id)
            self.model = CLIPModel.from_pretrained(self.model_id).to(self.device)
            self.is_loaded = True
            logger.info("✓ CLIP loaded successfully.")
        except Exception as e:
            logger.error("Failed to load CLIP: %s", e)
            raise

    def calculate_similarity(self, image: Image.Image, texts: List[str]) -> List[float]:
        if not self.is_loaded: self.load()
        inputs = self.processor(text=texts, images=image, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Normalize to 0.0 - 1.0 range approx
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).cpu().numpy().flatten().tolist()
            # Or cosine similarity
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            similarities = (image_embeds @ text_embeds.T).cpu().numpy().flatten()
            
            # Clamp between 0 and 1
            scores = [max(0.0, min(1.0, float(sim))) for sim in similarities]
            return scores


class SiglipCritic(BaseVisionCritic):
    def __init__(self):
        super().__init__()
        self.model_name = "siglip"
        self.model_id = "google/siglip-base-patch16-224"

    def load(self):
        if self.is_loaded: return
        from transformers import AutoProcessor, AutoModel
        logger.info("Loading SigLIP model: %s", self.model_id)
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = AutoModel.from_pretrained(self.model_id).to(self.device)
            self.is_loaded = True
            logger.info("✓ SigLIP loaded successfully.")
        except Exception as e:
            logger.error("Failed to load SigLIP: %s", e)
            raise

    def calculate_similarity(self, image: Image.Image, texts: List[str]) -> List[float]:
        if not self.is_loaded: self.load()
        inputs = self.processor(text=texts, images=image, padding="max_length", return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Siglip uses dot product or sigmoid
            logits_per_image = outputs.logits_per_image
            probs = torch.sigmoid(logits_per_image).cpu().numpy().flatten().tolist()
            return probs


class CriticFactory:
    _instance = None
    
    @classmethod
    def get_critic(cls) -> Optional[BaseVisionCritic]:
        if cls._instance is not None and cls._instance.is_loaded:
            return cls._instance
            
        requested = os.environ.get("VISION_MODEL", "siglip").lower()
        
        # Priority order: SigLIP -> CLIP -> Florence-2 (future)
        models_to_try = []
        if requested == "siglip":
            models_to_try = [SiglipCritic, ClipCritic]
        elif requested == "clip":
            models_to_try = [ClipCritic, SiglipCritic]
        else:
            models_to_try = [SiglipCritic, ClipCritic]
            
        for critic_cls in models_to_try:
            critic = critic_cls()
            try:
                critic.load()
                cls._instance = critic
                return critic
            except Exception as e:
                logger.warning("Critic %s failed to load, falling back. Error: %s", critic.model_name, e)
                
        logger.error("All vision critic models failed to load. Critic is unavailable.")
        return None
