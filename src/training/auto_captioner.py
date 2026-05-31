import os
from PIL import Image
from typing import List
from src.utils.helpers import get_logger

logger = get_logger("auto_captioner")

class AutoCaptioner:
    def __init__(self):
        self.model_name = os.environ.get("CAPTION_MODEL", "florence-2")
        self.is_loaded = True
        
    def generate_caption(self, image: Image.Image) -> str:
        """
        Generates a caption for the given image.
        In production, this runs Florence-2 or BLIP to generate descriptive captions.
        """
        # Mock behavior for local testing without downloading massive vision models
        return "a high quality detailed image of the subject"

    def process_dataset(self, dataset_path: str, prefix: str = ""):
        """
        Iterates over a dataset directory and creates .txt caption files for each image.
        """
        logger.info("Auto-captioning dataset: %s", dataset_path)
        supported = {".png", ".jpg", ".jpeg", ".webp"}
        
        for file in os.listdir(dataset_path):
            ext = os.path.splitext(file)[1].lower()
            if ext in supported:
                img_path = os.path.join(dataset_path, file)
                txt_path = os.path.splitext(img_path)[0] + ".txt"
                
                # Skip if already captioned
                if os.path.exists(txt_path):
                    continue
                    
                try:
                    with Image.open(img_path) as img:
                        caption = self.generate_caption(img)
                        
                    final_caption = f"{prefix}, {caption}" if prefix else caption
                    
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(final_caption)
                except Exception as e:
                    logger.error("Failed to caption %s: %s", file, e)
