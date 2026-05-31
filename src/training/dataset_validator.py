import os
from PIL import Image
from typing import Dict, Any, List
from src.utils.helpers import get_logger

logger = get_logger("dataset_validator")

class DatasetValidator:
    def __init__(self, min_images: int = 5):
        self.min_images = min_images
        self.supported_formats = {".png", ".jpg", ".jpeg", ".webp"}

    def validate(self, dataset_path: str) -> Dict[str, Any]:
        """
        Validates the dataset directory.
        Checks for min image count, corrupted images, and calculates some stats.
        """
        if not os.path.exists(dataset_path):
            return {"valid": False, "error": "Dataset path does not exist"}

        images = []
        corrupted = []
        resolutions = []
        
        for file in os.listdir(dataset_path):
            ext = os.path.splitext(file)[1].lower()
            if ext in self.supported_formats:
                filepath = os.path.join(dataset_path, file)
                try:
                    with Image.open(filepath) as img:
                        img.verify() # check corruption
                    # open again to get size as verify() might close it
                    with Image.open(filepath) as img:
                        resolutions.append(img.size)
                        images.append(file)
                except Exception as e:
                    corrupted.append(file)
                    
        total_valid = len(images)
        
        report = {
            "valid": total_valid >= self.min_images and len(corrupted) == 0,
            "total_images": total_valid,
            "corrupted_images": corrupted,
            "resolutions": list(set(resolutions)),
        }
        
        if total_valid < self.min_images:
            report["error"] = f"Not enough images. Found {total_valid}, need {self.min_images}."
        elif corrupted:
            report["error"] = f"Found {len(corrupted)} corrupted images."
            
        return report
