from abc import ABC, abstractmethod
from typing import Optional

class BaseImageBackend(ABC):
    """
    Abstract base class for all image generation backends.
    """

    def __init__(self):
        self.pipe = None
        self.is_loaded = False
        self.device = "cpu"

    @abstractmethod
    def load_model(self):
        """Load the model pipeline into memory."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 512,
        height: int = 512,
        steps: int = 20,
        seed: Optional[int] = None
    ):
        """
        Generate an image using the loaded model.
        Returns a PIL Image.
        """
        pass
