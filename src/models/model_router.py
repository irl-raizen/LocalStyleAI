import os
from typing import Optional
from src.utils.helpers import get_logger, get_device

logger = get_logger("model_router")

MODEL_SD15 = "sd15"
MODEL_SDXL = "sdxl"
MODEL_FLUX_SCHNELL = "flux_schnell"
MODEL_FLUX_DEV = "flux_dev"

# Fallback chain mapping a failed model to the next fallback option
FALLBACK_CHAIN = {
    MODEL_FLUX_DEV: MODEL_SDXL,
    MODEL_FLUX_SCHNELL: MODEL_SDXL,
    MODEL_SDXL: MODEL_SD15,
    MODEL_SD15: None  # End of the line
}

class ModelRouter:
    """
    Manages loading, caching, and routing requests to different image generation backends.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelRouter, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.backends = {}
        self.active_model_name = None
        self.active_backend = None

    def _create_backend(self, model_name: str):
        if model_name == MODEL_SD15:
            from src.models.backends.sd15_backend import SD15Backend
            return SD15Backend()
        elif model_name == MODEL_SDXL:
            from src.models.backends.sdxl_backend import SDXLBackend
            return SDXLBackend()
        elif model_name == MODEL_FLUX_SCHNELL:
            from src.models.backends.flux_schnell_backend import FluxSchnellBackend
            return FluxSchnellBackend()
        elif model_name == MODEL_FLUX_DEV:
            from src.models.backends.flux_dev_backend import FluxDevBackend
            return FluxDevBackend()
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def load_model(self, model_name: Optional[str] = None):
        """
        Loads the specified model (or the one from IMAGE_MODEL env var) with a fallback mechanism.
        Returns the successfully loaded backend instance and its name.
        """
        if model_name is None:
            model_name = os.environ.get("IMAGE_MODEL", MODEL_FLUX_DEV)

        logger.info("Model selected: %s", model_name)
        current_attempt = model_name

        while current_attempt is not None:
            try:
                # Check cache first
                if current_attempt not in self.backends:
                    self.backends[current_attempt] = self._create_backend(current_attempt)
                
                backend = self.backends[current_attempt]
                if not backend.is_loaded:
                    backend.load_model()
                    logger.info("Model loaded: %s", current_attempt)
                
                self.active_model_name = current_attempt
                self.active_backend = backend
                return backend, current_attempt

            except Exception as e:
                logger.error("Failed to load model %s: %s", current_attempt, e)
                fallback = FALLBACK_CHAIN.get(current_attempt)
                if fallback:
                    logger.warning("Fallback triggered. Attempting to load: %s", fallback)
                    current_attempt = fallback
                else:
                    logger.error("No more fallbacks available. Generation cannot proceed.")
                    raise RuntimeError("All models in the fallback chain failed to load.") from e

    def get_active_backend(self):
        """Returns the currently active backend, loading the default if none is loaded."""
        if self.active_backend is None:
            self.load_model()
        return self.active_backend

    def generate(self, prompt: str, negative_prompt: Optional[str] = None, **kwargs):
        """
        Routes the generation request to the active backend.
        """
        backend = self.get_active_backend()
        
        # Determine defaults based on the model if not provided in kwargs
        if "steps" not in kwargs:
            if self.active_model_name == MODEL_FLUX_SCHNELL:
                kwargs["steps"] = 4
            elif self.active_model_name == MODEL_FLUX_DEV:
                kwargs["steps"] = 28
            elif self.active_model_name == MODEL_SDXL:
                kwargs["steps"] = 30
            else:
                kwargs["steps"] = 20
                
        if "width" not in kwargs or "height" not in kwargs:
            if self.active_model_name in [MODEL_FLUX_DEV, MODEL_FLUX_SCHNELL, MODEL_SDXL]:
                kwargs["width"] = kwargs.get("width", 1024)
                kwargs["height"] = kwargs.get("height", 1024)
            else:
                kwargs["width"] = kwargs.get("width", 512)
                kwargs["height"] = kwargs.get("height", 512)

        logger.info("Generation started (Model: %s, steps: %s)", self.active_model_name, kwargs["steps"])
        result = backend.generate(prompt=prompt, negative_prompt=negative_prompt, **kwargs)
        logger.info("Generation completed")
        return result
