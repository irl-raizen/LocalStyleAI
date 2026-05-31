"""
LocalStyleAI - Structured Prompt Builder (Extraction)

Analyzes a user's prompt using a local LLM via Ollama and extracts its
semantic components into a structured Pydantic model.
"""

import os
import json
import requests
from typing import Optional
from pydantic import BaseModel, Field, model_validator
from src.utils.helpers import get_logger

logger = get_logger("prompt_structurer")

# Configuration defaults (align with prompt_enhancer)
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_OLLAMA_MODEL = "qwen3:8b"
DEFAULT_OLLAMA_TIMEOUT = 10.0


class StructuredPrompt(BaseModel):
    """
    Structured representation of a text-to-image prompt.
    Includes current semantic components and placeholders for future compatibility.
    """
    subject: str = Field(default="", description="The main subject of the image")
    appearance: str = Field(default="", description="The clothing, physical details, or appearance of the subject")
    action: str = Field(default="", description="The action being performed or pose of the subject")
    environment: str = Field(default="", description="The location, setting, or background of the scene")
    lighting: str = Field(default="", description="The lighting conditions (e.g. golden hour, dramatic shadows)")
    camera: str = Field(default="", description="The camera angle, shot type, or composition details")
    style: str = Field(default="", description="The art style or specific aesthetic requested")
    mood: str = Field(default="", description="The emotional mood or atmosphere")
    quality: str = Field(default="", description="Quality tag indicators (e.g. masterpiece, highly detailed)")

    # Future compatibility fields (currently defaults to empty strings)
    character_name: str = Field(default="", description="Name of a specific character")
    character_traits: str = Field(default="", description="Traits or personality description of a character")
    pose: str = Field(default="", description="Pose description of the subject")
    background: str = Field(default="", description="Dedicated background description")
    color_palette: str = Field(default="", description="Specific color palette or color scheme")

    @model_validator(mode='before')
    @classmethod
    def normalize_fields(cls, data):
        if not isinstance(data, dict):
            return data
        
        # Normalize fields that should be strings but might be dicts or lists from LLM
        for key, value in data.items():
            if isinstance(value, dict):
                parts = []
                for k, v in value.items():
                    if isinstance(v, (dict, list)):
                        parts.append(f"{k}: {json.dumps(v)}")
                    else:
                        parts.append(f"{k}: {v}")
                data[key] = ", ".join(parts)
            elif isinstance(value, list):
                data[key] = ", ".join(str(item) for item in value)
        return data


SYSTEM_PROMPT = (
    "You are an expert AI prompt analyzer for text-to-image models (Stable Diffusion).\n"
    "Your job is to analyze a user prompt and extract its semantic components into a structured JSON object.\n\n"
    "Rules:\n"
    "1. You must output a single valid JSON object containing all required fields. Use an empty string if a field is not present or unknown in the user prompt.\n"
    "2. Do not include any explanations, markdown code blocks (like ```json), or extra text outside the JSON.\n"
    "3. Adhere to the following JSON schema:\n"
    "{\n"
    "  \"subject\": \"...\",\n"
    "  \"appearance\": \"...\",\n"
    "  \"action\": \"...\",\n"
    "  \"environment\": \"...\",\n"
    "  \"lighting\": \"...\",\n"
    "  \"camera\": \"...\",\n"
    "  \"style\": \"...\",\n"
    "  \"mood\": \"...\",\n"
    "  \"quality\": \"...\",\n"
    "  \"character_name\": \"\",\n"
    "  \"character_traits\": \"\",\n"
    "  \"pose\": \"\",\n"
    "  \"background\": \"\",\n"
    "  \"color_palette\": \"\"\n"
    "}"
)


def structure_prompt(prompt: str, style: str = "default") -> Optional[StructuredPrompt]:
    """
    Extract semantic components from the user prompt using a local Ollama LLM.

    Args:
        prompt: Original user prompt.
        style: Selected style name.

    Returns:
        StructuredPrompt: Pydantic model containing extracted fields.
        None: If extraction fails, enabling fallback to Phase 1 prompt enhancer.
    """
    logger.info("Prompt received: '%s' (style: '%s')", prompt, style)

    # Load configuration dynamically from environment variables
    url = os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL)
    model = os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    
    timeout_val = os.environ.get("OLLAMA_TIMEOUT")
    if timeout_val is not None:
        try:
            timeout = float(timeout_val)
        except ValueError:
            logger.warning("Invalid OLLAMA_TIMEOUT env var '%s', using default.", timeout_val)
            timeout = DEFAULT_OLLAMA_TIMEOUT
    else:
        timeout = DEFAULT_OLLAMA_TIMEOUT

    # Build prompt for LLM
    user_message = f"User Prompt: {prompt}\nRequested Style: {style}"

    payload = {
        "model": model,
        "prompt": user_message,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "format": "json"
    }

    try:
        logger.info("Connecting to Ollama at %s to extract structured prompt (model: %s, timeout: %.1fs)", url, model, timeout)
        response = requests.post(url, json=payload, timeout=timeout)
        
        if response.status_code != 200:
            logger.error("Ollama server returned error status %d: %s", response.status_code, response.text)
            return None

        response_json = response.json()
        raw_response = response_json.get("response", "").strip()

        if not raw_response:
            logger.warning("Empty response received from Ollama.")
            return None

        # Clean markdown code blocks from response if present
        cleaned_response = raw_response
        if cleaned_response.startswith("```"):
            lines = cleaned_response.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned_response = "\n".join(lines).strip()

        try:
            parsed_data = json.loads(cleaned_response)
            
            # Map style parameter to the structured style field if style is not default and not extracted
            if style != "default" and not parsed_data.get("style"):
                parsed_data["style"] = style

            # Create Pydantic model instance (automatically validates and assigns defaults)
            structured_obj = StructuredPrompt(**parsed_data)
            logger.info("Structured prompt extracted successfully.")
            return structured_obj

        except (json.JSONDecodeError, ValueError) as e:
            logger.error("Failed to parse or validate JSON response from Ollama: %s. Raw response: '%s'", e, raw_response)
            return None

    except requests.exceptions.Timeout:
        logger.error("Timeout occurred while contacting Ollama server at %s", url)
        return None
    except requests.exceptions.RequestException as e:
        logger.error("Ollama server is unavailable: %s", e)
        return None
