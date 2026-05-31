import os
import json
import requests
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, model_validator
from src.utils.helpers import get_logger

logger = get_logger("character_extractor")

# Configuration defaults
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_OLLAMA_MODEL = "qwen3:8b"
DEFAULT_OLLAMA_TIMEOUT = 10.0

class CharacterMemory(BaseModel):
    name: str = Field(default="", description="Name of the character")
    age: str = Field(default="", description="Age or age range")
    hair: str = Field(default="", description="Hair color and style")
    eyes: str = Field(default="", description="Eye color")
    clothing: str = Field(default="", description="Default clothing or outfit")
    traits: List[str] = Field(default_factory=list, description="Personality traits or other descriptors")

    @model_validator(mode='before')
    @classmethod
    def normalize_fields(cls, data):
        if not isinstance(data, dict):
            return data
        
        # Ensure traits is a list of strings
        traits = data.get("traits", [])
        if isinstance(traits, str):
            data["traits"] = [t.strip() for t in traits.split(",") if t.strip()]
        elif isinstance(traits, dict):
            data["traits"] = [f"{k}: {v}" for k, v in traits.items()]
            
        # Ensure string fields are strings
        for field in ["name", "age", "hair", "eyes", "clothing"]:
            val = data.get(field, "")
            if isinstance(val, (dict, list)):
                data[field] = str(val)
            elif val is None:
                data[field] = ""
                
        return data

SYSTEM_PROMPT = (
    "You are an AI character memory extractor. "
    "Your job is to analyze a user's image generation prompt and determine if they are describing a character. "
    "If a character with clear physical attributes (like name, age, hair, eyes, clothing, traits) is described, extract them into a JSON object. "
    "Rules:\n"
    "1. Output a valid JSON object ONLY. No extra text, no markdown.\n"
    "2. If no clear character is described, output an empty JSON object: {}\n"
    "3. Use the following schema if a character is found:\n"
    "{\n"
    "  \"name\": \"...\",\n"
    "  \"age\": \"...\",\n"
    "  \"hair\": \"...\",\n"
    "  \"eyes\": \"...\",\n"
    "  \"clothing\": \"...\",\n"
    "  \"traits\": [\"...\", \"...\"]\n"
    "}"
)

def extract_character(prompt: str) -> Optional[CharacterMemory]:
    """
    Extract character information from a prompt using Ollama.
    """
    url = os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL)
    model = os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    
    timeout_val = os.environ.get("OLLAMA_TIMEOUT")
    timeout = float(timeout_val) if timeout_val is not None else DEFAULT_OLLAMA_TIMEOUT

    payload = {
        "model": model,
        "prompt": f"User Prompt: {prompt}",
        "system": SYSTEM_PROMPT,
        "stream": False,
        "format": "json"
    }

    try:
        response = requests.post(url, json=payload, timeout=timeout)
        if response.status_code != 200:
            return None

        response_json = response.json()
        raw_response = response_json.get("response", "").strip()

        if not raw_response:
            return None

        # Clean markdown code blocks from response if present
        cleaned = raw_response
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines[0].startswith("```"): lines = lines[1:]
            if lines and lines[-1].startswith("```"): lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        parsed_data = json.loads(cleaned)
        
        if not parsed_data or not parsed_data.get("name"):
            # Empty JSON means no character detected
            return None

        char_obj = CharacterMemory(**parsed_data)
        logger.info("Character detected: %s", char_obj.name)
        return char_obj

    except Exception as e:
        logger.error("Failed to extract character: %s", e)
        return None
