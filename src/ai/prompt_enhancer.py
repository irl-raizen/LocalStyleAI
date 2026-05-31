"""
LocalStyleAI - AI Prompt Intelligence Layer

Connects to a local Ollama instance to enhance image prompts and generate
style-appropriate negative prompts.
"""

import os
import json
import requests
from src.utils.helpers import get_logger

logger = get_logger("prompt_enhancer")

# Reusable constant for default negative prompt
DEFAULT_NEGATIVE_PROMPT = (
    "blurry, low quality, worst quality, bad anatomy, extra fingers, "
    "extra limbs, deformed face, cropped, watermark, text, logo"
)

# Configuration defaults
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_OLLAMA_MODEL = "qwen3:8b"
DEFAULT_OLLAMA_TIMEOUT = 10.0

# System prompt instructing the LLM on how to enhance prompts
SYSTEM_PROMPT = (
    "You are an expert AI prompt engineer for image generation models (like Stable Diffusion).\n"
    "Your task is to take a vague user prompt and expand it into a highly detailed image prompt, "
    "and generate a corresponding negative prompt.\n\n"
    "Follow these rules strictly:\n"
    "1. Expand vague prompts to add rich visual details, lighting, environment, composition, and camera information.\n"
    "2. Preserve the original user intent.\n"
    "3. Respect and adapt the prompt based on the requested style (e.g., anime_clean, ghibli_clean, lineart_clean).\n"
    "4. Output MUST be valid JSON only. Do not output markdown, do not write code blocks (like ```json), do not write explanations.\n"
    "5. The JSON output must have exactly two keys:\n"
    "   - \"enhanced_prompt\": the detailed prompt string.\n"
    "   - \"negative_prompt\": the comma-separated negative prompt string.\n\n"
    "Required JSON Format:\n"
    "{\n"
    "  \"enhanced_prompt\": \"...\",\n"
    "  \"negative_prompt\": \"...\"\n"
    "}"
)


def enhance_prompt(prompt: str, style: str = "default") -> dict:
    """
    Enhance user prompt using a local Ollama LLM.

    Args:
        prompt: Original user prompt.
        style: Selected style name.

    Returns:
        dict: A dictionary containing "enhanced_prompt" and "negative_prompt".
              Falls back to the original prompt if any error occurs.
    """
    logger.info("Prompt received: '%s' (style: '%s')", prompt, style)

    # Prepare fallback behavior
    fallback = {
        "enhanced_prompt": prompt,
        "negative_prompt": DEFAULT_NEGATIVE_PROMPT
    }

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

    # Build prompt for the LLM
    user_message = f"User Prompt: {prompt}\nRequested Style: {style}"

    payload = {
        "model": model,
        "prompt": user_message,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "format": "json"
    }

    try:
        logger.info("Connecting to Ollama at %s (model: %s, timeout: %.1fs)", url, model, timeout)
        response = requests.post(url, json=payload, timeout=timeout)
        
        if response.status_code != 200:
            logger.error("Ollama server returned error status %d: %s", response.status_code, response.text)
            return fallback

        response_json = response.json()
        raw_response = response_json.get("response", "").strip()

        if not raw_response:
            logger.warning("Empty response received from Ollama.")
            return fallback

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
            enhanced_prompt = parsed_data.get("enhanced_prompt")
            negative_prompt = parsed_data.get("negative_prompt")

            if not enhanced_prompt:
                logger.warning("JSON parsed successfully but 'enhanced_prompt' was missing or empty.")
                return fallback

            if not negative_prompt:
                negative_prompt = DEFAULT_NEGATIVE_PROMPT

            logger.info("Enhanced prompt generated: '%s'", enhanced_prompt)
            return {
                "enhanced_prompt": enhanced_prompt,
                "negative_prompt": negative_prompt
            }

        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON response from Ollama: %s. Raw response: '%s'", e, raw_response)
            return fallback

    except requests.exceptions.Timeout:
        logger.error("Timeout occurred while contacting Ollama server at %s", url)
        return fallback
    except requests.exceptions.RequestException as e:
        logger.error("Ollama server is unavailable: %s", e)
        return fallback