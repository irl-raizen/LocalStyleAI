"""
LocalStyleAI - Structured Prompt Composer

Converts a structured prompt representation into a highly optimized,
comma-separated plain-text prompt suitable for Stable Diffusion.
"""

from typing import Union
from src.ai.prompt_structurer import StructuredPrompt
from src.utils.helpers import get_logger

logger = get_logger("prompt_composer")


def compose_prompt(structured: Union[StructuredPrompt, dict]) -> str:
    """
    Convert a StructuredPrompt model or dictionary into an optimized diffusion prompt.

    Args:
        structured: A StructuredPrompt object or dictionary containing prompt fields.

    Returns:
        str: Composed and optimized plain-text prompt.
    """
    if isinstance(structured, dict):
        structured = StructuredPrompt(**structured)

    parts = []

    # 1. Subject description (merging subject, appearance and action if available)
    subject_desc = ""
    if structured.subject:
        subject_desc = structured.subject
        if structured.appearance:
            # Avoid duplicating "with"
            if not structured.appearance.lower().startswith("with "):
                subject_desc += f" with {structured.appearance}"
            else:
                subject_desc += f" {structured.appearance}"
        if structured.action:
            subject_desc += f" {structured.action}"
    else:
        # Fallbacks if subject is empty
        if structured.appearance:
            subject_desc = structured.appearance
            if structured.action:
                subject_desc += f" {structured.action}"
        elif structured.action:
            subject_desc = structured.action

    if subject_desc:
        parts.append(subject_desc)

    # 2. Environment (location, setting, background)
    if structured.environment:
        parts.append(structured.environment)
    elif structured.background:
        parts.append(structured.background)

    # 3. Lighting conditions
    if structured.lighting:
        light_val = structured.lighting
        # Normalize: append "lighting" if not present
        if "light" not in light_val.lower():
            light_val = f"{light_val} lighting"
        parts.append(light_val)

    # 4. Camera settings (composition, angle, shot type)
    if structured.camera:
        parts.append(structured.camera)

    # 5. Mood / Atmosphere
    if structured.mood:
        parts.append(structured.mood)

    # 6. Aesthetic style
    if structured.style:
        style_val = structured.style.lower()
        if style_val == "anime_clean":
            parts.append("anime style, cel shaded, vibrant colors, clean lines, anime art")
        elif style_val == "ghibli_clean":
            parts.append("studio ghibli style, hand painted, watercolor, soft lighting, miyazaki art")
        elif style_val == "lineart_clean":
            parts.append("line art, ink drawing, black and white, clean outlines, sketch art, no color")
        else:
            parts.append(structured.style)

    # 7. Quality indicators
    if structured.quality:
        parts.append(structured.quality)
    else:
        parts.append("masterpiece quality, highly detailed")

    # Clean and filter all parts
    cleaned_parts = [p.strip() for p in parts if p.strip()]
    composed_prompt = ", ".join(cleaned_parts)

    logger.info("Composed prompt generated: '%s'", composed_prompt)
    return composed_prompt
