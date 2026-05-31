import os
import json
import requests
from typing import Optional
from pydantic import BaseModel, Field
from src.utils.helpers import get_logger

logger = get_logger("edit_interpreter")

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_OLLAMA_MODEL = "qwen3:8b"

class EditPlan(BaseModel):
    action: str = Field(..., description="Type of action: modify, replace, add, remove, environment")
    target: str = Field(default="", description="The object, character, or region to edit")
    attribute: str = Field(default="", description="The attribute being changed (e.g., color, size)")
    value: str = Field(default="", description="The new value or object (e.g., larger, axe, red)")

SYSTEM_PROMPT = """
You are an image editing interpreter.
Convert the user's natural language edit instruction into a structured JSON edit plan.

Actions:
- "modify": Change an attribute (e.g. "Make dragon larger", "Change jacket to blue")
- "replace": Swap an object (e.g. "Replace sword with axe")
- "add": Add a new object/effect (e.g. "Add snowfall", "Add a moon")
- "remove": Delete an object (e.g. "Remove tree")
- "environment": Change the overall scene (e.g. "Change sunset to night")

Output ONLY valid JSON matching this schema:
{
    "action": "...",
    "target": "...",
    "attribute": "...",
    "value": "..."
}
"""

def interpret_edit(instruction: str) -> Optional[EditPlan]:
    url = os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL)
    model = os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    timeout = float(os.environ.get("OLLAMA_TIMEOUT", 10.0))

    payload = {
        "model": model,
        "prompt": f"Instruction: {instruction}",
        "system": SYSTEM_PROMPT,
        "stream": False,
        "format": "json"
    }

    logger.info("Interpreting edit instruction: '%s'", instruction)

    try:
        response = requests.post(url, json=payload, timeout=timeout)
        if response.status_code != 200:
            return None

        raw = response.json().get("response", "").strip()
        if not raw: return None
        
        # Clean markdown if present
        if raw.startswith("```"):
            lines = raw.splitlines()
            if lines[0].startswith("```"): lines = lines[1:]
            if lines and lines[-1].startswith("```"): lines = lines[:-1]
            raw = "\n".join(lines).strip()

        data = json.loads(raw)
        plan = EditPlan(**data)
        logger.info("Edit interpreted: %s", plan.model_dump())
        return plan

    except Exception as e:
        logger.error("Failed to interpret edit: %s", e)
        return None
