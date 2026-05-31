from typing import Optional
from pydantic import BaseModel, Field

class SceneMemory(BaseModel):
    scene_name: str = Field(..., description="Unique name of the scene")
    environment: str = Field(default="", description="The location or environment details")
    lighting: str = Field(default="", description="Lighting conditions of the scene")
    time_of_day: str = Field(default="", description="Time of day")

def format_scene_for_prompt(scene: SceneMemory) -> str:
    """Format scene attributes into a prompt string."""
    parts = []
    if scene.environment:
        parts.append(scene.environment)
    if scene.lighting:
        parts.append(f"{scene.lighting} lighting")
    if scene.time_of_day:
        parts.append(scene.time_of_day)
    return ", ".join(parts)
