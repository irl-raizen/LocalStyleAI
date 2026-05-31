from typing import Optional
from pydantic import BaseModel, Field

class StyleMemory(BaseModel):
    preferred_style: str = Field(default="default", description="Preferred image generation style")
    
    # Can be extended with preferred aspect ratios, colors, etc.
