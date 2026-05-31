"""
LocalStyleAI — FastAPI Server

Provides a REST API and web UI for generating styled images.

Run with:
    uvicorn api.app:app --host 0.0.0.0 --port 8000
"""

import os
import sys
from io import BytesIO

import torch
from fastapi import FastAPI, Form
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse

from src.utils.helpers import (
    STYLE_PROMPTS,
    STYLE_NEGATIVE_PROMPTS,
    AVAILABLE_STYLES,
    LORA_DIR,
    get_device,
    get_dtype,
    get_logger,
)
from src.ai.prompt_enhancer import enhance_prompt, DEFAULT_NEGATIVE_PROMPT
from src.ai.prompt_structurer import structure_prompt
from src.ai.prompt_composer import compose_prompt

logger = get_logger("api")

from src.models.model_router import ModelRouter
from src.memory.memory_engine import MemoryEngine

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="LocalStyleAI",
    description="Generate styled images locally using multiple backends (SD1.5, SDXL, FLUX).",
    version="1.0.0",
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)



# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/debug/model")
async def debug_model():
    """Benchmark endpoint to inspect the active model."""
    try:
        router = ModelRouter()
        backend = router.get_active_backend()
        return JSONResponse({
            "active_model": router.active_model_name,
            "loaded": backend.is_loaded,
            "device": backend.device,
            "vram_optimized": True if backend.device == "cuda" else False
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the web UI, or return a health-check JSON."""
    # Try to find index.html in api/ directory or scripts/ directory
    for candidate in [
        os.path.join(SCRIPT_DIR, "static", "index.html"),
        os.path.join(PROJECT_DIR, "api", "static", "index.html"),
    ]:
        if os.path.exists(candidate):
            with open(candidate, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())

    return HTMLResponse(content="<h1>LocalStyleAI API is running</h1><p>Web UI not found.</p>")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return JSONResponse({"status": "ok", "gpu": torch.cuda.is_available()})


@app.get("/memory", response_class=HTMLResponse)
async def memory_ui():
    """Serve the Memory Manager UI."""
    for candidate in [
        os.path.join(SCRIPT_DIR, "static", "memory.html"),
        os.path.join(PROJECT_DIR, "api", "static", "memory.html"),
    ]:
        if os.path.exists(candidate):
            with open(candidate, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Memory UI not found.</h1>")

@app.get("/styles")
async def styles():
    """List available styles."""
    return JSONResponse({"styles": AVAILABLE_STYLES})


@app.get("/debug/prompt")
async def debug_prompt(prompt: str, style: str = "default"):
    """Debug endpoint to inspect structured prompt extraction and composition."""
    logger.info("Prompt received: '%s' (style: '%s')", prompt, style)
    
    structured = structure_prompt(prompt=prompt, style=style)
    if structured is not None:
        logger.info("Structured prompt extracted: %s", structured.model_dump())
        composed = compose_prompt(structured)
        logger.info("Composed prompt generated: '%s'", composed)
        negative = STYLE_NEGATIVE_PROMPTS.get(style, DEFAULT_NEGATIVE_PROMPT)
        return JSONResponse({
            "structured": structured.model_dump(),
            "composed_prompt": composed,
            "negative_prompt": negative
        })
    else:
        logger.info("Structured prompt extraction failed, falling back to Phase 1 prompt enhancer.")
        enhanced = enhance_prompt(prompt=prompt, style=style)
        logger.info("Composed prompt generated: '%s'", enhanced["enhanced_prompt"])
        return JSONResponse({
            "structured": {},
            "composed_prompt": enhanced["enhanced_prompt"],
            "negative_prompt": enhanced["negative_prompt"]
        })


@app.post("/generate")
async def generate(prompt: str = Form(...), style: str = Form("default")):
    """Generate an image from a text prompt with an optional style."""
    try:
        logger.info("Request — prompt='%s...' style='%s'", prompt[:50], style)
        from src.inference.generate import generate_image
        
        img = generate_image(
            prompt=prompt,
            style=style,
            steps=None,
            width=None,
            height=None,
            guidance_scale=None
        )

        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        logger.error("Generation failed: %s", e, exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------------------------------------------------------------------------
# Memory Endpoints
# ---------------------------------------------------------------------------

@app.get("/memory/characters")
async def get_characters():
    """Get all stored characters."""
    engine = MemoryEngine()
    return JSONResponse({"characters": engine.get_all_characters()})

@app.get("/memory/characters/{name}")
async def get_character(name: str):
    """Get a specific stored character."""
    engine = MemoryEngine()
    char = engine.get_character(name)
    if char:
        return JSONResponse(char.model_dump())
    return JSONResponse({"error": "Character not found"}, status_code=404)

@app.delete("/memory/characters/{name}")
async def delete_character(name: str):
    """Delete a stored character."""
    engine = MemoryEngine()
    success = engine.delete_character(name)
    if success:
        return JSONResponse({"status": "success", "message": f"Deleted {name}"})
    return JSONResponse({"error": "Character not found"}, status_code=404)

@app.get("/memory/styles")
async def get_styles():
    """Get stored styles."""
    engine = MemoryEngine()
    style = engine.get_style()
    if style:
        return JSONResponse(style.model_dump())
    return JSONResponse({"preferred_style": "default"})

@app.get("/memory/scenes")
async def get_scenes():
    """Get all stored scenes."""
    engine = MemoryEngine()
    return JSONResponse({"scenes": engine.get_all_scenes()})

# ---------------------------------------------------------------------------
# Direct execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
