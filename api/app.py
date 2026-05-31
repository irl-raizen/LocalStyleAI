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

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="LocalStyleAI",
    description="Generate styled images locally using Stable Diffusion + LoRA adapters.",
    version="1.0.0",
)

# Global state
_pipe = None
_current_lora_style = None
_base_unet_state = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)


# ---------------------------------------------------------------------------
# Pipeline management
# ---------------------------------------------------------------------------

def _get_pipeline():
    """Lazy-load the Stable Diffusion pipeline and cache it."""
    global _pipe, _base_unet_state
    if _pipe is not None:
        return _pipe

    from diffusers import StableDiffusionPipeline
    from src.utils.helpers import BASE_MODELS

    device = get_device()
    dtype = get_dtype(device)

    logger.info("Loading base pipeline...")
    for model_id in BASE_MODELS:
        try:
            logger.info("  Trying: %s ...", model_id)
            _pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
            logger.info("  ✓ Loaded: %s", model_id)
            break
        except Exception as e:
            logger.warning("  ✗ Failed: %s", e)

    if _pipe is None:
        raise RuntimeError("All base models failed to load.")

    _pipe = _pipe.to(device)
    _pipe.safety_checker = None

    if device == "cuda":
        _pipe.enable_attention_slicing()

    # Cache base UNet on CPU for clean LoRA switching
    _base_unet_state = {k: v.cpu().clone() for k, v in _pipe.unet.state_dict().items()}
    logger.info("Pipeline ready on %s.", device)
    return _pipe


def _load_lora(pipeline, style: str):
    """Load the correct LoRA for a style, restoring base weights first."""
    global _current_lora_style, _base_unet_state

    if style == "default" or style not in STYLE_PROMPTS:
        if _current_lora_style is not None:
            pipeline.unet.load_state_dict(_base_unet_state)
            _current_lora_style = None
        return

    if style == _current_lora_style:
        return

    lora_folder = os.path.join(LORA_DIR, style)
    peft_file = os.path.join(lora_folder, "adapter_model.safetensors")

    if not os.path.exists(peft_file) and not os.path.isdir(lora_folder):
        logger.warning("No LoRA weights for '%s'", style)
        return

    try:
        pipeline.unet.load_state_dict(_base_unet_state)
        pipeline.load_lora_weights(lora_folder)
        _current_lora_style = style
        logger.info("LoRA '%s' loaded.", style)
    except Exception as e:
        logger.error("Failed to load LoRA '%s': %s", style, e)
        pipeline.unet.load_state_dict(_base_unet_state)
        _current_lora_style = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

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
        pipeline = _get_pipeline()

        _load_lora(pipeline, style)

        # Phase 2: Structured Prompt Extraction
        structured = structure_prompt(prompt=prompt, style=style)
        
        if structured is not None:
            logger.info("Structured prompt extracted: %s", structured.model_dump())
            full_prompt = compose_prompt(structured)
            logger.info("Composed prompt generated: '%s'", full_prompt)
            negative = STYLE_NEGATIVE_PROMPTS.get(style, DEFAULT_NEGATIVE_PROMPT)
        else:
            # Fall back to Phase 1 enhanced prompt system
            logger.info("Structured prompt extraction failed, falling back to Phase 1 prompt enhancer.")
            enhanced = enhance_prompt(prompt=prompt, style=style)
            
            # Keep original style prefixing fallback logic if Phase 1 also fell back
            if enhanced["enhanced_prompt"] == prompt:
                if style in STYLE_PROMPTS:
                    full_prompt = f"{STYLE_PROMPTS[style]}, {prompt}"
                    negative = STYLE_NEGATIVE_PROMPTS.get(style, "")
                else:
                    full_prompt = prompt
                    negative = enhanced["negative_prompt"]
            else:
                full_prompt = enhanced["enhanced_prompt"]
                negative = enhanced["negative_prompt"]

        logger.info("Generating...")
        logger.info("Generation started")
        result = pipeline(
            prompt=full_prompt,
            negative_prompt=negative or None,
            height=512,
            width=512,
            guidance_scale=7.5,
            num_inference_steps=30,
        )
        img = result.images[0]
        logger.info("Generation completed")

        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        logger.error("Generation failed: %s", e, exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------------------------------------------------------------------------
# Direct execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
