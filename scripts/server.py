import sys
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
from io import BytesIO
from PIL import Image
import torch
import os

# Resolve paths relative to this script's location (not CWD)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

app = FastAPI(title='LocalStyleAI')

# --- Style prompt tokens (must match train_lora.py) ---
STYLE_PROMPTS = {
    "anime_clean": "anime style, cel shaded, vibrant colors, clean lines, anime art",
    "ghibli_clean": "studio ghibli style, hand painted, watercolor, soft lighting, miyazaki art",
    "lineart_clean": "line art, ink drawing, black and white, clean outlines, sketch art, no color",
}

STYLE_NEGATIVE_PROMPTS = {
    "anime_clean": "photograph, realistic, 3d render, line art, sketch, ghibli",
    "ghibli_clean": "photograph, realistic, 3d render, anime, cel shaded, line art, sketch",
    "lineart_clean": "photograph, realistic, color, painting, anime, cel shaded, ghibli, vibrant",
}


@app.get("/", response_class=HTMLResponse)
async def read_root():
    # Try script-relative path first, then project-relative
    path = os.path.join(SCRIPT_DIR, 'index.html')
    if not os.path.exists(path):
        path = os.path.join(PROJECT_DIR, 'scripts', 'index.html')
    if not os.path.exists(path):
        return HTMLResponse(content="index.html not found", status_code=404)

    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


# --- Configuration ---
LORA_DIR = os.path.join(PROJECT_DIR, 'loras')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D_TYPE = torch.float16 if DEVICE == 'cuda' else torch.float32

# Global state
pipe = None
current_lora_style = None  # Track which LoRA is currently loaded
base_unet_state = None     # Cache base UNet for clean LoRA switching


def get_pipeline():
    """Load the base Stable Diffusion pipeline once and cache it."""
    global pipe, base_unet_state
    if pipe is not None:
        return pipe

    print('Loading base pipeline...')
    from diffusers import StableDiffusionPipeline

    models_to_try = [
        "Lykon/dreamshaper-8",
        "SG161222/Realistic_Vision_V5.1_noVAE",
        "runwayml/stable-diffusion-v1-5",
    ]

    for model_id in models_to_try:
        try:
            print(f"  Trying: {model_id}...")
            pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=D_TYPE)
            print(f"  Loaded: {model_id}")
            break
        except Exception as e:
            print(f"  Failed: {e}")

    if pipe is None:
        print("ERROR: All models failed to load.")
        return None

    pipe = pipe.to(DEVICE)
    pipe.safety_checker = None

    if DEVICE == 'cuda':
        pipe.enable_attention_slicing()
        print("Attention slicing enabled.")

    # Cache the base UNet state dict for clean LoRA switching
    # IMPORTANT: Keep this on CPU to save VRAM on 4GB cards
    base_unet_state = {k: v.cpu().clone() for k, v in pipe.unet.state_dict().items()}
    print(f"Base UNet state cached to CPU ({len(base_unet_state)} tensors).")

    print(f"Pipeline ready on {DEVICE}.")
    return pipe


def load_lora_for_style(pipeline, style):
    """Load the correct LoRA for the requested style, unloading any previous one."""
    global current_lora_style, base_unet_state

    if style == 'default' or style not in STYLE_PROMPTS:
        # Restore base model (no LoRA)
        if current_lora_style is not None:
            print(f"Unloading LoRA '{current_lora_style}', restoring base model...")
            pipeline.unet.load_state_dict(base_unet_state)
            current_lora_style = None
        return

    if style == current_lora_style:
        print(f"LoRA '{style}' already loaded.")
        return

    # Path check: try directory/folder first, then look for adapter_model.safetensors inside
    lora_folder = os.path.join(LORA_DIR, style)
    lora_file = os.path.join(LORA_DIR, f"{style}.safetensors")
    
    # PEFT format uses adapter_model.safetensors inside a folder
    peft_file = os.path.join(lora_folder, "adapter_model.safetensors")
    
    target_path = None
    if os.path.exists(peft_file):
        target_path = lora_folder # load_lora_weights handles the folder if it has the file
    elif os.path.exists(lora_file):
        target_path = lora_file
    elif os.path.exists(lora_folder):
        target_path = lora_folder

    if not target_path:
        print(f"Warning: No LoRA weights found for style '{style}' at {lora_folder} or {lora_file}")
        return

    try:
        # Step 1: Restore base UNet (clean slate) - ensures we don't 'stack' LoRAs
        print(f"Restoring base UNet before loading '{style}'...")
        pipeline.unet.load_state_dict(base_unet_state)

        # Step 2: Load the new LoRA weights natively
        print(f"Loading LoRA weights from {target_path}...")
        # Native load_lora_weights is more VRAM efficient and stable than PeftModel wrapper
        pipeline.load_lora_weights(target_path)
        
        current_lora_style = style
        print(f"LoRA '{style}' loaded successfully.")
    except Exception as e:
        print(f"Warning: Failed to load LoRA '{style}': {e}")
        import traceback
        traceback.print_exc()
        # Restore base model on failure
        pipeline.unet.load_state_dict(base_unet_state)
        current_lora_style = None


@app.post('/generate')
async def generate(prompt: str = Form(...), style: str = Form('default')):
    try:
        print(f"Request: prompt='{prompt[:50]}...' style='{style}'")
        pipeline = get_pipeline()

        if pipeline is None:
            return HTMLResponse(content="Pipeline failed to load", status_code=500)

        # Load the correct LoRA (with clean unload/reload)
        load_lora_for_style(pipeline, style)

        # Build the full prompt with style tokens
        if style in STYLE_PROMPTS:
            style_prefix = STYLE_PROMPTS[style]
            full_prompt = f"{style_prefix}, {prompt}"
            negative_prompt = STYLE_NEGATIVE_PROMPTS.get(style, "")
            print(f"Full prompt: '{full_prompt[:80]}...'")
            print(f"Negative: '{negative_prompt}'")
        else:
            full_prompt = prompt
            negative_prompt = "blurry, low quality, deformed, ugly"

        print("Generating image...")
        out = pipeline(
            prompt=full_prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            height=512,
            width=512,
            guidance_scale=7.5,
            num_inference_steps=30,
        )
        img = out.images[0]
        print("Generation complete.")

        buf = BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        return StreamingResponse(buf, media_type='image/png')

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return HTMLResponse(content=f"Error: {e}", status_code=500)


# Run with: uvicorn scripts.server:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
