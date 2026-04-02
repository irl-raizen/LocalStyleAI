from diffusers import StableDiffusionPipeline
import torch
import os

# --- Style prompt tokens (must match train_lora.py and server.py) ---
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

TEST_SUBJECTS = {
    "anime_clean": "a beautiful young girl with flowing silver hair, gazing at the stars",
    "ghibli_clean": "a cozy magical cottage hidden in a lush green forest",
    "lineart_clean": "a ferocious dragon perched on a mountain",
}

LORA_DIR = "./loras"

try:
    print("Loading pipeline...")
    models_to_try = ["Lykon/dreamshaper-8", "SG161222/Realistic_Vision_V5.1_noVAE", "runwayml/stable-diffusion-v1-5"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d_type = torch.float16 if device == 'cuda' else torch.float32

    pipe = None
    for model_id in models_to_try:
        try:
            print(f"Trying to load model: {model_id}...")
            pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=d_type)
            print(f"Successfully loaded {model_id}")
            break
        except Exception as e:
            print(f"Failed to load {model_id}: {e}")

    if pipe is None:
        raise Exception("All models failed to load.")

    assert pipe is not None
    pipe = pipe.to(device)
    pipe.safety_checker = None
    if device == 'cuda':
        pipe.enable_attention_slicing()

    # Cache base UNet state for clean LoRA switching
    base_unet_state = {k: v.clone() for k, v in pipe.unet.state_dict().items()}

    print(f"Pipeline loaded on {device}. Starting style tests...")
    os.makedirs('./exports', exist_ok=True)

    for style_name in ["anime_clean", "ghibli_clean", "lineart_clean"]:
        print(f"\n{'='*60}")
        print(f"Testing style: {style_name}")
        print(f"{'='*60}")

        # Restore base UNet (clean slate)
        pipe.unet.load_state_dict(base_unet_state)

        # Load LoRA for this style
        lora_path = os.path.join(LORA_DIR, style_name)
        if os.path.exists(lora_path):
            try:
                from peft import PeftModel
                pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
                pipe.unet = pipe.unet.to(device)
                print(f"LoRA '{style_name}' loaded from {lora_path}")
            except Exception as e:
                print(f"Warning: Failed to load LoRA: {e}")
        else:
            print(f"Warning: No LoRA found at {lora_path}, using base model only")

        # Build prompt
        style_prefix = STYLE_PROMPTS[style_name]
        subject = TEST_SUBJECTS[style_name]
        full_prompt = f"{style_prefix}, {subject}, high quality, best quality, detailed, sharp focus"
        negative_prompt = STYLE_NEGATIVE_PROMPTS[style_name]

        print(f"Prompt: {full_prompt}")
        print(f"Negative: {negative_prompt}")

        for img_idx in range(1, 4):  # 3 test images per style
            print(f"Generating {style_name} image {img_idx}/3...")
            image = pipe(
                full_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=30,
                guidance_scale=7.5,
                height=512,
                width=512,
            ).images[0]
            out_path = f"./exports/test_{style_name}_{img_idx}.png"
            image.save(out_path)
            print(f"Saved: {out_path}")

    print("\n" + "="*60)
    print("All style tests complete! Check ./exports/ for results.")
    print("="*60)

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
