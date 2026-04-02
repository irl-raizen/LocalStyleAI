from diffusers import StableDiffusionPipeline
import torch
import os
from peft import PeftModel

style_name = "anime_clean"
lora_path = f"./loras/{style_name}"
model_id = "Lykon/dreamshaper-8"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == 'cuda' else torch.float32

print(f"Loading model {model_id}...")
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)
pipe.safety_checker = None
if device == 'cuda':
    pipe.enable_attention_slicing()

print(f"Loading LoRA from {lora_path}...")
pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path).to(device)

prompt = "anime style, cel shaded, vibrant colors, clean lines, anime art, a beautiful young girl with flowing silver hair, gazing at the stars, high quality, best quality, detailed, sharp focus"
negative_prompt = "photograph, realistic, 3d render, line art, sketch, ghibli, blurry, low quality, deformed, ugly"

os.makedirs("./exports/audit_test", exist_ok=True)

for i in range(1, 3):
    print(f"Generating sample {i}...")
    image = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=30,
        guidance_scale=7.5,
        height=512,
        width=512,
    ).images[0]
    out_path = f"./exports/audit_test/anime_trained_{i}.png"
    image.save(out_path)
    print(f"Saved to {out_path}")
