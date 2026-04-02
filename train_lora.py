import os
import torch
import json
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import numpy as np
from diffusers import StableDiffusionPipeline, DDPMScheduler

# --- Style prompt tokens for conditioned LoRA training ---
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


def parse_args():
    parser = argparse.ArgumentParser(description="Train LoRA for a specific style")
    parser.add_argument('--style', type=str, required=True, help="Style folder name (e.g. anime_clean)")
    parser.add_argument('--steps', type=int, default=1200, help="Number of training steps")
    parser.add_argument('--lr', type=float, default=2e-5, help="Learning rate")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size")
    parser.add_argument('--resolution', type=int, default=512, help="Image resolution")
    return parser.parse_args()


class StyleDataset(Dataset):
    """Dataset that loads images for a specific style from the manifest."""
    def __init__(self, manifest_path, target_style, resolution=512):
        self.entries = []
        self.resolution = resolution
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    if entry['style'] == target_style:
                        self.entries.append(entry)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        img = Image.open(entry['file']).convert('RGB').resize(
            (self.resolution, self.resolution), Image.LANCZOS
        )
        # Normalize to [-1, 1] for VAE
        arr = np.array(img).astype(np.float32) / 255.0
        arr = arr * 2.0 - 1.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # [C, H, W]

        # Simple data augmentation: random horizontal flip
        if np.random.rand() > 0.5:
            tensor = torch.flip(tensor, dims=[2])

        return tensor


def train():
    args = parse_args()
    STYLE = args.style
    STEPS = args.steps
    LR = args.lr
    BATCH_SIZE = args.batch_size
    RESOLUTION = args.resolution

    MANIFEST = Path('./exports/manifest.jsonl')
    OUT_DIR = Path('./loras')
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    D_TYPE = torch.float16 if DEVICE == 'cuda' else torch.float32

    # Validate style
    if STYLE not in STYLE_PROMPTS:
        print(f"ERROR: Unknown style '{STYLE}'. Available: {list(STYLE_PROMPTS.keys())}")
        return

    print(f"=" * 60)
    print(f"LoRA Training — Style: {STYLE}")
    print(f"Style prompt: {STYLE_PROMPTS[STYLE]}")
    print(f"Device: {DEVICE} | dtype: {D_TYPE} | Steps: {STEPS} | LR: {LR}")
    print(f"=" * 60)

    # --- Load dataset ---
    if not MANIFEST.exists():
        print("ERROR: Manifest not found. Run dataset_prep.py first.")
        return

    ds = StyleDataset(MANIFEST, STYLE, RESOLUTION)
    if len(ds) == 0:
        print(f"ERROR: No images found for style '{STYLE}' in manifest!")
        return
    print(f"Dataset loaded: {len(ds)} images for style '{STYLE}'")

    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)

    # --- Load pipeline components ---
    print("Loading base model...")
    models_to_try = [
        "Lykon/dreamshaper-8",
        "SG161222/Realistic_Vision_V5.1_noVAE",
        "runwayml/stable-diffusion-v1-5",
    ]

    pipe = None
    loaded_model_id = None
    for model_id in models_to_try:
        try:
            print(f"  Trying: {model_id}...")
            pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=D_TYPE)
            loaded_model_id = model_id
            print(f"  Loaded: {model_id}")
            break
        except Exception as e:
            print(f"  Failed: {e}")

    if pipe is None:
        print("ERROR: All models failed to load.")
        return

    # Apply attention slicing on pipeline BEFORE extracting components
    if DEVICE == 'cuda':
        pipe.enable_attention_slicing()
        print("Attention slicing enabled on pipeline.")

    # Extract components
    vae = pipe.vae.to(DEVICE, dtype=torch.float32) # FORCE fp32 for VAE (prevents NaNs)
    unet = pipe.unet.to(DEVICE)
    text_encoder = pipe.text_encoder.to(DEVICE, dtype=torch.float32) # FORCE fp32
    tokenizer = pipe.tokenizer
    noise_scheduler = DDPMScheduler.from_pretrained(
        loaded_model_id,
        subfolder="scheduler"
    )

    # Freeze VAE and text encoder
    vae.requires_grad_(False)
    vae.eval()
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    # Free the pipeline from GPU — only keep extracted components
    # This frees ~500MB of VRAM on RTX 3050
    del pipe
    torch.cuda.empty_cache()

    # --- Encode style prompt ONCE (used for all training steps) ---
    style_prompt = STYLE_PROMPTS[STYLE]
    print(f"Encoding style prompt: '{style_prompt}'")
    with torch.no_grad():
        text_input = tokenizer(
            style_prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        style_text_embeddings = text_encoder(text_input.input_ids.to(DEVICE))[0]
        # Cast to model dtype (D_TYPE) to avoid mismatch while keeping generation stable
        style_text_embeddings = style_text_embeddings.to(dtype=D_TYPE)
    print(f"Style embedding shape: {style_text_embeddings.shape}")

    # --- Setup LoRA on UNet ---
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["to_q", "to_v", "to_k", "to_out.0"],
        lora_dropout=0.05,
    )
    unet = get_peft_model(unet, lora_config)
    unet.train()

    # GPU optimizations
    if DEVICE == 'cuda':
        torch.backends.cudnn.benchmark = True

    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in unet.parameters())
    print(f"Trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    # Cast ONLY the LoRA trainable parameters to fp32 for stable gradient computation.
    # Non-trainable UNet weights stay in fp16 to save VRAM.
    for name, param in unet.named_parameters():
        if param.requires_grad:
            param.data = param.data.float()

    print(f"LoRA params cast to fp32. Non-trainable UNet stays fp16.")

    torch.cuda.empty_cache()
    print(f"VRAM freed before training loop.")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=2e-5, # Lower LR for better stability on small datasets
        weight_decay=1e-2,
    )

    # --- Training loop with CONDITIONED diffusion loss ---
    print(f"\nStarting training for {STEPS} steps...")
    pbar = tqdm(total=STEPS, desc=f"Training {STYLE}")
    step = 0
    running_loss = 0.0
    nan_count = 0
    done = False

    while not done:
        for batch in dl:
            if done:
                break

            pixel_values = batch.to(DEVICE, dtype=torch.float32) # VAE needs fp32

            # 1. Encode images to latent space via VAE
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                latents = latents.to(dtype=torch.float32) # Keep latents in fp32 for stability

            # 2. Sample random noise (fp32)
            noise = torch.randn_like(latents)

            # 3. Sample random timesteps
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=DEVICE
            ).long()

            # 4. Add noise to latents (forward diffusion)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # 5. Predict noise with UNet using STYLE-CONDITIONED text embeddings (autocast)
            encoder_hidden_states = style_text_embeddings.expand(
                latents.shape[0], -1, -1
            )

            optimizer.zero_grad()

            with torch.autocast("cuda"):
                noise_pred = unet(
                    noisy_latents.to(dtype=D_TYPE),
                    timesteps,
                    encoder_hidden_states,
                ).sample

                # 6. Compute MSE loss
                loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float())

            if torch.isnan(loss):
                continue

            # 7. Backprop
            loss.backward()

            # 8. Gradient clipping
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)

            # 9. Optimizer step
            optimizer.step()

            running_loss += loss.item()
            step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{running_loss/step:.4f}")
            pbar.update(1)

            if step >= STEPS:
                done = True
                break

    pbar.close()
    print(f"\nTraining complete. Average loss: {running_loss/step:.4f}")

    # --- Save LoRA weights ---
    save_path = OUT_DIR / STYLE
    save_path.mkdir(parents=True, exist_ok=True)

    # Save using PEFT's save method (compatible with load_attn_procs)
    unet.save_pretrained(str(save_path))
    print(f"LoRA weights saved to: {save_path}")

    # Cleanup GPU memory
    del unet, vae, text_encoder, optimizer
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()

    print(f"Done training '{STYLE}'!")


if __name__ == "__main__":
    train()
