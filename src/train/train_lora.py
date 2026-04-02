"""
LocalStyleAI - LoRA Training Module

Trains style-conditioned LoRA adapters on top of a Stable Diffusion base model.
Optimized for low-VRAM GPUs (RTX 3050 4 GB+).
"""

import os
import sys
import json
import argparse

import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionPipeline, DDPMScheduler

from src.utils.helpers import (
    STYLE_PROMPTS,
    STYLE_NEGATIVE_PROMPTS,
    LORA_DIR,
    MANIFEST_PATH,
    BASE_MODELS,
    get_device,
    get_dtype,
    get_logger,
)

logger = get_logger("train")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class StyleDataset(Dataset):
    """Loads images for a specific style from the JSONL manifest."""

    def __init__(self, manifest_path: str, target_style: str, resolution: int = 512):
        self.entries: list[dict] = []
        self.resolution = resolution

        manifest = Path(manifest_path)
        if manifest.exists():
            with open(manifest, "r") as f:
                for line in f:
                    entry = json.loads(line)
                    if entry["style"] == target_style:
                        self.entries.append(entry)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> torch.Tensor:
        entry = self.entries[idx]
        img = Image.open(entry["file"]).convert("RGB").resize(
            (self.resolution, self.resolution), Image.LANCZOS
        )
        # Normalize to [-1, 1] for VAE
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = arr * 2.0 - 1.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # [C, H, W]

        # Random horizontal flip augmentation
        if np.random.rand() > 0.5:
            tensor = torch.flip(tensor, dims=[2])

        return tensor


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse training CLI arguments."""
    parser = argparse.ArgumentParser(description="Train a style LoRA adapter")
    parser.add_argument("--style", type=str, required=True,
                        help=f"Style name — one of {list(STYLE_PROMPTS.keys())}")
    parser.add_argument("--steps", type=int, default=1200,
                        help="Total training steps (default: 1200)")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate (default: 2e-5)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (default: 1)")
    parser.add_argument("--resolution", type=int, default=512,
                        help="Training image resolution (default: 512)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace | None = None):
    """
    Main training entry point.

    Loads a base SD model, attaches LoRA adapters to the UNet,
    and trains with the conditioned diffusion loss.
    """
    if args is None:
        args = parse_args()

    style = args.style
    steps = args.steps
    lr = args.lr
    batch_size = args.batch_size
    resolution = args.resolution

    device = get_device()
    dtype = get_dtype(device)

    # Validate style
    if style not in STYLE_PROMPTS:
        logger.error("Unknown style '%s'. Available: %s", style, list(STYLE_PROMPTS.keys()))
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("LoRA Training — Style: %s", style)
    logger.info("Prompt: %s", STYLE_PROMPTS[style])
    logger.info("Device: %s | dtype: %s | Steps: %d | LR: %.2e", device, dtype, steps, lr)
    logger.info("=" * 60)

    # --- Dataset ---
    if not os.path.exists(MANIFEST_PATH):
        logger.error("Manifest not found at %s. Run dataset_prep first.", MANIFEST_PATH)
        sys.exit(1)

    ds = StyleDataset(MANIFEST_PATH, style, resolution)
    if len(ds) == 0:
        logger.error("No images found for style '%s' in manifest!", style)
        sys.exit(1)
    logger.info("Dataset: %d images for style '%s'", len(ds), style)

    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    # --- Load base model ---
    logger.info("Loading base model...")
    pipe = None
    loaded_model_id = None
    for model_id in BASE_MODELS:
        try:
            logger.info("  Trying: %s ...", model_id)
            pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
            loaded_model_id = model_id
            logger.info("  ✓ Loaded: %s", model_id)
            break
        except Exception as e:
            logger.warning("  ✗ Failed: %s", e)

    if pipe is None:
        logger.error("All base models failed to load.")
        sys.exit(1)

    if device == "cuda":
        pipe.enable_attention_slicing()

    # Extract components
    vae = pipe.vae.to(device, dtype=torch.float32)        # fp32 prevents NaN
    unet = pipe.unet.to(device)
    text_encoder = pipe.text_encoder.to(device, dtype=torch.float32)
    tokenizer = pipe.tokenizer
    noise_scheduler = DDPMScheduler.from_pretrained(loaded_model_id, subfolder="scheduler")

    # Freeze VAE and text encoder
    vae.requires_grad_(False)
    vae.eval()
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    del pipe
    torch.cuda.empty_cache()

    # --- Encode style prompt ---
    style_prompt = STYLE_PROMPTS[style]
    logger.info("Encoding style prompt: '%s'", style_prompt)
    with torch.no_grad():
        text_input = tokenizer(
            style_prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        style_embeddings = text_encoder(text_input.input_ids.to(device))[0]
        style_embeddings = style_embeddings.to(dtype=dtype)

    # --- Attach LoRA ---
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["to_q", "to_v", "to_k", "to_out.0"],
        lora_dropout=0.05,
    )
    unet = get_peft_model(unet, lora_config)
    unet.train()

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total = sum(p.numel() for p in unet.parameters())
    logger.info("Trainable params: %s / %s (%.2f%%)", f"{trainable:,}", f"{total:,}",
                100 * trainable / total)

    # Cast trainable LoRA params to fp32 for gradient stability
    for _, param in unet.named_parameters():
        if param.requires_grad:
            param.data = param.data.float()

    torch.cuda.empty_cache()

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(unet.parameters(), lr=lr, weight_decay=1e-2)

    # --- Training loop ---
    logger.info("Starting training for %d steps...", steps)
    pbar = tqdm(total=steps, desc=f"Training {style}")
    step = 0
    running_loss = 0.0
    done = False

    while not done:
        for batch in dl:
            if done:
                break

            pixel_values = batch.to(device, dtype=torch.float32)

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                latents = latents.to(dtype=torch.float32)

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=device,
            ).long()

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            encoder_hidden_states = style_embeddings.expand(latents.shape[0], -1, -1)

            optimizer.zero_grad()

            with torch.autocast("cuda"):
                noise_pred = unet(
                    noisy_latents.to(dtype=dtype),
                    timesteps,
                    encoder_hidden_states,
                ).sample
                loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float())

            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{running_loss / step:.4f}")
            pbar.update(1)

            if step >= steps:
                done = True
                break

    pbar.close()
    logger.info("Training complete. Average loss: %.4f", running_loss / max(step, 1))

    # --- Save LoRA ---
    save_path = os.path.join(LORA_DIR, style)
    os.makedirs(save_path, exist_ok=True)
    unet.save_pretrained(save_path)
    logger.info("LoRA weights saved to: %s", save_path)

    del unet, vae, text_encoder, optimizer
    if device == "cuda":
        torch.cuda.empty_cache()

    logger.info("Done training '%s'!", style)


if __name__ == "__main__":
    train()
