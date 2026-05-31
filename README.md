<p align="center">
  <h1 align="center">🎨 Local Style AI Generator</h1>
  <p align="center">
    <strong>Train and generate styled images locally using Stable Diffusion + LoRA</strong>
  </p>
  <p align="center">
    <a href="#features">Features</a> •
    <a href="#tech-stack">Tech Stack</a> •
    <a href="#installation">Installation</a> •
    <a href="#usage">Usage</a> •
    <a href="#dataset-setup">Dataset</a> •
    <a href="#api-reference">API</a>
  </p>
</p>

---

## 📖 About

**Local Style AI Generator** is a fully local, GPU-accelerated image generation system that lets you **train custom art styles** and **generate images** — all on your own machine with no cloud dependency.

Train LoRA adapters on your own image datasets to teach the model new art styles (anime, Ghibli, line art, or anything you want), then generate images through a CLI, Python API, or a sleek web interface.

Optimized to run on **consumer GPUs with as little as 4 GB VRAM** (RTX 3050+).

---

## ✨ Features

| Feature | Description |
|---|---|
| 🖼️ **Multi-style Generation** | Anime, Studio Ghibli, line art — or train your own |
| 🧠 **Custom LoRA Training** | Train style adapters on small datasets (~20-50 images) |
| ⚡ **Low-VRAM Optimized** | Runs on 4 GB GPUs with CPU offloading & attention slicing |
| 🌐 **REST API + Web UI** | FastAPI backend with a built-in glassmorphism web interface |
| 🔼 **Real-ESRGAN Upscaling** | 2× upscaling built in for high-resolution outputs |
| 📦 **Single Entry Point** | `run.py` for training, generation, dataset prep, and server |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Base Model | Stable Diffusion v1.5 / DreamShaper 8 |
| Fine-tuning | LoRA via PEFT |
| ML Framework | PyTorch 2.1 + CUDA 11.8 |
| Diffusion Library | Hugging Face Diffusers |
| API Server | FastAPI + Uvicorn |
| Upscaling | Real-ESRGAN |
| Frontend | Vanilla HTML/CSS/JS with glassmorphism design |

---

## 📁 Project Structure

```
LocalStyleAI/
├── src/
│   ├── train/
│   │   └── train_lora.py         # LoRA training pipeline
│   ├── inference/
│   │   └── generate.py           # Image generation module
│   ├── data/
│   │   └── dataset_prep.py       # Dataset processing & manifest creation
│   └── utils/
│       └── helpers.py            # Shared constants, logging, model loader
│
├── api/
│   ├── app.py                    # FastAPI server
│   └── static/
│       └── index.html            # Web UI
│
├── configs/
│   └── anime_dataset.toml        # Kohya-ss training config
│
├── scripts/
│   └── prepare_kohya_data.py     # Caption file generator for kohya-ss
│
├── tests/
│   ├── test_model.py             # Model loading & generation tests
│   └── test_gen.py               # API server endpoint tests
│
├── data/                         # Training images (not tracked by git)
│   ├── anime_clean/
│   ├── ghibli_clean/
│   └── lineart_clean/
│
├── loras/                        # Trained LoRA weights (generated)
├── exports/                      # Generated image outputs
│
├── run.py                        # 🚀 Main entry point
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### Prerequisites

- **Python** 3.10+
- **NVIDIA GPU** with 4 GB+ VRAM (RTX 2060+ recommended)
- **CUDA** 11.8 or 12.1

### 1. Clone the repository

```bash
git clone https://github.com/irl-raizen/LocalStyleAI.git
cd LocalStyleAI
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note (CUDA 12.1 users):** Replace the PyTorch install:
> ```bash
> pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 \
>     --index-url https://download.pytorch.org/whl/cu121
> ```

---

## 🧠 AI Prompt Intelligence Layer (Ollama)

This project features an AI-powered prompt enhancement layer that expands simple, vague prompts into rich visual prompts using a local LLM via **Ollama**.

### Setup Local LLM

1. Install [Ollama](https://ollama.com).
2. Download and run the default model (`qwen3:8b`):
   ```bash
   ollama run qwen3:8b
   ```

### How It Works

This system uses a **Structured Prompt Architecture** (Phase 2):

1. **User Prompt**: The user inputs a simple text query (e.g. `"a samurai standing in rain"`).
2. **Prompt Analyzer**: Connects to the local LLM to extract semantic fields into a structured JSON schema:
   - `subject`, `appearance`, `action`, `environment`, `lighting`, `camera`, `style`, `mood`, `quality`.
   - Also supports future fields for character traits, pose, and custom color palettes.
3. **Prompt Composer**: Converts the structured components into an optimized, comma-separated diffusion prompt.
4. **Stable Diffusion**: Receives the final composed prompt for high-fidelity generation.

If structured extraction fails (e.g., if Ollama is unavailable or times out), the system automatically falls back to the Phase 1 prompt enhancer or style prefixing to ensure generation never crashes.

### Configuration

You can customize the connection to Ollama using the following environment variables:

| Environment Variable | Default Value | Description |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | The Ollama API endpoint |
| `OLLAMA_MODEL` | `qwen3:8b` | The local model to use for enhancement |
| `OLLAMA_TIMEOUT` | `10.0` | Connection timeout in seconds |

Example override (Windows):
```powershell
$env:OLLAMA_MODEL="llama3.2:1b"
python run.py --mode server
```

---

## 🧠 Memory Engine (Phase 4)

LocalStyleAI features a persistent memory system that enriches prompts automatically using stored information.

### Features
1. **Character Memory**: When you describe a character in a prompt (e.g., *"Sarah, a 24-year-old woman with long red hair"*), the system uses Ollama to extract and securely save her traits. Future prompts can simply mention *"Sarah"* and the Memory Engine will automatically inject her stored physical attributes before generation.
2. **Scene Memory**: Automatically maps scene keywords to their associated lighting and environments.
3. **Style Memory**: Remembers your preferred style. If you don't explicitly specify a style for a prompt, the system will apply the last used style.
4. **Auto-Recovery**: Corrupted memory JSON files are automatically detected, backed up, and safely reset so generations never crash.

You can view and manage stored memories at the **Memory Manager UI**: `http://localhost:8000/memory`

---

## 👁️ Vision Critic & Auto-Regeneration (Phase 5)

A closed-loop image quality system that evaluates generated images against the original user intent and automatically improves poor results.

### Features
1. **Vision Critic Models**: Uses `google/siglip-base-patch16-224` (or `openai/clip-vit-base-patch32`) to "look" at the generated image and compare it with the prompt. 
2. **Prompt Matcher**: Extracts key objects from the prompt and asks the Vision Critic if they are present in the image.
3. **Auto-Regeneration Loop**: If the image fails the quality threshold or misses key elements, the system automatically appends repair hints (e.g., `IMPORTANT: Clearly visible red dragon`) and regenerates up to 3 times.
4. **Metadata Storage**: Saves execution metrics (score, attempts, critic model) inside the image headers and alongside the image as a JSON file.

### Configuration

| Environment Variable | Default Value | Description |
|---|---|---|
| `VISION_MODEL` | `siglip` | Which vision critic to use (`siglip` or `clip`) |
| `MAX_REGEN_ATTEMPTS` | `3` | Maximum regeneration retries |
| `MIN_ACCEPTABLE_SCORE`| `0.75` | Minimum overall visual score to accept immediately |

---

## 🖌️ Scene Editing & Intelligent Inpainting (Phase 6)

Modify existing generated images through natural language instructions without regenerating the entire image.

### Features
1. **Edit Interpreter**: Powered by Ollama, converts conversational instructions (e.g., *"Make dragon larger"* or *"Add snowfall"*) into structured actionable edit plans.
2. **Intelligent Inpainting**: Automatically routes edits to FLUX img2img, SDXL Inpainting, or SD1.5 Inpainting depending on your active model.
3. **Region Selector**: Dynamically isolates the object or environment to edit. 
4. **Edit Vision Critic Verification**: The Vision Critic evaluates the final edit to ensure it actually matches your instruction before presenting it.

You can edit images directly from the main Web UI by scrolling down to the **Scene Editor**.

---

## 📦 Asset Studio & Automated Training (Phase 7)

Transform LocalStyleAI into a complete AI asset creation platform. Automatically train, organize, version, and deploy LoRA assets.

### Features
1. **Asset Registry**: A persistent database tracking all your trained Characters, Styles, Environments, and Objects.
2. **Automated LoRA Training**: A background queue system that automatically validates your dataset, generates captions using Florence-2/BLIP, and runs Kohya-ss training scripts without blocking the UI.
3. **Evaluation Engine**: Automatically evaluates trained LoRAs using the Vision Critic and assigns them a quality score.
4. **Intelligent Generation**: When generating an image, if you mention a registered Character (e.g., "Sarah"), the `ModelRouter` will dynamically load their latest LoRA file behind the scenes!

You can manage your assets via the new **Asset Studio** and **Training Dashboard** pages linked in the UI header.

---

## 🖼️ Image Generation Models

LocalStyleAI supports multiple backend models through a robust `ModelRouter`.

### Supported Models

- `flux_dev`: High-quality FLUX.1-dev model (default).
- `flux_schnell`: Fast FLUX.1-schnell model.
- `sdxl`: Stable Diffusion XL.
- `sd15`: Stable Diffusion v1.5 (with LoRA support).

### Fallback System

If a model fails to load (e.g. out of memory), the router automatically falls back to a lighter model to ensure generation never crashes:
`flux_dev` -> `sdxl` -> `sd15`

### Configuration

You can customize the model by setting the `IMAGE_MODEL` environment variable.

| Environment Variable | Default Value | Description |
|---|---|---|
| `IMAGE_MODEL` | `flux_dev` | The active image model backend |

Example:
```powershell
$env:IMAGE_MODEL="flux_schnell"
python run.py --mode server
```

---

## 🚀 Usage

All operations go through the unified `run.py` entry point:

### Generate an image

```bash
python run.py --mode generate --prompt "a sunset over the ocean" --style ghibli_clean
```

### Train a LoRA adapter

```bash
python run.py --mode train --style anime_clean --steps 1200 --lr 2e-5
```

### Prepare a dataset

```bash
python run.py --mode prepare
```

### Start the API server

```bash
python run.py --mode server --port 8000
```

Then open **http://localhost:8000** for the web UI.

---

## 🗂️ Dataset Setup

Place your training images in style-specific subdirectories:

```
data/
├── anime_clean/       # 20-50 anime-style images
│   ├── img_001.png
│   ├── img_002.jpg
│   └── ...
├── ghibli_clean/      # 20-50 Ghibli-style images
└── lineart_clean/     # 20-50 line art images
```

**Supported formats:** `.png`, `.jpg`, `.jpeg`, `.webp`
**Minimum size:** 200×200 pixels (auto-resized to 512×512)

Then run the dataset preparation:

```bash
python run.py --mode prepare
```

This validates images, resizes them, and generates a `manifest.jsonl` for training.

---

## 🌐 API Reference

Start the server with `python run.py --mode server`, then:

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Web UI |
| `/health` | GET | Health check — `{"status": "ok", "gpu": true}` |
| `/styles` | GET | List available styles |
| `/generate` | POST | Generate an image (form: `prompt`, `style`) |
| `/debug/prompt` | GET | Debug structured prompt components (query: `prompt`, `style`) |
| `/debug/model` | GET | Debug model backend configurations and active model |
| `/memory` | GET | Memory Manager Web UI |
| `/memory/characters` | GET | Get all stored characters |
| `/memory/characters/{name}`| DELETE| Delete a stored character |
| `/memory/scenes` | GET | Get all stored scenes |
| `/memory/styles` | GET | Get stored styles |
| `/debug/critic` | GET | Get active vision critic model status |
| `/debug/evaluate` | POST | Evaluate an image upload against a text prompt |
| `/edit` | POST | Edit an existing image (form: `instruction`, `image`) |
| `/debug/edit` | POST | Debug edit interpretation plan |
| `/train` | POST | Start an automated LoRA training job |
| `/training/jobs` | GET | List all active training jobs |
| `/assets` | GET | List all registered assets |
| `/assets/{asset_id}` | GET | Get a specific asset |
| `/assets/{asset_id}` | DELETE | Delete an asset |
| `/assets/search` | POST | Search the asset registry |

### Example: Generate via API

```python
import requests

response = requests.post("http://localhost:8000/generate", data={
    "prompt": "a girl in anime style, vibrant colors",
    "style": "anime_clean",
})

with open("output.png", "wb") as f:
    f.write(response.content)
```

---

## 🖼️ Example Outputs

| Anime Style | Ghibli Style | Line Art |
|---|---|---|
| *anime_clean* | *ghibli_clean* | *lineart_clean* |
| Cel shaded, vibrant | Hand painted, watercolor | Ink drawing, B&W |

> Generate your own samples with:
> ```bash
> python run.py --mode generate --prompt "a dragon" --style lineart_clean
> ```

---

## 📋 Commands Quick Reference

| Command | What it does |
|---|---|
| `python run.py --mode generate --prompt "..." --style anime_clean` | Generate a styled image |
| `python run.py --mode train --style ghibli_clean` | Train a LoRA on Ghibli data |
| `python run.py --mode prepare` | Process training images |
| `python run.py --mode server` | Launch the web API |
| `python tests/test_model.py` | Test model loading & generation |
| `python tests/test_gen.py` | Test API endpoints |

---

## 🔧 Configuration

**Base model priority** (auto-selected):
1. DreamShaper 8
2. Realistic Vision V5.1
3. Stable Diffusion v1.5

**LoRA training defaults:**
- Rank: 8
- Alpha: 32
- Target modules: `to_q`, `to_v`, `to_k`, `to_out.0`
- Learning rate: 2e-5
- Steps: 1200

All configurable via CLI arguments.

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙏 Acknowledgements

- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [PEFT / LoRA](https://github.com/huggingface/peft)
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- [DreamShaper](https://civitai.com/models/4384/dreamshaper)
