# 🎨 LocalStyleAI

**LocalStyleAI** is a locally-run AI image stylization system powered by Stable Diffusion and custom-trained LoRA adapters. Generate anime, Ghibli, lineart, and more — entirely on your own machine with no cloud dependencies.

---

## ✨ Features

- 🖼️ **Multi-style image generation** — anime, Ghibli, lineart, and custom styles
- 🧠 **Custom LoRA training** — train your own style adapters on local datasets
- ⚡ **Low-VRAM optimized** — runs on GPUs with as little as 4GB VRAM (RTX 3050+)
- 🌐 **REST API server** — FastAPI backend with a built-in web UI
- 🔼 **Built-in upscaling** — Real-ESRGAN upscaler for high-resolution outputs

---

## 🗂️ Project Structure

```
LocalStyleAI/
├── scripts/
│   ├── server.py          # FastAPI inference server
│   └── index.html         # Web UI frontend
├── data/
│   ├── anime_clean/       # Training images for anime style
│   ├── ghibli_clean/      # Training images for Ghibli style
│   └── lineart_clean/     # Training images for lineart style
├── loras/                 # Trained LoRA weights (generated after training)
├── exports/               # Generated image outputs
├── train_lora.py          # LoRA training script
├── dataset_prep.py        # Dataset preparation and captioning
├── upscale.py             # Real-ESRGAN upscaling script
├── requirements.txt       # Python dependencies
└── README.md
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.10+
- NVIDIA GPU with 4GB+ VRAM (RTX 2060+ recommended)
- CUDA 11.8 or 12.1 installed

### 1. Clone the repository

```bash
git clone https://github.com/your-username/LocalStyleAI.git
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

> **Note:** The `torch` package above uses CUDA 11.8 (`cu118`). If you have CUDA 12.1, install PyTorch separately:
> ```bash
> pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 --index-url https://download.pytorch.org/whl/cu121
> ```

---

## 🚀 Running the Server

```bash
uvicorn scripts.server:app --host 0.0.0.0 --port 8000
```

Then open your browser at **http://localhost:8000** to access the web UI.

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/generate` | POST | Generate an image from a prompt |
| `/styles` | GET | List available styles / LoRAs |
| `/upscale` | POST | Upscale a generated image |

---

## 🏋️ Training a LoRA

### 1. Prepare your dataset

Place your training images inside:

```
data/
└── <style_name>/          # e.g. data/anime_clean/
    ├── image_001.jpg
    ├── image_002.jpg
    └── ...
```

Then run the dataset preparation script to auto-caption images:

```bash
python dataset_prep.py
```

### 2. Start training

```bash
python train_lora.py
```

Training configuration (rank, learning rate, steps, etc.) can be edited at the top of `train_lora.py`.

Trained LoRA weights will be saved to `loras/<style_name>/`.

---

## 🖼️ Generating Images

### Via the Web UI

1. Start the server (see above)
2. Open **http://localhost:8000**
3. Enter a prompt, select a style, and hit **Generate**

### Via the API

```python
import requests

response = requests.post("http://localhost:8000/generate", json={
    "prompt": "a girl in anime style, detailed, vibrant colors",
    "style": "anime",
    "steps": 30,
    "guidance_scale": 7.5,
    "width": 512,
    "height": 512
})

with open("output.png", "wb") as f:
    f.write(response.content)
```

---

## 🔼 Upscaling Images

```bash
python upscale.py --input exports/output.png --output exports/output_4x.png --scale 4
```

---

## 🛠️ Configuration

The server automatically selects a base model in this priority order:
1. **DreamShaper** (if present in Hugging Face cache)
2. **Realistic Vision**
3. **Stable Diffusion v1.5** (fallback)

LoRA weights in `loras/<style>/` are loaded automatically when a style is selected.

---

## 📋 Requirements Overview

| Package | Version | Purpose |
|---|---|---|
| torch | 2.1.2 | Deep learning backend |
| diffusers | 0.25.1 | Stable Diffusion pipeline |
| transformers | 4.37.2 | Text encoder |
| accelerate | 0.26.1 | Training acceleration |
| peft | ≥0.7.0 | LoRA adapter support |
| fastapi | ≥0.108.0 | API server |
| Real-ESRGAN | latest | Image upscaling |

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙏 Credits

- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- [PEFT / LoRA](https://github.com/huggingface/peft)
