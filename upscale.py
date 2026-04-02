import os
import sys
import cv2
import glob
import torch
import torchvision
import torchvision.transforms
import torchvision.transforms.functional as F

# Monkey patch basicsr torchvision compatibility bug natively for RTX 3050 portability
if not hasattr(torchvision.transforms, 'functional_tensor'):
    sys.modules['torchvision.transforms.functional_tensor'] = F

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

def main():
    input_dir = './exports'
    output_dir = './exports/upscaled'
    os.makedirs(output_dir, exist_ok=True)
    
    # Check for images
    images = []
    for ext in ('*.png', '*.jpg', '*.jpeg'):
        images.extend(glob.glob(os.path.join(input_dir, ext)))
        
    if not images:
        print(f"No images found in {input_dir}")
        return
        
    print(f"Found {len(images)} images in {input_dir}.")
    
    # For RTX 3050 safe limits, we use the compact realesr-animevideov3 model (x2) 
    # or general RealESRGAN_x2plus. Let's use RealESRGAN_x2plus for general quality.
    
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    netscale = 2
    model_name = 'RealESRGAN_x2plus'
    
    print(f"Initializing 2x upscaler ({model_name})...")
    
    # Set tile size to avoid OOM on 4GB-6GB VRAM GPUs (RTX 3050 safe)
    tile_size = 400 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_half = True if torch.cuda.is_available() else False

    try:
        # The library downloads the model automatically if not present locally
        upsampler = RealESRGANer(
            scale=netscale,
            model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
            model=model,
            tile=tile_size,
            tile_pad=10,
            pre_pad=0,
            half=use_half,
            device=device
        )
        print("Upsampler initialized successfully.")
    except Exception as e:
        print(f"Failed to load upsampler: {e}")
        return

    for img_path in images:
        filename = os.path.basename(img_path)
        out_path = os.path.join(output_dir, f"upscaled_2x_{filename}")
        
        print(f"Upscaling {filename}...")
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            print(f"Failed to read {filename}")
            continue
            
        try:
            output, _ = upsampler.enhance(img, outscale=2)
            cv2.imwrite(out_path, output)
            print(f"Saved to {out_path}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error upscaling {filename}: {e}")

if __name__ == "__main__":
    main()
