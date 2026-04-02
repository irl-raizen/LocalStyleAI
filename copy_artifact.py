import shutil
import os

src = r"C:/Users/Ashamumbai/.gemini/antigravity/brain/3666ac03-6e50-4f9d-931f-22901f6f97c1/ghibli_landscape_sample_1763759133951.png"
dst_dir = r"localstyleai/data/ghibli"
dst = os.path.join(dst_dir, "image_01.png")

os.makedirs(dst_dir, exist_ok=True)

try:
    shutil.copy(src, dst)
    print(f"Copied {src} to {dst}")
except Exception as e:
    print(f"Error copying: {e}")
