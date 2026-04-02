import requests
import shutil
import os

url = "http://127.0.0.1:8000/generate"
prompts = [
    "A futuristic city with flying cars, cyberpunk style, high quality",
    "A dense cyberpunk forest at night, glowing neon trees",
    "An anime portrait of a cyberpunk character"
]

os.makedirs('./exports', exist_ok=True)

for i, p in enumerate(prompts):
    data = {"prompt": p, "style": "default"}
    print(f"Sending request {i+1} to {url}...")
    try:
        response = requests.post(url, data=data, stream=True)
        if response.status_code == 200:
            out_file = f"./exports/test_server_output_{i+1}.png"
            with open(out_file, "wb") as f:
                response.raw.decode_content = True
                shutil.copyfileobj(response.raw, f)
            print(f"Success! Image saved to {out_file}")
        else:
            print(f"Failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error: {e}")
