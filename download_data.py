import requests
import os

urls = {
    "lineart": "https://upload.wikimedia.org/wikipedia/commons/1/1f/Line-art_drawing_of_a_cat.jpg",
    "ghibli": "https://upload.wikimedia.org/wikipedia/commons/7/76/Hendrik_Voogd_-_Italian_landscape_with_Umbrella_Pines.jpg",
    "anime": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/Anime_Girl.svg/800px-Anime_Girl.svg.png"
}

for style, url in urls.items():
    os.makedirs(f'localstyleai/data/{style}', exist_ok=True)
    try:
        print(f"Downloading {style} from {url}...")
        resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        if resp.status_code == 200:
            # Determine extension
            ext = 'jpg' if 'jpg' in url else 'png'
            with open(f'localstyleai/data/{style}/image_01.{ext}', 'wb') as f:
                f.write(resp.content)
            print(f"Downloaded {style}")
        else:
            print(f"Failed {style}: {resp.status_code}")
    except Exception as e:
        print(f"Failed {style}: {e}")
