"""
LocalStyleAI - Server API test.

Sends requests to the running API server to verify /generate works.
Start the server first:  python run.py --mode server
"""

import os
import sys
import requests
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.helpers import EXPORTS_DIR, get_logger

logger = get_logger("test_gen")

SERVER_URL = "http://127.0.0.1:8000"

TEST_PROMPTS = [
    ("A futuristic city with flying cars, cyberpunk style", "default"),
    ("A peaceful mountain lake at sunset", "ghibli_clean"),
    ("A fierce wolf howling at the moon", "lineart_clean"),
]


def test_health():
    """Verify the server is running."""
    resp = requests.get(f"{SERVER_URL}/health")
    assert resp.status_code == 200, f"Health check failed: {resp.status_code}"
    logger.info("✓ Health check passed: %s", resp.json())


def test_styles():
    """Verify /styles endpoint returns available styles."""
    resp = requests.get(f"{SERVER_URL}/styles")
    assert resp.status_code == 200
    data = resp.json()
    assert "styles" in data
    logger.info("✓ Available styles: %s", data["styles"])


def test_generate():
    """Send generation requests and save outputs."""
    os.makedirs(EXPORTS_DIR, exist_ok=True)

    for i, (prompt, style) in enumerate(TEST_PROMPTS, 1):
        logger.info("Request %d: prompt='%s...' style='%s'", i, prompt[:40], style)
        resp = requests.post(
            f"{SERVER_URL}/generate",
            data={"prompt": prompt, "style": style},
            stream=True,
        )
        if resp.status_code == 200:
            out_path = os.path.join(EXPORTS_DIR, f"test_server_{i}.png")
            with open(out_path, "wb") as f:
                resp.raw.decode_content = True
                shutil.copyfileobj(resp.raw, f)
            logger.info("✓ Saved: %s", out_path)
        else:
            logger.error("✗ Failed (%d): %s", resp.status_code, resp.text[:200])


if __name__ == "__main__":
    test_health()
    test_styles()
    test_generate()
    logger.info("All server tests complete!")
