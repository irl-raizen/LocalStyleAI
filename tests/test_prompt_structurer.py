"""
Unit Tests for Phase 2: Structured Prompt Builder & API Debug Endpoint
"""

import os
import json
import unittest
from unittest.mock import patch, MagicMock
import requests
from fastapi.testclient import TestClient

from api.app import app
from src.ai.prompt_structurer import structure_prompt, StructuredPrompt
from src.ai.prompt_composer import compose_prompt
from src.ai.prompt_enhancer import DEFAULT_NEGATIVE_PROMPT


class TestStructuredPromptBuilder(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    @patch("requests.post")
    def test_prompt_extraction_success(self, mock_post):
        """Test successful structured prompt extraction from Ollama."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": json.dumps({
                "subject": "samurai warrior",
                "appearance": "traditional armor",
                "action": "standing",
                "environment": "rainy street",
                "lighting": "moody overcast lighting",
                "camera": "cinematic medium shot",
                "style": "anime_clean",
                "mood": "dramatic",
                "quality": "high detail"
            })
        }
        mock_post.return_value = mock_response

        res = structure_prompt("a samurai standing in rain", "anime_clean")
        self.assertIsNotNone(res)
        self.assertEqual(res.subject, "samurai warrior")
        self.assertEqual(res.appearance, "traditional armor")
        self.assertEqual(res.action, "standing")
        self.assertEqual(res.environment, "rainy street")
        self.assertEqual(res.lighting, "moody overcast lighting")
        self.assertEqual(res.camera, "cinematic medium shot")
        self.assertEqual(res.style, "anime_clean")
        self.assertEqual(res.mood, "dramatic")
        self.assertEqual(res.quality, "high detail")

    @patch("requests.post")
    def test_prompt_extraction_missing_fields(self, mock_post):
        """Test prompt extraction defaults missing fields to empty strings."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": json.dumps({
                "subject": "dragon",
                "environment": "mountains"
            })
        }
        mock_post.return_value = mock_response

        res = structure_prompt("dragon in mountains", "default")
        self.assertIsNotNone(res)
        self.assertEqual(res.subject, "dragon")
        self.assertEqual(res.environment, "mountains")
        self.assertEqual(res.action, "")
        self.assertEqual(res.camera, "")
        self.assertEqual(res.style, "")

    @patch("requests.post")
    def test_prompt_normalization_nested_fields(self, mock_post):
        """Test that nested dicts and lists from LLM are successfully normalized into strings."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": json.dumps({
                "subject": "samurai",
                "appearance": {
                    "skin_color": "dark",
                    "hair_color": "brown"
                },
                "color_palette": ["blue", "grey"]
            })
        }
        mock_post.return_value = mock_response

        res = structure_prompt("samurai", "default")
        self.assertIsNotNone(res)
        self.assertEqual(res.subject, "samurai")
        self.assertEqual(res.appearance, "skin_color: dark, hair_color: brown")
        self.assertEqual(res.color_palette, "blue, grey")

    @patch("requests.post")
    def test_prompt_extraction_invalid_json(self, mock_post):
        """Test that invalid JSON from Ollama returns None (indicating fallback needed)."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "{invalid json"
        }
        mock_post.return_value = mock_response

        res = structure_prompt("dragon", "default")
        self.assertIsNone(res)

    @patch("requests.post")
    def test_prompt_extraction_timeout(self, mock_post):
        """Test that a timeout returns None (indicating fallback needed)."""
        mock_post.side_effect = requests.exceptions.Timeout("Request timed out")

        res = structure_prompt("dragon", "default")
        self.assertIsNone(res)

    def test_prompt_composition(self):
        """Test that compose_prompt formats fields correctly into a diffusion prompt."""
        structured = StructuredPrompt(
            subject="dragon",
            appearance="golden scales",
            action="flying high",
            environment="mountain peaks",
            lighting="sunset",
            camera="aerial view",
            style="ghibli_clean",
            mood="serene",
            quality="masterpiece"
        )
        composed = compose_prompt(structured)
        
        self.assertIn("dragon with golden scales flying high", composed)
        self.assertIn("mountain peaks", composed)
        self.assertIn("sunset lighting", composed)
        self.assertIn("aerial view", composed)
        self.assertIn("studio ghibli style, hand painted, watercolor, soft lighting, miyazaki art", composed)
        self.assertIn("serene", composed)
        self.assertIn("masterpiece", composed)

    def test_prompt_composition_dict(self):
        """Test compose_prompt with dict input."""
        structured_dict = {
            "subject": "wizard",
            "action": "casting a spell",
            "lighting": "neon glow"
        }
        composed = compose_prompt(structured_dict)
        self.assertIn("wizard casting a spell", composed)
        self.assertIn("neon glow lighting", composed)

    @patch("api.app.structure_prompt")
    def test_api_debug_endpoint_success(self, mock_structure):
        """Test GET /debug/prompt endpoint when extraction succeeds."""
        mock_structure.return_value = StructuredPrompt(
            subject="cyberpunk city",
            environment="neon streets",
            lighting="volumetric lighting",
            style="anime_clean"
        )

        response = self.client.get("/debug/prompt?prompt=cyberpunk city&style=anime_clean")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("structured", data)
        self.assertEqual(data["structured"]["subject"], "cyberpunk city")
        self.assertEqual(data["structured"]["environment"], "neon streets")
        self.assertIn("composed_prompt", data)
        self.assertIn("cyberpunk city", data["composed_prompt"])
        self.assertIn("anime style", data["composed_prompt"])
        self.assertIn("negative_prompt", data)

    @patch("api.app.structure_prompt")
    @patch("api.app.enhance_prompt")
    def test_api_debug_endpoint_fallback(self, mock_enhance, mock_structure):
        """Test GET /debug/prompt fallback to Phase 1 enhanced prompt system on extraction failure."""
        mock_structure.return_value = None
        mock_enhance.return_value = {
            "enhanced_prompt": "enhanced sunset landscape",
            "negative_prompt": "ugly, blurry"
        }

        response = self.client.get("/debug/prompt?prompt=sunset&style=default")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["structured"], {})
        self.assertEqual(data["composed_prompt"], "enhanced sunset landscape")
        self.assertEqual(data["negative_prompt"], "ugly, blurry")
        
        mock_enhance.assert_called_once_with(prompt="sunset", style="default")


if __name__ == "__main__":
    unittest.main()
