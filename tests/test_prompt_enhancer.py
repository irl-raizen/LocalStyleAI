"""
Unit Tests for AI Prompt Intelligence Layer
"""

import os
import json
import unittest
from unittest.mock import patch, MagicMock
import requests

from src.ai.prompt_enhancer import (
    enhance_prompt,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_OLLAMA_URL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_TIMEOUT,
)


class TestPromptEnhancer(unittest.TestCase):

    @patch("requests.post")
    def test_successful_enhancement(self, mock_post):
        """Test successful prompt enhancement with valid JSON response from Ollama."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": json.dumps({
                "enhanced_prompt": "a beautiful cinematic landscape with mountains and rivers",
                "negative_prompt": "ugly, blurry"
            })
        }
        mock_post.return_value = mock_response

        res = enhance_prompt("mountains", "default")
        self.assertEqual(res["enhanced_prompt"], "a beautiful cinematic landscape with mountains and rivers")
        self.assertEqual(res["negative_prompt"], "ugly, blurry")

        # Verify correct arguments were passed to requests.post
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(args[0], DEFAULT_OLLAMA_URL)
        
        payload = kwargs["json"]
        self.assertEqual(payload["model"], DEFAULT_OLLAMA_MODEL)
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["format"], "json")
        self.assertIn("mountains", payload["prompt"])

    @patch("requests.post")
    def test_invalid_json_response(self, mock_post):
        """Test fallback when Ollama returns invalid or unparseable JSON response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "{invalid_json:..."
        }
        mock_post.return_value = mock_response

        res = enhance_prompt("mountains", "default")
        self.assertEqual(res["enhanced_prompt"], "mountains")
        self.assertEqual(res["negative_prompt"], DEFAULT_NEGATIVE_PROMPT)

    @patch("requests.post")
    def test_missing_fields_json_response(self, mock_post):
        """Test fallback when Ollama returns JSON missing enhanced_prompt field."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": json.dumps({
                "something_else": "here"
            })
        }
        mock_post.return_value = mock_response

        res = enhance_prompt("mountains", "default")
        self.assertEqual(res["enhanced_prompt"], "mountains")
        self.assertEqual(res["negative_prompt"], DEFAULT_NEGATIVE_PROMPT)

    @patch("requests.post")
    def test_ollama_unavailable(self, mock_post):
        """Test fallback when Ollama server is completely unavailable (ConnectionError)."""
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")

        res = enhance_prompt("mountains", "default")
        self.assertEqual(res["enhanced_prompt"], "mountains")
        self.assertEqual(res["negative_prompt"], DEFAULT_NEGATIVE_PROMPT)

    @patch("requests.post")
    def test_timeout(self, mock_post):
        """Test fallback when Ollama request times out."""
        mock_post.side_effect = requests.exceptions.Timeout("Request timed out")

        res = enhance_prompt("mountains", "default")
        self.assertEqual(res["enhanced_prompt"], "mountains")
        self.assertEqual(res["negative_prompt"], DEFAULT_NEGATIVE_PROMPT)

    @patch("requests.post")
    def test_style_aware_prompt_generation(self, mock_post):
        """Test that the style parameter is correctly passed to the Ollama prompt."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": json.dumps({
                "enhanced_prompt": "whimsical ghibli landscape",
                "negative_prompt": "ugly"
            })
        }
        mock_post.return_value = mock_response

        res = enhance_prompt("castle", "ghibli_clean")
        self.assertEqual(res["enhanced_prompt"], "whimsical ghibli landscape")

        mock_post.assert_called_once()
        payload = mock_post.call_args[1]["json"]
        self.assertIn("ghibli_clean", payload["prompt"])
        self.assertIn("castle", payload["prompt"])

    @patch("requests.post")
    def test_config_override_via_env(self, mock_post):
        """Test that configuration can be overridden using environment variables."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": json.dumps({
                "enhanced_prompt": "enhanced",
                "negative_prompt": "negative"
            })
        }
        mock_post.return_value = mock_response

        custom_env = {
            "OLLAMA_URL": "http://custom-ollama:11434/api/generate",
            "OLLAMA_MODEL": "custom-model:latest",
            "OLLAMA_TIMEOUT": "5.5"
        }

        with patch.dict(os.environ, custom_env):
            res = enhance_prompt("mountains", "default")
            self.assertEqual(res["enhanced_prompt"], "enhanced")
            
            mock_post.assert_called_once()
            args, kwargs = mock_post.call_args
            self.assertEqual(args[0], "http://custom-ollama:11434/api/generate")
            self.assertEqual(kwargs["json"]["model"], "custom-model:latest")
            self.assertEqual(kwargs["timeout"], 5.5)


if __name__ == "__main__":
    unittest.main()
