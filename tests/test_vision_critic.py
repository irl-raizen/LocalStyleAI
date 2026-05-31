"""
Unit Tests for Phase 5: Vision Critic
"""
import unittest
from unittest.mock import patch, MagicMock
from PIL import Image

from src.critic.critic_models import CriticFactory, BaseVisionCritic
from src.critic.prompt_matcher import evaluate_image_against_prompt, extract_key_elements
from src.critic.regeneration_manager import RegenerationManager
from api.app import app
from fastapi.testclient import TestClient

class DummyCritic(BaseVisionCritic):
    def __init__(self):
        super().__init__()
        self.model_name = "dummy"
        self.is_loaded = True
        self.device = "cpu"
        self.scores = [0.9]
    def load(self): pass
    def calculate_similarity(self, image, texts):
        # Return mock scores matching the length of texts
        # Usually we'll return high for the first, low for the rest to test missing elements
        return self.scores[:len(texts)] + [0.05] * (len(texts) - len(self.scores))

class TestVisionCritic(unittest.TestCase):
    def setUp(self):
        CriticFactory._instance = DummyCritic()
        self.client = TestClient(app)
        
    def tearDown(self):
        CriticFactory._instance = None

    def test_extract_key_elements(self):
        prompt = "red dragon carrying blue crystal, in a dark cave"
        elements = extract_key_elements(prompt)
        self.assertIn("red dragon carrying blue crystal", elements)
        self.assertIn("dark cave", elements)

    def test_evaluate_image_against_prompt(self):
        img = Image.new("RGB", (64, 64))
        # Prompt splits into "red dragon"
        prompt = "red dragon, blue crystal"
        # Dummy critic returns [0.9] followed by [0.05]
        # So "red dragon" gets 0.9 (matched), "blue crystal" gets 0.05 (missing)
        result = evaluate_image_against_prompt(img, prompt)
        
        self.assertIn("blue crystal", result["missing_elements"])
        self.assertIn("red dragon", result["matched_elements"])
        self.assertGreater(result["overall_score"], 0)

    def test_regeneration_manager_repair(self):
        manager = RegenerationManager()
        missing = ["blue crystal"]
        prompt = "red dragon"
        repaired = manager.repair_prompt(prompt, missing)
        
        self.assertIn("IMPORTANT:", repaired)
        self.assertIn("blue crystal", repaired)

    def test_regeneration_loop(self):
        manager = RegenerationManager()
        manager.max_attempts = 2
        
        # We need a generate func that tracks calls
        self.calls = 0
        def dummy_generate(prompt, **kwargs):
            self.calls += 1
            return Image.new("RGB", (64, 64))
            
        # Our dummy critic always says the second element is missing, keeping score low
        # Actually overall_score might be high enough to pass. Let's force it below threshold
        CriticFactory._instance.scores = [0.1] 
        
        img, metadata = manager.run_generation_loop(dummy_generate, "test prompt, missing thing")
        
        # Should have tried 2 times
        self.assertEqual(self.calls, 2)
        self.assertEqual(metadata["attempts"], 2)

    def test_api_debug_critic(self):
        response = self.client.get("/debug/critic")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["critic_model"], "dummy")
        self.assertTrue(data["loaded"])

if __name__ == "__main__":
    unittest.main()
