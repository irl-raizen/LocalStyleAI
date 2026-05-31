"""
Unit Tests for Phase 6: Scene Editing
"""
import unittest
from unittest.mock import patch, MagicMock
from PIL import Image

from src.editing.scene_analyzer import SceneAnalyzer
from src.editing.edit_interpreter import EditPlan, interpret_edit
from src.editing.region_selector import RegionSelector
from src.editing.edit_manager import EditManager
from api.app import app
from fastapi.testclient import TestClient

class TestSceneEditing(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_scene_analyzer(self):
        analyzer = SceneAnalyzer()
        img = Image.new("RGB", (64, 64))
        result = analyzer.analyze(img, "a red dragon")
        
        self.assertIn("red dragon", result["objects"])
        self.assertEqual(result["environment"], "extracted from prompt")

    @patch("src.editing.edit_interpreter.requests.post")
    def test_edit_interpreter(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": '{"action": "modify", "target": "dragon", "attribute": "size", "value": "larger"}'
        }
        mock_post.return_value = mock_response

        plan = interpret_edit("Make dragon larger")
        self.assertIsNotNone(plan)
        self.assertEqual(plan.action, "modify")
        self.assertEqual(plan.target, "dragon")
        self.assertEqual(plan.value, "larger")

    def test_region_selector(self):
        selector = RegionSelector()
        img = Image.new("RGB", (100, 100))
        result = selector.locate_and_mask(img, "dragon")
        
        self.assertTrue(result["mask_found"])
        self.assertIsNotNone(result["mask"])
        self.assertGreater(result["confidence"], 0.0)

    @patch("src.editing.edit_manager.evaluate_image_against_prompt")
    @patch("src.editing.edit_manager.interpret_edit")
    @patch("src.editing.edit_manager.InpaintingEngine.apply_edit")
    def test_edit_manager(self, mock_apply, mock_interpret, mock_evaluate):
        # Setup mocks
        mock_evaluate.return_value = {"overall_score": 0.9}
        mock_interpret.return_value = EditPlan(action="add", target="sky", attribute="", value="moon")
        mock_img = Image.new("RGB", (64, 64))
        mock_apply.return_value = mock_img
        
        manager = EditManager()
        edited, metadata = manager.process_edit(mock_img, "Add a moon")
        
        # Verify metadata outputs
        self.assertEqual(metadata["action"], "add")
        self.assertEqual(metadata["edit_plan"]["value"], "moon")

if __name__ == "__main__":
    unittest.main()
