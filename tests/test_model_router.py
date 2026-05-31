"""
Unit Tests for Phase 3: Model Router and Fallback System
"""
import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from api.app import app
from src.models.model_router import ModelRouter, MODEL_FLUX_DEV, MODEL_SDXL, MODEL_SD15

class TestModelRouter(unittest.TestCase):
    def setUp(self):
        # Reset the singleton for each test
        ModelRouter._instance = None
        self.client = TestClient(app)

    @patch("src.models.backends.flux_dev_backend.FluxDevBackend.load_model")
    def test_router_selection_and_cache(self, mock_load):
        router = ModelRouter()
        
        # Test selection
        backend, name = router.load_model(MODEL_FLUX_DEV)
        self.assertEqual(name, MODEL_FLUX_DEV)
        self.assertEqual(router.active_model_name, MODEL_FLUX_DEV)
        
        # Backend should have been cached
        self.assertIn(MODEL_FLUX_DEV, router.backends)
        
        # Second call shouldn't create a new instance
        backend2, name2 = router.load_model(MODEL_FLUX_DEV)
        self.assertIs(backend, backend2)

    def test_invalid_model_name(self):
        router = ModelRouter()
        with self.assertRaises(RuntimeError):
            router.load_model("nonexistent_model_123")

    @patch("src.models.backends.flux_dev_backend.FluxDevBackend.load_model")
    @patch("src.models.backends.sdxl_backend.SDXLBackend.load_model")
    def test_fallback_behavior(self, mock_sdxl_load, mock_flux_load):
        # Simulate Flux failing
        mock_flux_load.side_effect = RuntimeError("OOM")
        
        router = ModelRouter()
        backend, name = router.load_model(MODEL_FLUX_DEV)
        
        # Should have fallen back to SDXL
        self.assertEqual(name, MODEL_SDXL)
        self.assertEqual(router.active_model_name, MODEL_SDXL)
        mock_sdxl_load.assert_called_once()

    @patch("src.models.model_router.ModelRouter.get_active_backend")
    def test_debug_endpoint(self, mock_get_active):
        mock_backend = MagicMock()
        mock_backend.is_loaded = True
        mock_backend.device = "cuda"
        mock_get_active.return_value = mock_backend
        
        router = ModelRouter()
        router.active_model_name = MODEL_FLUX_DEV

        response = self.client.get("/debug/model")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertEqual(data["active_model"], MODEL_FLUX_DEV)
        self.assertTrue(data["loaded"])
        self.assertEqual(data["device"], "cuda")
        self.assertTrue(data["vram_optimized"])

if __name__ == "__main__":
    unittest.main()
