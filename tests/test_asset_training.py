"""
Unit Tests for Phase 7: Asset Studio & Training
"""
import os
import unittest
import time
from unittest.mock import patch, MagicMock
from PIL import Image

from src.assets.asset_registry import AssetRegistry, AssetRecord
from src.assets.asset_versioning import AssetVersioning
from src.assets.asset_search import AssetSearch
from src.training.dataset_validator import DatasetValidator
from src.training.auto_captioner import AutoCaptioner
from src.training.training_manager import TrainingManager
from src.training.training_queue import TrainingQueue
from api.app import app
from fastapi.testclient import TestClient

class TestAssetTraining(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.registry = AssetRegistry()
        self.registry.filepath = "test_registry.json"
        if os.path.exists(self.registry.filepath):
            os.remove(self.registry.filepath)
        
        # Reset Queue Singleton
        TrainingQueue._instance = None

    def tearDown(self):
        if os.path.exists(self.registry.filepath):
            os.remove(self.registry.filepath)

    def test_asset_registry_crud(self):
        asset = AssetRecord(asset_type="character", name="Alice", tags=["test"])
        self.registry.register(asset)
        
        loaded = self.registry.get(asset.asset_id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.name, "Alice")
        
        self.registry.delete(asset.asset_id)
        self.assertIsNone(self.registry.get(asset.asset_id))

    def test_asset_versioning(self):
        versioning = AssetVersioning(self.registry)
        
        # Initially v1
        self.assertEqual(versioning.get_next_version("Bob", "character"), "v1")
        
        # Register v1
        self.registry.register(AssetRecord(asset_type="character", name="Bob", version="v1"))
        
        # Next should be v2
        self.assertEqual(versioning.get_next_version("Bob", "character"), "v2")

    def test_asset_search(self):
        self.registry.register(AssetRecord(asset_type="character", name="Charlie", tags=["red", "blue"]))
        self.registry.register(AssetRecord(asset_type="style", name="Cyberpunk", tags=["neon"]))
        
        search = AssetSearch(self.registry)
        
        results = search.search(asset_type="character")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "Charlie")
        
        results = search.search(tags=["neon"])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "Cyberpunk")
        
        results = search.search(query="char")
        self.assertEqual(len(results), 1)

    def test_dataset_validator(self):
        validator = DatasetValidator(min_images=1)
        
        # Valid path test
        # We need a dummy dir with an image
        os.makedirs("test_dataset", exist_ok=True)
        img = Image.new("RGB", (64, 64))
        img.save("test_dataset/1.png")
        
        report = validator.validate("test_dataset")
        self.assertTrue(report["valid"])
        self.assertEqual(report["total_images"], 1)
        
        # Clean up
        os.remove("test_dataset/1.png")
        os.rmdir("test_dataset")

    def test_auto_captioner(self):
        captioner = AutoCaptioner()
        
        os.makedirs("test_dataset", exist_ok=True)
        img = Image.new("RGB", (64, 64))
        img.save("test_dataset/1.png")
        
        captioner.process_dataset("test_dataset", prefix="Alice")
        
        self.assertTrue(os.path.exists("test_dataset/1.txt"))
        with open("test_dataset/1.txt", "r") as f:
            content = f.read()
            self.assertTrue(content.startswith("Alice"))
            
        # Clean up
        os.remove("test_dataset/1.png")
        os.remove("test_dataset/1.txt")
        os.rmdir("test_dataset")

    def test_training_queue_and_manager(self):
        manager = TrainingManager()
        # Override manager registry to test registry
        manager.registry = self.registry
        manager.versioning = AssetVersioning(self.registry)
        
        os.makedirs("test_dataset", exist_ok=True)
        img = Image.new("RGB", (64, 64))
        img.save("test_dataset/1.png")
        img.save("test_dataset/2.png")
        img.save("test_dataset/3.png")
        
        job_id = manager.start_training_job("test_dataset", "TestAsset", "style")
        
        queue = TrainingQueue()
        job = queue.get_job(job_id)
        self.assertIsNotNone(job)
        self.assertEqual(job["asset_name"], "TestAsset")
        
        # Wait a bit for the background thread to at least process captions
        time.sleep(2)
        
        job = queue.get_job(job_id)
        # It should be training by now
        self.assertIn(job["status"], ["Training", "Evaluating", "Completed"])
        
        # Wait for completion (training loop is ~5 secs)
        time.sleep(6)
        
        job = queue.get_job(job_id)
        self.assertEqual(job["status"], "Completed")
        
        # Verify asset was registered
        assets = self.registry.list_all()
        self.assertTrue(any(a.name == "TestAsset" for a in assets))
        
        # Clean up
        for f in os.listdir("test_dataset"):
            os.remove(os.path.join("test_dataset", f))
        os.rmdir("test_dataset")

if __name__ == "__main__":
    unittest.main()
