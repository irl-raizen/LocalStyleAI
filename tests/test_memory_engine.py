"""
Unit Tests for Phase 4: Memory Engine
"""
import os
import json
import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from api.app import app
from src.memory.memory_store import MemoryStore
from src.memory.memory_engine import MemoryEngine
from src.memory.character_extractor import CharacterMemory, extract_character
from src.memory.scene_memory import SceneMemory
from src.memory.style_memory import StyleMemory

class TestMemoryStore(unittest.TestCase):
    def setUp(self):
        self.test_file = "test_memory.json"
        self.store = MemoryStore(self.test_file)

    def tearDown(self):
        # Cleanup files
        if os.path.exists(self.store.filepath):
            os.remove(self.store.filepath)
        bak_path = self.store.filepath + ".corrupt.bak"
        if os.path.exists(bak_path):
            os.remove(bak_path)

    def test_save_and_load(self):
        data = {"key": "value"}
        self.store.save(data)
        loaded = self.store.load()
        self.assertEqual(data, loaded)

    def test_corrupted_json_recovery(self):
        # Write corrupted JSON
        with open(self.store.filepath, "w") as f:
            f.write("{invalid_json:")
            
        # Loading should trigger recovery
        loaded = self.store.load()
        self.assertEqual(loaded, {})
        
        # Check if backup was created
        self.assertTrue(os.path.exists(self.store.filepath + ".corrupt.bak"))

class TestCharacterExtractor(unittest.TestCase):
    @patch("src.memory.character_extractor.requests.post")
    def test_extract_character_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": '{"name": "Sarah", "age": "24", "hair": "long red hair"}'
        }
        mock_post.return_value = mock_response

        char = extract_character("Sarah, a 24-year-old woman with long red hair")
        self.assertIsNotNone(char)
        self.assertEqual(char.name, "Sarah")
        self.assertEqual(char.age, "24")

    @patch("src.memory.character_extractor.requests.post")
    def test_extract_character_empty(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": '{}'
        }
        mock_post.return_value = mock_response

        char = extract_character("A generic landscape")
        self.assertIsNone(char)

class TestMemoryEngine(unittest.TestCase):
    def setUp(self):
        MemoryEngine._instance = None
        self.engine = MemoryEngine()
        # Override file paths for testing
        self.engine.char_store = MemoryStore("test_chars.json")
        self.engine.scene_store = MemoryStore("test_scenes.json")
        self.engine.style_store = MemoryStore("test_styles.json")

    def tearDown(self):
        for store in [self.engine.char_store, self.engine.scene_store, self.engine.style_store]:
            if os.path.exists(store.filepath): os.remove(store.filepath)

    def test_character_persistence(self):
        char = CharacterMemory(name="John", age="30")
        self.engine.save_character(char)
        
        loaded_char = self.engine.get_character("John")
        self.assertIsNotNone(loaded_char)
        self.assertEqual(loaded_char.name, "John")

        self.engine.delete_character("John")
        self.assertIsNone(self.engine.get_character("John"))

    def test_scene_persistence(self):
        scene = SceneMemory(scene_name="Tokyo Street", environment="cyberpunk", lighting="neon")
        self.engine.save_scene(scene)
        
        loaded_scene = self.engine.get_scene("Tokyo Street")
        self.assertIsNotNone(loaded_scene)
        self.assertEqual(loaded_scene.lighting, "neon")

    def test_style_persistence(self):
        style = StyleMemory(preferred_style="anime_clean")
        self.engine.save_style(style)
        
        loaded_style = self.engine.get_style()
        self.assertIsNotNone(loaded_style)
        self.assertEqual(loaded_style.preferred_style, "anime_clean")

    @patch("src.memory.memory_engine.extract_character")
    def test_inject_memory(self, mock_extract):
        mock_extract.return_value = None
        
        # Setup memory
        char = CharacterMemory(name="Sarah", age="24", hair="red")
        self.engine.save_character(char)
        scene = SceneMemory(scene_name="Tokyo", environment="city", lighting="neon")
        self.engine.save_scene(scene)
        
        prompt = "Sarah walking in Tokyo"
        injected_prompt, injected_style = self.engine.inject_memory(prompt, "default")
        
        self.assertIn("Sarah", injected_prompt)
        self.assertIn("24-year-old", injected_prompt)
        self.assertIn("red", injected_prompt)
        self.assertIn("city", injected_prompt)
        self.assertIn("neon lighting", injected_prompt)

class TestMemoryEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        MemoryEngine._instance = None
        self.engine = MemoryEngine()
        self.engine.char_store = MemoryStore("test_api_chars.json")

    def tearDown(self):
        if os.path.exists(self.engine.char_store.filepath):
            os.remove(self.engine.char_store.filepath)

    def test_get_characters(self):
        self.engine.save_character(CharacterMemory(name="Alice"))
        response = self.client.get("/memory/characters")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(any(c["name"] == "Alice" for c in data["characters"]))

    def test_delete_character(self):
        self.engine.save_character(CharacterMemory(name="Bob"))
        response = self.client.delete("/memory/characters/Bob")
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(self.engine.get_character("Bob"))

if __name__ == "__main__":
    unittest.main()
