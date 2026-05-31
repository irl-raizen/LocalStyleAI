import os
import json
import shutil
from typing import Any, Dict
from src.utils.helpers import get_logger

logger = get_logger("memory_store")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "memory")

class MemoryStore:
    """
    Handles JSON persistence for memory files with auto-creation and corruption recovery.
    """
    def __init__(self, filename: str):
        self.filepath = os.path.join(DATA_DIR, filename)
        self._ensure_exists()

    def _ensure_exists(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        if not os.path.exists(self.filepath):
            self.save({})

    def load(self) -> Dict[str, Any]:
        """Load data from JSON file with corruption recovery."""
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("Corruption detected in %s: %s", self.filepath, e)
            # Backup corrupted file
            backup_path = self.filepath + ".corrupt.bak"
            shutil.copy2(self.filepath, backup_path)
            logger.info("Corrupted file backed up to %s", backup_path)
            
            # Create a clean file
            self.save({})
            return {}
        except Exception as e:
            logger.error("Error loading %s: %s", self.filepath, e)
            return {}

    def save(self, data: Dict[str, Any]):
        """Save data to JSON file."""
        try:
            # Write to a temporary file first for atomic-like save
            temp_path = self.filepath + ".tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            # Replace the actual file
            os.replace(temp_path, self.filepath)
        except Exception as e:
            logger.error("Failed to save memory to %s: %s", self.filepath, e)
