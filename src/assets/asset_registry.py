import os
import json
import uuid
import shutil
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from src.utils.helpers import get_logger

logger = get_logger("asset_registry")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "assets")

class AssetRecord(BaseModel):
    asset_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    asset_type: str = Field(..., description="Character, Style, Product, Object, Environment, Custom")
    name: str = Field(...)
    version: str = Field(default="v1")
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    preview_image: str = Field(default="")
    lora_path: str = Field(default="")
    tags: List[str] = Field(default_factory=list)
    score: float = Field(default=0.0)

class AssetRegistry:
    def __init__(self):
        self.filepath = os.path.join(DATA_DIR, "registry.json")
        self._ensure_exists()

    def _ensure_exists(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        if not os.path.exists(self.filepath):
            self.save({})

    def load(self) -> Dict[str, dict]:
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("Registry corruption detected: %s", e)
            backup_path = self.filepath + ".corrupt.bak"
            shutil.copy2(self.filepath, backup_path)
            self.save({})
            return {}
        except Exception as e:
            logger.error("Error loading registry: %s", e)
            return {}

    def save(self, data: Dict[str, dict]):
        try:
            temp_path = self.filepath + ".tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            os.replace(temp_path, self.filepath)
        except Exception as e:
            logger.error("Failed to save registry: %s", e)

    def register(self, asset: AssetRecord):
        data = self.load()
        data[asset.asset_id] = asset.model_dump()
        self.save(data)
        logger.info("Registered asset: %s (%s)", asset.name, asset.version)

    def get(self, asset_id: str) -> Optional[AssetRecord]:
        data = self.load()
        if asset_id in data:
            return AssetRecord(**data[asset_id])
        return None

    def delete(self, asset_id: str) -> bool:
        data = self.load()
        if asset_id in data:
            del data[asset_id]
            self.save(data)
            logger.info("Deleted asset: %s", asset_id)
            return True
        return False

    def list_all(self) -> List[AssetRecord]:
        data = self.load()
        return [AssetRecord(**item) for item in data.values()]
