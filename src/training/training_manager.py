import os
import threading
import time
from typing import Dict, Any, List

from src.training.dataset_validator import DatasetValidator
from src.training.auto_captioner import AutoCaptioner
from src.training.evaluation_engine import EvaluationEngine
from src.training.training_queue import TrainingQueue
from src.assets.asset_registry import AssetRegistry, AssetRecord
from src.assets.asset_versioning import AssetVersioning
from src.utils.helpers import get_logger

logger = get_logger("training_manager")

class TrainingManager:
    def __init__(self):
        self.validator = DatasetValidator(min_images=3) # set to 3 for easier testing
        self.captioner = AutoCaptioner()
        self.evaluator = EvaluationEngine()
        self.queue = TrainingQueue()
        self.registry = AssetRegistry()
        self.versioning = AssetVersioning(self.registry)

    def start_training_job(self, dataset_path: str, asset_name: str, asset_type: str, tags: List[str] = None):
        job_id = self.queue.add_job(dataset_path, asset_name, asset_type)
        
        # Run in a background thread to simulate asynchronous training
        thread = threading.Thread(target=self._run_job_pipeline, args=(job_id, dataset_path, asset_name, asset_type, tags))
        thread.daemon = True
        thread.start()
        
        return job_id

    def _run_job_pipeline(self, job_id: str, dataset_path: str, asset_name: str, asset_type: str, tags: List[str]):
        try:
            self.queue.update_job(job_id, status="Preparing", progress=0.1, log="Validating dataset...")
            validation_report = self.validator.validate(dataset_path)
            
            if not validation_report["valid"]:
                self.queue.update_job(job_id, status="Failed", log=f"Validation failed: {validation_report.get('error')}")
                return
                
            self.queue.update_job(job_id, progress=0.2, log="Auto-captioning dataset...")
            # Auto-captioning prefix could be the asset name
            self.captioner.process_dataset(dataset_path, prefix=asset_name)
            
            self.queue.update_job(job_id, status="Training", progress=0.3, log="Starting LoRA training...")
            # Mock training duration
            for i in range(4, 9):
                time.sleep(1) # Simulated epoch
                self.queue.update_job(job_id, progress=i/10.0, log=f"Training epoch {i-3}/5...")
                
            mock_lora_path = os.path.join(dataset_path, f"{asset_name}_lora.safetensors")
            # Create a dummy file
            with open(mock_lora_path, "w") as f: f.write("mock_lora_weights")
            
            self.queue.update_job(job_id, status="Evaluating", progress=0.9, log="Running Vision Critic evaluation...")
            eval_report = self.evaluator.evaluate(mock_lora_path, asset_type)
            overall_score = eval_report.get("overall_score", 0.0)
            
            self.queue.update_job(job_id, log=f"Evaluation score: {overall_score}")
            
            # Register Asset
            version = self.versioning.get_next_version(asset_name, asset_type)
            asset_record = AssetRecord(
                asset_type=asset_type,
                name=asset_name,
                version=version,
                lora_path=mock_lora_path,
                tags=tags or [],
                score=overall_score
            )
            self.registry.register(asset_record)
            
            self.queue.update_job(job_id, status="Completed", progress=1.0, log=f"Asset registered as {asset_name} {version}")
            
        except Exception as e:
            logger.error("Job %s failed: %s", job_id, e)
            self.queue.update_job(job_id, status="Failed", log=f"Error: {str(e)}")
