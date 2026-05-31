import uuid
from typing import Dict, Any, List
from src.utils.helpers import get_logger

logger = get_logger("training_queue")

class TrainingQueue:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TrainingQueue, cls).__new__(cls)
            cls._instance.jobs = {}
        return cls._instance

    def add_job(self, dataset_path: str, asset_name: str, asset_type: str) -> str:
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = {
            "job_id": job_id,
            "dataset_path": dataset_path,
            "asset_name": asset_name,
            "asset_type": asset_type,
            "status": "Queued",
            "progress": 0.0,
            "logs": []
        }
        logger.info("Added training job %s for %s", job_id, asset_name)
        return job_id
        
    def update_job(self, job_id: str, status: str = None, progress: float = None, log: str = None):
        if job_id in self.jobs:
            if status: self.jobs[job_id]["status"] = status
            if progress is not None: self.jobs[job_id]["progress"] = progress
            if log: 
                self.jobs[job_id]["logs"].append(log)
                logger.info("[Job %s] %s", job_id, log)

    def get_job(self, job_id: str) -> Dict[str, Any]:
        return self.jobs.get(job_id)

    def list_jobs(self) -> List[Dict[str, Any]]:
        return list(self.jobs.values())
