import os
from typing import Dict, Any
from PIL import Image

from src.critic.prompt_matcher import evaluate_image_against_prompt
from src.utils.helpers import get_logger

logger = get_logger("regeneration_manager")

class RegenerationManager:
    def __init__(self):
        self.max_attempts = int(os.environ.get("MAX_REGEN_ATTEMPTS", 3))
        self.min_score = float(os.environ.get("MIN_ACCEPTABLE_SCORE", 0.75))

    def repair_prompt(self, original_prompt: str, missing_elements: list[str]) -> str:
        """
        Appends repair hints to the prompt based on missing elements.
        """
        if not missing_elements:
            return original_prompt
            
        repair_hints = ", ".join(missing_elements)
        # We append a strong hint to the prompt
        repaired = f"{original_prompt}, IMPORTANT: Clearly visible {repair_hints}"
        logger.info("Prompt repaired -> %s", repaired)
        return repaired

    def run_generation_loop(self, generate_func, prompt: str, **kwargs) -> tuple[Image.Image, Dict[str, Any]]:
        """
        Runs the generation function, evaluates the output, and retries if necessary.
        generate_func should be a callable that takes (prompt, **kwargs) and returns a PIL Image.
        """
        attempts = 0
        current_prompt = prompt
        best_image = None
        best_score = -1.0
        best_eval = {}

        logger.info("Critic generation loop started for prompt: '%s'", prompt)

        while attempts < self.max_attempts:
            attempts += 1
            logger.info("Generation attempt %d/%d", attempts, self.max_attempts)
            
            # Generate
            image = generate_func(current_prompt, **kwargs)
            
            # Evaluate
            evaluation = evaluate_image_against_prompt(image, current_prompt)
            score = evaluation.get("overall_score", 1.0)
            
            logger.info("Critic score: %.2f", score)
            
            if evaluation.get("missing_elements"):
                logger.info("Missing elements detected: %s", evaluation["missing_elements"])

            if score > best_score:
                best_score = score
                best_image = image
                best_eval = evaluation
            
            # If score is acceptable or no critic available, break
            if score >= self.min_score or evaluation.get("critic_unavailable", False):
                if evaluation.get("critic_unavailable", False):
                    logger.warning("Critic unavailable. Skipping regeneration loop.")
                else:
                    logger.info("Score %.2f >= %.2f. Acceptance threshold met.", score, self.min_score)
                break
                
            # If missing elements are detected and we haven't reached max attempts, repair
            if attempts < self.max_attempts and evaluation.get("missing_elements"):
                logger.info("Score below threshold. Triggering regeneration...")
                current_prompt = self.repair_prompt(current_prompt, evaluation["missing_elements"])

        # Add metadata to evaluation results
        best_eval["attempts"] = attempts
        logger.info("Generation loop completed. Final score: %.2f after %d attempts.", best_score, attempts)
        
        return best_image, best_eval
