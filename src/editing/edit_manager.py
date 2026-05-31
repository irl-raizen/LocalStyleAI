from PIL import Image
from typing import Tuple, Dict, Any
import json

from src.editing.scene_analyzer import SceneAnalyzer
from src.editing.edit_interpreter import interpret_edit
from src.editing.region_selector import RegionSelector
from src.editing.inpainting_engine import InpaintingEngine
from src.critic.prompt_matcher import evaluate_image_against_prompt
from src.utils.helpers import get_logger

logger = get_logger("edit_manager")

class EditManager:
    def __init__(self):
        self.analyzer = SceneAnalyzer()
        self.selector = RegionSelector()
        self.inpaint = InpaintingEngine()

    def process_edit(self, image: Image.Image, instruction: str) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Processes an edit instruction on an existing image.
        Returns the edited image and metadata.
        """
        logger.info("Starting edit process for instruction: '%s'", instruction)
        
        # 1. Analyze Scene
        scene = self.analyzer.analyze(image)
        logger.info("Scene analysis: %s", scene)
        
        # 2. Interpret Instruction
        plan = interpret_edit(instruction)
        if not plan:
            logger.warning("Could not interpret edit plan. Aborting edit.")
            return image, {"error": "Failed to interpret instruction", "critic_score": 0.0}
            
        target = plan.target
        action = plan.action
        value = plan.value
        
        # Construct an inpainting prompt based on the action
        if action == "replace":
            inpaint_prompt = f"A high quality {value}"
        elif action == "modify":
            inpaint_prompt = f"A {value} {target}"
        elif action == "add":
            inpaint_prompt = f"Add {value}"
            target = "environment" # Add to the environment
        elif action == "remove":
            inpaint_prompt = "background, empty space, matching surroundings"
        elif action == "environment":
            inpaint_prompt = f"A {value} scene"
            target = "environment"
        else:
            inpaint_prompt = f"{value} {target}"

        # 3. Region Selection
        region = self.selector.locate_and_mask(image, target)
        mask = region["mask"]
        confidence = region["confidence"]
        
        # 4. Inpainting
        logger.info("Executing inpainting with prompt: '%s'", inpaint_prompt)
        edited_image = self.inpaint.apply_edit(image, mask, inpaint_prompt)
        
        # 5. Vision Critic Validation
        # Verify that the edit was successfully applied
        validation_prompt = inpaint_prompt if action != "remove" else "empty"
        evaluation = evaluate_image_against_prompt(edited_image, validation_prompt)
        score = evaluation.get("overall_score", 1.0)
        
        logger.info("Edit critic score: %.2f", score)
        
        metadata = {
            "edit_instruction": instruction,
            "target": target,
            "action": action,
            "mask_confidence": confidence,
            "critic_score": score,
            "scene_analysis": scene,
            "edit_plan": plan.model_dump()
        }
        
        return edited_image, metadata
