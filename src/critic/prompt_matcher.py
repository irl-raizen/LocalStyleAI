import re
from typing import List, Dict, Any
from PIL import Image

from src.critic.critic_models import CriticFactory
from src.utils.helpers import get_logger

logger = get_logger("prompt_matcher")

def extract_key_elements(prompt: str) -> List[str]:
    """
    Very naive extraction of noun phrases / key elements from a prompt.
    In a full production environment, this would use an NLP parser (like spaCy) 
    or an LLM call to extract specific objects.
    Here we split by commas and strip out common stop words to get chunks.
    """
    # Split by comma
    chunks = [c.strip() for c in prompt.split(",")]
    elements = []
    
    stop_words = {"a", "an", "the", "with", "and", "or", "of", "in", "on", "at", "to"}
    
    for chunk in chunks:
        if not chunk: continue
        # Simple cleanup
        words = chunk.split()
        filtered_words = [w for w in words if w.lower() not in stop_words]
        if filtered_words:
            elements.append(" ".join(filtered_words))
            
    return elements

def evaluate_image_against_prompt(image: Image.Image, prompt: str) -> Dict[str, Any]:
    """
    Extracts key elements from the prompt and compares them against the image
    using the active vision critic model.
    """
    critic = CriticFactory.get_critic()
    
    if critic is None:
        return {
            "overall_score": 1.0,
            "prompt_match": 1.0,
            "style_match": 1.0,
            "composition_score": 1.0,
            "missing_elements": [],
            "matched_elements": [],
            "critic_unavailable": True
        }
        
    elements = extract_key_elements(prompt)
    if not elements:
        return {
            "overall_score": 1.0,
            "prompt_match": 1.0,
            "style_match": 1.0,
            "composition_score": 1.0,
            "missing_elements": [],
            "matched_elements": []
        }
        
    # We ask the critic how confident it is that these elements exist in the image
    scores = critic.calculate_similarity(image, elements)
    
    missing_elements = []
    matched_elements = []
    
    # Let's say a score >= 0.1 for SigLIP / CLIP indicates presence (can be tuned)
    # Actually, siglip probabilities are somewhat strict, we can use a threshold of 0.1 for missing.
    THRESHOLD = 0.1
    
    for element, score in zip(elements, scores):
        if score < THRESHOLD:
            missing_elements.append(element)
        else:
            matched_elements.append(element)
            
    prompt_match = sum(scores) / len(scores) if scores else 0.0
    
    # We can also compute an overall match against the full prompt
    full_score = critic.calculate_similarity(image, [prompt])[0]
    
    # Simple overall score heuristic
    overall_score = (prompt_match + full_score) / 2.0
    
    return {
        "overall_score": round(overall_score, 3),
        "prompt_match": round(prompt_match, 3),
        "style_match": round(full_score, 3), # simplified
        "composition_score": round(full_score, 3),
        "missing_elements": missing_elements,
        "matched_elements": matched_elements
    }
