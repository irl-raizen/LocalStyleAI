import re
from typing import Optional, Dict, Any, List, Tuple
from src.memory.memory_store import MemoryStore
from src.memory.character_extractor import extract_character, CharacterMemory
from src.memory.scene_memory import SceneMemory, format_scene_for_prompt
from src.memory.style_memory import StyleMemory
from src.utils.helpers import get_logger

logger = get_logger("memory_engine")

class MemoryEngine:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MemoryEngine, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.char_store = MemoryStore("characters.json")
        self.scene_store = MemoryStore("scenes.json")
        self.style_store = MemoryStore("styles.json")

    # --- Character Memory ---

    def save_character(self, char_data: CharacterMemory):
        data = self.char_store.load()
        name_key = char_data.name.lower()
        data[name_key] = char_data.model_dump()
        self.char_store.save(data)
        logger.info("Character stored: %s", char_data.name)

    def get_character(self, name: str) -> Optional[CharacterMemory]:
        data = self.char_store.load()
        char_dict = data.get(name.lower())
        if char_dict:
            return CharacterMemory(**char_dict)
        return None

    def get_all_characters(self) -> List[Dict[str, Any]]:
        return list(self.char_store.load().values())

    def delete_character(self, name: str) -> bool:
        data = self.char_store.load()
        if name.lower() in data:
            del data[name.lower()]
            self.char_store.save(data)
            return True
        return False

    # --- Scene Memory ---

    def save_scene(self, scene_data: SceneMemory):
        data = self.scene_store.load()
        name_key = scene_data.scene_name.lower()
        data[name_key] = scene_data.model_dump()
        self.scene_store.save(data)
        logger.info("Scene stored: %s", scene_data.scene_name)

    def get_scene(self, name: str) -> Optional[SceneMemory]:
        data = self.scene_store.load()
        scene_dict = data.get(name.lower())
        if scene_dict:
            return SceneMemory(**scene_dict)
        return None

    def get_all_scenes(self) -> List[Dict[str, Any]]:
        return list(self.scene_store.load().values())
        
    def delete_scene(self, name: str) -> bool:
        data = self.scene_store.load()
        if name.lower() in data:
            del data[name.lower()]
            self.scene_store.save(data)
            return True
        return False

    # --- Style Memory ---

    def save_style(self, style_data: StyleMemory):
        # We store a single preferred style config for the user
        data = self.style_store.load()
        data["global"] = style_data.model_dump()
        self.style_store.save(data)
        logger.info("Style stored: %s", style_data.preferred_style)

    def get_style(self) -> Optional[StyleMemory]:
        data = self.style_store.load()
        if "global" in data:
            return StyleMemory(**data["global"])
        return None

    # --- Injection Logic ---

    def inject_memory(self, prompt: str, style: str) -> Tuple[str, str]:
        """
        Scans the prompt for known characters and scenes, injects their details,
        and extracts any new characters. Also uses preferred style if requested.
        Returns (injected_prompt, injected_style).
        """
        injected_prompt = prompt
        injected_style = style

        # 1. Check for preferred style
        if style == "default":
            pref_style = self.get_style()
            if pref_style and pref_style.preferred_style != "default":
                injected_style = pref_style.preferred_style
                logger.info("Memory injected: style -> %s", injected_style)
        else:
            # If user explicitly selected a style, update memory
            self.save_style(StyleMemory(preferred_style=style))

        # 2. Extract new character if described
        extracted_char = extract_character(prompt)
        if extracted_char:
            self.save_character(extracted_char)

        # 3. Inject known scenes
        scenes = self.scene_store.load()
        for scene_key, scene_dict in scenes.items():
            # If scene name is mentioned in prompt
            # Use regex for whole word match case-insensitive
            pattern = r'\b' + re.escape(scene_key) + r'\b'
            if re.search(pattern, injected_prompt, re.IGNORECASE):
                scene = SceneMemory(**scene_dict)
                scene_desc = format_scene_for_prompt(scene)
                if scene_desc:
                    injected_prompt += f", {scene_desc}"
                    logger.info("Memory injected: scene %s", scene.scene_name)

        # 4. Inject known characters
        characters = self.char_store.load()
        for char_key, char_dict in characters.items():
            pattern = r'\b' + re.escape(char_key) + r'\b'
            if re.search(pattern, injected_prompt, re.IGNORECASE):
                char = CharacterMemory(**char_dict)
                char_parts = []
                if char.age: char_parts.append(f"{char.age}-year-old")
                if char.hair: char_parts.append(char.hair)
                if char.eyes: char_parts.append(char.eyes + " eyes")
                if char.clothing: char_parts.append(char.clothing)
                if char.traits: char_parts.extend(char.traits)
                
                if char_parts:
                    char_desc = ", ".join(char_parts)
                    # Replace character name with name + description
                    # Example: "Sarah riding a motorcycle" -> "Sarah, 24-year-old, long red hair... riding a motorcycle"
                    replacement = f"{char.name}, {char_desc},"
                    injected_prompt = re.sub(pattern, replacement, injected_prompt, flags=re.IGNORECASE)
                    logger.info("Memory injected: character %s", char.name)

        return injected_prompt, injected_style
