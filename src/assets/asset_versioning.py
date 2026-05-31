from typing import List
from src.assets.asset_registry import AssetRegistry, AssetRecord

class AssetVersioning:
    def __init__(self, registry: AssetRegistry):
        self.registry = registry

    def get_next_version(self, name: str, asset_type: str) -> str:
        """
        Determines the next version string for an asset (e.g. 'v1', 'v2').
        Never overwrites previous versions.
        """
        all_assets = self.registry.list_all()
        # Find all assets matching name and type
        matching = [a for a in all_assets if a.name.lower() == name.lower() and a.asset_type == asset_type]
        
        if not matching:
            return "v1"
            
        max_v = 0
        for m in matching:
            v_str = m.version.lower().replace("v", "")
            try:
                v_num = int(v_str)
                if v_num > max_v: max_v = v_num
            except ValueError:
                pass
                
        return f"v{max_v + 1}"
