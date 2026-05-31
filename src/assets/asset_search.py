from typing import List, Optional
from src.assets.asset_registry import AssetRegistry, AssetRecord

class AssetSearch:
    def __init__(self, registry: AssetRegistry):
        self.registry = registry

    def search(
        self, 
        query: str = "", 
        asset_type: str = "", 
        tags: List[str] = None, 
        min_score: float = 0.0
    ) -> List[AssetRecord]:
        
        all_assets = self.registry.list_all()
        results = []
        
        query = query.lower()
        tags = [t.lower() for t in tags] if tags else []
        asset_type = asset_type.lower()
        
        for asset in all_assets:
            if min_score > 0 and asset.score < min_score:
                continue
                
            if asset_type and asset.asset_type.lower() != asset_type:
                continue
                
            if tags:
                asset_tags = [t.lower() for t in asset.tags]
                if not any(t in asset_tags for t in tags):
                    continue
                    
            if query:
                if query not in asset.name.lower() and query not in " ".join(asset.tags).lower():
                    continue
                    
            results.append(asset)
            
        # Sort by score descending
        results.sort(key=lambda a: a.score, reverse=True)
        return results
