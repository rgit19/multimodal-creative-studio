"""
OmniCraft Studio - Feedback & History Management Module
Tracks generation provenance, user ratings, thumbs up/down, regeneration cycles, and persistent history.
"""

import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any


class FeedbackManager:
    def __init__(self, storage_path: str = "data/history.json"):
        self.storage_path = storage_path
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        self.history: List[Dict[str, Any]] = self._load_history()

    def _load_history(self) -> List[Dict[str, Any]]:
        """Load persistent history from JSON file."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save_history(self) -> None:
        """Write in-memory history to disk safely."""
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving history: {e}")

    def add_generation(
        self,
        modality: str,
        prompt: str,
        output: Any,
        model_name: str,
        parameters: Optional[Dict[str, Any]] = None,
        is_regeneration: bool = False,
        parent_id: Optional[str] = None
    ) -> str:
        """Record a new generation event."""
        record_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        record = {
            "id": record_id,
            "timestamp": timestamp,
            "modality": modality,
            "prompt": prompt,
            "output": output,
            "model_name": model_name,
            "parameters": parameters or {},
            "rating": None,
            "feedback_type": None,  # "thumbs_up" or "thumbs_down"
            "notes": "",
            "is_regeneration": is_regeneration,
            "parent_id": parent_id
        }

        self.history.insert(0, record)  # Newest first
        self._save_history()
        return record_id

    def update_feedback(
        self,
        record_id: str,
        rating: Optional[int] = None,
        feedback_type: Optional[str] = None,
        notes: Optional[str] = None
    ) -> bool:
        """Update rating and feedback for a generated item."""
        for item in self.history:
            if item["id"] == record_id:
                if rating is not None:
                    item["rating"] = rating
                if feedback_type is not None:
                    item["feedback_type"] = feedback_type
                if notes is not None:
                    item["notes"] = notes
                self._save_history()
                return True
        return False

    def get_history(self, limit: int = 50, modality: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve filtered history items."""
        if modality:
            filtered = [item for item in self.history if item.get("modality") == modality]
            return filtered[:limit]
        return self.history[:limit]

    def get_record(self, record_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific record by its ID."""
        for item in self.history:
            if item["id"] == record_id:
                return item
        return None

    def clear_history(self) -> None:
        """Clear all stored history records."""
        self.history = []
        self._save_history()

    def get_statistics(self) -> Dict[str, Any]:
        """Compute aggregated statistics for the project dashboard."""
        total = len(self.history)
        if total == 0:
            return {
                "total_generations": 0,
                "ratings_count": 0,
                "avg_rating": 0.0,
                "thumbs_up": 0,
                "thumbs_down": 0,
                "modality_breakdown": {}
            }

        rated_items = [item["rating"] for item in self.history if item.get("rating") is not None]
        avg_rating = sum(rated_items) / len(rated_items) if rated_items else 0.0
        thumbs_up = sum(1 for item in self.history if item.get("feedback_type") == "thumbs_up")
        thumbs_down = sum(1 for item in self.history if item.get("feedback_type") == "thumbs_down")

        breakdown = {}
        for item in self.history:
            m = item.get("modality", "other")
            breakdown[m] = breakdown.get(m, 0) + 1

        return {
            "total_generations": total,
            "ratings_count": len(rated_items),
            "avg_rating": round(avg_rating, 2),
            "thumbs_up": thumbs_up,
            "thumbs_down": thumbs_down,
            "modality_breakdown": breakdown
        }
