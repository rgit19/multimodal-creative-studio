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

    def add_chat_turn(
        self,
        conversation_id: str,
        title: str,
        turn_index: int,
        prompt: str,
        response: str,
        model_name: str,
    ) -> str:
        """Persist a completed chat turn in the existing generation history."""
        record_id = self.add_generation(
            modality="chat",
            prompt=prompt,
            output=response,
            model_name=model_name,
            parameters={
                "conversation_id": conversation_id,
                "conversation_title": title,
                "turn_index": turn_index,
                "updated_at": datetime.now().astimezone().isoformat(),
            },
        )
        return record_id

    def get_chat_conversations(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Return recent saved conversations, newest first."""
        conversations: Dict[str, Dict[str, Any]] = {}
        for item in self.history:
            parameters = item.get("parameters") or {}
            conversation_id = parameters.get("conversation_id")
            if item.get("modality") != "chat" or not conversation_id:
                continue

            updated_at = parameters.get("updated_at", item.get("timestamp", ""))
            current = conversations.get(conversation_id)
            if current is None or updated_at > current["updated_at"]:
                conversations[conversation_id] = {
                    "id": conversation_id,
                    "title": parameters.get("conversation_title") or item["prompt"][:48],
                    "updated_at": updated_at,
                    "preview": item["prompt"],
                }

        return sorted(
            conversations.values(),
            key=lambda conversation: conversation["updated_at"],
            reverse=True,
        )[:limit]

    def get_chat_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Load a conversation as ordered user and assistant messages."""
        turns = [
            item
            for item in self.history
            if item.get("modality") == "chat"
            and (item.get("parameters") or {}).get("conversation_id")
            == conversation_id
        ]
        turns.sort(key=lambda item: (item.get("parameters") or {}).get("turn_index", 0))

        messages: List[Dict[str, Any]] = []
        for turn in turns:
            messages.extend(
                [
                    {"role": "user", "content": turn["prompt"]},
                    {
                        "role": "assistant",
                        "content": str(turn["output"]),
                        "record_id": turn["id"],
                    },
                ]
            )
        return messages

    def rename_chat_conversation(self, conversation_id: str, title: str) -> bool:
        """Rename all history records associated with one conversation."""
        changed = False
        for item in self.history:
            parameters = item.get("parameters") or {}
            if (
                item.get("modality") == "chat"
                and parameters.get("conversation_id") == conversation_id
            ):
                parameters["conversation_title"] = title
                changed = True
        if changed:
            self._save_history()
        return changed

    def delete_chat_conversation(self, conversation_id: str) -> bool:
        """Delete only the selected chat conversation from generation history."""
        remaining = [
            item
            for item in self.history
            if not (
                item.get("modality") == "chat"
                and (item.get("parameters") or {}).get("conversation_id")
                == conversation_id
            )
        ]
        changed = len(remaining) != len(self.history)
        if changed:
            self.history = remaining
            self._save_history()
        return changed

    def update_chat_turn(
        self,
        record_id: str,
        response: str,
        model_name: str,
    ) -> bool:
        """Replace an assistant response when the user retries a chat turn."""
        for item in self.history:
            if item.get("id") == record_id and item.get("modality") == "chat":
                item["output"] = response
                item["model_name"] = model_name
                item.setdefault("parameters", {})["updated_at"] = (
                    datetime.now().astimezone().isoformat()
                )
                self._save_history()
                return True
        return False

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
