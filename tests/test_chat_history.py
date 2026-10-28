import tempfile
import unittest
from pathlib import Path

from modules.feedback_manager import FeedbackManager


class ChatHistoryTests(unittest.TestCase):
    def test_chat_turns_can_be_reopened_renamed_and_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.json"
            manager = FeedbackManager(str(path))
            record_id = manager.add_chat_turn(
                conversation_id="conversation-1",
                title="A chat about testing",
                turn_index=0,
                prompt="How should I test this function?",
                response="Start with a focused unit test.",
                model_name="mock-model",
            )

            conversations = manager.get_chat_conversations()
            self.assertEqual(conversations[0]["id"], "conversation-1")
            self.assertEqual(conversations[0]["title"], "A chat about testing")
            self.assertEqual(
                manager.get_chat_messages("conversation-1"),
                [
                    {"role": "user", "content": "How should I test this function?"},
                    {
                        "role": "assistant",
                        "content": "Start with a focused unit test.",
                        "record_id": record_id,
                    },
                ],
            )

            manager.update_chat_turn(record_id, "Use a mock for the dependency.", "mock-model")
            self.assertEqual(
                manager.get_chat_messages("conversation-1")[-1]["content"],
                "Use a mock for the dependency.",
            )

            self.assertTrue(
                manager.rename_chat_conversation("conversation-1", "Unit test help")
            )
            self.assertEqual(
                manager.get_chat_conversations()[0]["title"], "Unit test help"
            )
            self.assertTrue(manager.delete_chat_conversation("conversation-1"))
            self.assertEqual(manager.get_chat_conversations(), [])

            reopened = FeedbackManager(str(path))
            self.assertEqual(reopened.get_chat_conversations(), [])


if __name__ == "__main__":
    unittest.main()
