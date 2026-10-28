import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from modules.feedback_manager import FeedbackManager as RealFeedbackManager
from modules.model_manager import model_manager


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
NAVIGATION_OPTIONS = [
    "💬 Chat",
    "💻 Code",
    "🎨 Images",
    "👁️ Vision",
    "📊 History",
    "⚙️ Settings",
]


class AppQualityTests(unittest.TestCase):
    def test_all_navigation_pages_render_without_provider_requests(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            app = AppTest.from_file(str(APP_PATH)).run(timeout=30)

            self.assertFalse(app.exception)
            navigation = next(widget for widget in app.radio if widget.label == "Navigation")
            self.assertEqual(list(navigation.options), NAVIGATION_OPTIONS)

            page_controls = {
                "💬 Chat": lambda current: any(
                    widget.label == "Chat workspace"
                    for widget in current.segmented_control
                ),
                "💻 Code": lambda current: any(
                    widget.label == "Code task" for widget in current.radio
                ),
                "🎨 Images": lambda current: any(
                    widget.label == "Prompt" for widget in current.text_area
                ),
                "👁️ Vision": lambda current: any(
                    widget.label == "Task Mode" for widget in current.selectbox
                ),
                "📊 History": lambda current: any(
                    widget.label == "Filter by Modality"
                    for widget in current.selectbox
                ),
                "⚙️ Settings": lambda current: any(
                    widget.label == "Active Text Model"
                    for widget in current.selectbox
                ),
            }
            for page in NAVIGATION_OPTIONS:
                app.session_state["nav_page"] = page
                app.run(timeout=30)
                self.assertFalse(app.exception, f"{page} page raised an app exception")
                navigation = next(
                    widget for widget in app.radio if widget.label == "Navigation"
                )
                self.assertEqual(navigation.value, page)
                self.assertTrue(page_controls[page](app), f"{page} controls are missing")

            self.assertTrue(
                any("GEMINI_API_KEY" in message.value for message in app.info)
            )

    def test_configured_api_key_is_not_rendered_or_stored_in_session(self):
        secret = "MOCK_NOT_A_REAL_GEMINI_KEY_1234567890"
        with patch.dict(os.environ, {"GEMINI_API_KEY": secret}):
            app = AppTest.from_file(str(APP_PATH)).run(timeout=30)

        self.assertFalse(app.exception)
        visible_elements = (
            list(app.markdown)
            + list(app.caption)
            + list(app.info)
            + list(app.success)
            + list(app.warning)
            + list(app.error)
            + list(app.text)
        )
        self.assertNotIn(secret, repr(visible_elements))
        self.assertNotIn(secret, repr(app.session_state))

    def test_chat_suggestion_and_mocked_response_are_persisted(self):
        secret = "MOCK_NOT_A_REAL_GEMINI_KEY_1234567890"
        with tempfile.TemporaryDirectory() as directory:
            history_file = Path(directory) / "history.json"

            def create_history_manager(_storage_path):
                return RealFeedbackManager(str(history_file))

            def mocked_generation(**kwargs):
                kwargs["on_chunk"]("A mocked ")
                kwargs["on_chunk"]("chat response.")
                return "A mocked chat response."

            with (
                patch.dict(os.environ, {"GEMINI_API_KEY": secret}),
                patch(
                    "modules.feedback_manager.FeedbackManager",
                    side_effect=create_history_manager,
                ),
                patch.object(
                    model_manager,
                    "generate_text_gemini",
                    side_effect=mocked_generation,
                ) as generate,
            ):
                app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
                suggestion = next(
                    button
                    for button in app.button
                    if button.key == "chat_suggestion_0"
                )
                suggestion.click().run(timeout=30)
                self.assertIn("Help me write a vivid opening", app.chat_input[0].value)

                app.chat_input[0].set_value("Explain this test result.").run(timeout=30)

            self.assertFalse(app.exception)
            self.assertEqual(generate.call_count, 1)
            self.assertTrue(
                any(
                    "A mocked chat response." in markdown.value
                    for markdown in app.markdown
                )
            )
            conversations = app.session_state.feedback_manager.get_chat_conversations()
            self.assertEqual(len(conversations), 1)
            messages = app.session_state.feedback_manager.get_chat_messages(
                conversations[0]["id"]
            )
            self.assertEqual(messages[0]["content"], "Explain this test result.")
            self.assertEqual(messages[1]["content"], "A mocked chat response.")


if __name__ == "__main__":
    unittest.main()
