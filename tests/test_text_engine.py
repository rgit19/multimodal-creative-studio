import unittest
from unittest.mock import patch

from modules.text_engine import TextEngine


class TextEngineTests(unittest.TestCase):
    @patch("modules.text_engine.model_manager.generate_text_gemini")
    def test_story_generation_uses_mocked_provider_response(self, generate):
        generate.return_value = (
            "Chapter 1: First Signal\n"
            "A distant beacon appeared above the horizon.\n\n"
            "Chapter 2: The Reply\n"
            "The beacon answered in a familiar voice."
        )
        def on_chunk(_chunk):
            pass

        result = TextEngine().generate_story(
            api_key="mock-key",
            seed_prompt="A beacon appears",
            num_chapters=2,
            model="mock-model",
            on_chunk=on_chunk,
        )

        self.assertEqual(
            result["chapters"][0]["text"],
            "A distant beacon appeared above the horizon.",
        )
        self.assertEqual(result["model_used"], "mock-model")
        self.assertEqual(generate.call_args.kwargs["api_key"], "mock-key")
        self.assertEqual(generate.call_args.kwargs["model"], "mock-model")
        self.assertIs(generate.call_args.kwargs["on_chunk"], on_chunk)


if __name__ == "__main__":
    unittest.main()
