import typing
import unittest
from unittest.mock import patch

from PIL import Image

from modules.multimodal_vision import VisionEngine


class VisionEngineTests(unittest.TestCase):
    def test_public_method_annotations_resolve(self):
        typing.get_type_hints(VisionEngine.analyze_image_and_generate)

    def test_analysis_uses_mocked_provider_response(self):
        image = Image.new("RGB", (4, 3), color="purple")

        with patch(
            "modules.multimodal_vision.model_manager.analyze_image_gemini",
            return_value="A purple test image.",
        ) as analyze:
            result = VisionEngine().analyze_image_and_generate(
                api_key="mock-key",
                image=image,
                model="mock-text-model",
            )

        self.assertEqual(result["analysis"], "A purple test image.")
        self.assertEqual(result["features"]["dimensions"], "4x3")
        self.assertEqual(analyze.call_args.kwargs["api_key"], "mock-key")

    def test_provider_errors_are_propagated(self):
        with patch(
            "modules.multimodal_vision.model_manager.analyze_image_gemini",
            side_effect=RuntimeError("sanitized provider failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "sanitized provider failure"):
                VisionEngine().analyze_image_and_generate(
                    api_key="mock-key",
                    image=Image.new("RGB", (2, 2)),
                )


if __name__ == "__main__":
    unittest.main()
