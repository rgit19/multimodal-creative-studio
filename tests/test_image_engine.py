import io
import typing
import unittest
from unittest.mock import patch

from PIL import Image

from modules.image_engine import ImageEngine


class ImageEngineTests(unittest.TestCase):
    def test_public_method_annotations_resolve(self):
        typing.get_type_hints(ImageEngine.generate_image)

    def test_generate_image_uses_mocked_provider_image(self):
        image_buffer = io.BytesIO()
        Image.new("RGB", (3, 2), color="blue").save(image_buffer, format="PNG")

        with patch(
            "modules.image_engine.model_manager.generate_image_gemini",
            return_value=image_buffer.getvalue(),
        ) as generate:
            image, metadata = ImageEngine().generate_image(
                api_key="mock-key",
                prompt="A blue geometric landscape",
                model="mock-image-model",
            )

        self.assertEqual(image.size, (3, 2))
        self.assertEqual(metadata["source_model"], "mock-image-model")
        self.assertEqual(generate.call_args.kwargs["api_key"], "mock-key")

    def test_invalid_provider_image_is_reported(self):
        with patch(
            "modules.image_engine.model_manager.generate_image_gemini",
            return_value=b"not an image",
        ):
            with self.assertRaises((OSError, ValueError)):
                ImageEngine().generate_image(
                    api_key="mock-key",
                    prompt="A blue geometric landscape",
                )

    def test_provider_errors_are_propagated(self):
        with patch(
            "modules.image_engine.model_manager.generate_image_gemini",
            side_effect=RuntimeError("sanitized provider failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "sanitized provider failure"):
                ImageEngine().generate_image(
                    api_key="mock-key",
                    prompt="A blue geometric landscape",
                )


if __name__ == "__main__":
    unittest.main()
