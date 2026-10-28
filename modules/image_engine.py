"""
OmniCraft Studio - Gemini native image generation engine.
"""

import io
import time
from typing import Any, Dict, Optional, Tuple

from PIL import Image

from modules.model_manager import IMAGE_MODEL, model_manager


STYLE_PRESETS = {
    "Photorealistic": "photorealistic, natural cinematic lighting, highly detailed",
    "Cyberpunk / Neon": "cyberpunk aesthetic, neon magenta and cyan lighting, rainy reflective streets",
    "Anime / Studio Ghibli": "hand-painted anime aesthetic, soft watercolor textures, lush vibrant environment",
    "Watercolor Fantasy": "ethereal fantasy watercolor, delicate ink details, soft dreamlike color bleeds",
    "Modern 3D Minimalist": "isometric 3D render, smooth geometric forms, soft pastel studio lighting",
    "Oil Painting Classic": "classic oil painting, textured brushstrokes, chiaroscuro lighting, rich warm palette",
}

ASPECT_RATIOS = {
    "1:1 Square": "1:1",
    "16:9 Landscape": "16:9",
    "9:16 Portrait / Reel": "9:16",
    "4:3 Classic": "4:3",
}


class ImageEngine:
    """Generate images using Google's Gemini image model."""

    def generate_image(
        self,
        api_key: str,
        prompt: str,
        negative_prompt: str = "",
        style_preset: str = "Photorealistic",
        aspect_ratio_key: str = "1:1 Square",
        model: Optional[str] = None,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """Generate a 1K image from a prompt with Gemini."""
        if not prompt.strip():
            raise ValueError("Enter a prompt before generating an image.")

        aspect_ratio = ASPECT_RATIOS.get(aspect_ratio_key)
        if aspect_ratio is None:
            raise ValueError(f"Unsupported image aspect ratio: {aspect_ratio_key}")

        style = STYLE_PRESETS.get(style_preset)
        if style is None:
            raise ValueError(f"Unsupported image style: {style_preset}")

        prompt_parts = [
            prompt.strip(),
            f"Visual style: {style}.",
            f"Compose the image in a {aspect_ratio} aspect ratio.",
        ]
        if negative_prompt.strip():
            prompt_parts.append(f"Avoid including: {negative_prompt.strip()}.")

        full_prompt = " ".join(prompt_parts)
        active_model = model or IMAGE_MODEL
        start_time = time.time()
        image_bytes = model_manager.generate_image_gemini(
            prompt=full_prompt,
            aspect_ratio=aspect_ratio,
            api_key=api_key,
            model=active_model,
        )
        with Image.open(io.BytesIO(image_bytes)) as generated:
            image = generated.convert("RGB")

        return image, {
            "source_model": active_model,
            "prompt": prompt.strip(),
            "full_prompt": full_prompt,
            "negative_prompt": negative_prompt.strip(),
            "style_preset": style_preset,
            "aspect_ratio": aspect_ratio,
            "width": image.width,
            "height": image.height,
            "generation_time_ms": int((time.time() - start_time) * 1000),
        }


image_engine = ImageEngine()
