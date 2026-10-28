"""
OmniCraft Studio - Gemini-powered image understanding.
"""

import io
import time
from typing import Any, Dict, Optional

from PIL import Image, ImageStat

from modules.model_manager import TEXT_MODEL, model_manager


class VisionEngine:
    """Analyze images with Gemini and calculate local image measurements."""

    def analyze_image_and_generate(
        self,
        api_key: str,
        image: Image.Image,
        task_mode: str = "Marketing Copy & Social Media",
        custom_instructions: str = "",
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Analyze an image with Gemini and report its perceptual measurements."""
        start_time = time.time()
        active_model = model or TEXT_MODEL
        buffered = io.BytesIO()
        image.convert("RGB").save(buffered, format="JPEG")
        features = self._extract_image_features(image)
        analysis_text = model_manager.analyze_image_gemini(
            image_bytes=buffered.getvalue(),
            mime_type="image/jpeg",
            prompt=self._build_task_prompt(task_mode, custom_instructions),
            api_key=api_key,
            model=active_model,
        )

        return {
            "task_mode": task_mode,
            "analysis": analysis_text,
            "features": features,
            "model_used": f"{active_model} Vision",
            "generation_time_ms": int((time.time() - start_time) * 1000),
        }

    @staticmethod
    def _build_task_prompt(task_mode: str, custom_instructions: str) -> str:
        """Build the Gemini prompt for the selected image-analysis task."""
        prompts = {
            "Marketing Copy & Social Media": (
                "Analyze this image. Identify its subject, aesthetic, and mood. "
                "Generate a catchy headline, Instagram and LinkedIn captions, a call "
                "to action, and 10 relevant hashtags."
            ),
            "Technical Diagram & Architecture Deconstruction": (
                "Analyze this architecture diagram or visual layout. Identify "
                "components and data flows, then provide an architecture summary, "
                "component breakdown, and a proposed implementation outline."
            ),
            "Visual Story Starter": (
                "Study the scene, lighting, and elements in this image. Write an "
                "atmospheric opening chapter set in this visual world."
            ),
            "Accessibility & Detailed Visual Audit": (
                "Describe this image accurately for accessibility, then audit its "
                "colors, lighting, and composition."
            ),
        }
        prompt = prompts.get(
            task_mode,
            "Provide a comprehensive interpretation of this image.",
        )
        if custom_instructions.strip():
            prompt += f"\nAdditional user instructions: {custom_instructions.strip()}"
        return prompt

    @staticmethod
    def _extract_image_features(image: Image.Image) -> Dict[str, Any]:
        """Extract resolution, luminance, contrast, and dominant color metrics."""
        width, height = image.size
        aspect_ratio = round(width / max(height, 1), 2)
        stat = ImageStat.Stat(image.convert("RGB"))
        r_mean, g_mean, b_mean = [round(value, 1) for value in stat.mean]
        brightness = round(0.299 * r_mean + 0.587 * g_mean + 0.114 * b_mean, 1)
        is_dark = brightness < 110
        is_high_contrast = max(stat.stddev) > 55

        if r_mean > g_mean and r_mean > b_mean:
            dominant_channel = "Warm / Red-Orange"
        elif b_mean > r_mean and b_mean > g_mean:
            dominant_channel = "Cool / Blue-Cyan"
        else:
            dominant_channel = "Balanced / Green-Neutral"

        return {
            "dimensions": f"{width}x{height}",
            "aspect_ratio": f"{aspect_ratio}:1",
            "mean_rgb": f"RGB({int(r_mean)}, {int(g_mean)}, {int(b_mean)})",
            "perceived_brightness": (
                f"{brightness}/255 ({'Low-key / Moody' if is_dark else 'High-key / Vibrant'})"
            ),
            "color_temperature": dominant_channel,
            "contrast_profile": (
                "High Dynamic Contrast" if is_high_contrast else "Soft Dynamic Range"
            ),
        }


vision_engine = VisionEngine()
