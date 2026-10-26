"""
OmniCraft Studio - Model Manager Module
Coordinates API clients (Gemini, HuggingFace, Pollinations, Local Engine),
manages credentials, handles failovers, and provides uniform inference interfaces.
"""

import os
import requests
from typing import Dict, Any, Optional

try:
    from google import genai
    from google.genai import types
    HAS_GEMINI_SDK = True
except ImportError:
    HAS_GEMINI_SDK = False

try:
    from huggingface_hub import InferenceClient
    HAS_HF_SDK = True
except ImportError:
    HAS_HF_SDK = False


class ModelManager:
    """Manages connections to Generative AI backends with seamless fallbacks."""

    def __init__(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        self.hf_token = os.getenv("HF_TOKEN", "")
        self._gemini_client = None
        self._hf_client = None

        if self.gemini_api_key and HAS_GEMINI_SDK:
            try:
                self._gemini_client = genai.Client(api_key=self.gemini_api_key)
            except Exception as e:
                print(f"Warning: Could not initialize Gemini client: {e}")

    def update_keys(self, gemini_key: Optional[str] = None, hf_key: Optional[str] = None):
        """Update API keys dynamically from the user interface."""
        if gemini_key is not None:
            self.gemini_api_key = gemini_key.strip()
            if self.gemini_api_key and HAS_GEMINI_SDK:
                try:
                    self._gemini_client = genai.Client(api_key=self.gemini_api_key)
                except Exception as e:
                    print(f"Gemini client init error: {e}")
                    self._gemini_client = None
            else:
                self._gemini_client = None

        if hf_key is not None:
            self.hf_token = hf_key.strip()
            if self.hf_token and HAS_HF_SDK:
                try:
                    self._hf_client = InferenceClient(token=self.hf_token)
                except Exception as e:
                    print(f"HF client init error: {e}")
                    self._hf_client = None
            else:
                self._hf_client = None

    def get_status(self) -> Dict[str, Any]:
        """Return connectivity status of all configured model providers."""
        return {
            "gemini_ready": bool(self._gemini_client or self.gemini_api_key),
            "huggingface_ready": bool(self._hf_client or self.hf_token),
            "pollinations_ready": True,  # Public free inference
            "local_engine_ready": True,  # Built-in autonomous engine
            "active_mode": "Cloud Pre-trained (Gemini/HF)" if (self.gemini_api_key or self.hf_token) else "Autonomous / Public Cloud Hybrid"
        }

    def generate_text_gemini(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.7
    ) -> Optional[str]:
        """Generate text using Google Gemini API."""
        if not self._gemini_client and self.gemini_api_key and HAS_GEMINI_SDK:
            try:
                self._gemini_client = genai.Client(api_key=self.gemini_api_key)
            except Exception:
                pass

        if not self._gemini_client:
            return None

        try:
            config = types.GenerateContentConfig(
                temperature=temperature,
                system_instruction=system_instruction if system_instruction else None
            )
            response = self._gemini_client.models.generate_content(
                model=model,
                contents=prompt,
                config=config
            )
            return response.text
        except Exception as e:
            print(f"Gemini generation error: {e}")
            return None

    def analyze_image_gemini(
        self,
        image_bytes: bytes,
        mime_type: str,
        prompt: str,
        model: str = "gemini-2.5-flash"
    ) -> Optional[str]:
        """Perform multimodal visual analysis with Gemini Vision."""
        if not self._gemini_client and self.gemini_api_key and HAS_GEMINI_SDK:
            try:
                self._gemini_client = genai.Client(api_key=self.gemini_api_key)
            except Exception:
                pass

        if not self._gemini_client:
            return None

        try:
            contents = [
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                prompt
            ]
            response = self._gemini_client.models.generate_content(
                model=model,
                contents=contents
            )
            return response.text
        except Exception as e:
            print(f"Gemini vision error: {e}")
            return None

    def generate_pollinations_image(
        self,
        prompt: str,
        width: int = 768,
        height: int = 768,
        seed: Optional[int] = None,
        model: str = "flux"
    ) -> Optional[bytes]:
        """Generate images via Pollinations public generative AI API."""
        import urllib.parse
        encoded_prompt = urllib.parse.quote(prompt)
        url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width={width}&height={height}&model={model}&nologo=true"
        if seed is not None:
            url += f"&seed={seed}"

        try:
            resp = requests.get(url, timeout=18)
            if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("image/"):
                return resp.content
        except Exception as e:
            print(f"Pollinations request failed or timed out: {e}")
        return None


# Global singleton instance
model_manager = ModelManager()
