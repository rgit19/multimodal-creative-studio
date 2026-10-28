"""
Nexa AI - Gemini API client, credential management, and multimodal inference helpers.
Brand: Nexa AI | Tagline: Create. Code. Imagine.
"""

import base64
import hashlib
import os
import re
import threading
import time
from collections import defaultdict, deque
from contextlib import closing, suppress
from typing import Any, Callable, Deque, Dict, Iterator, List, Optional, Set, Tuple

from google import genai
from google.genai import types
from streamlit.errors import StreamlitSecretNotFoundError

# Supported Models Registry
SUPPORTED_TEXT_MODELS: List[str] = [
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
]

SUPPORTED_IMAGE_MODELS: List[str] = [
    "gemini-nano-banana-2.1",
    "imagen-3.0-generate-002",
]


def load_env_file(filepath: Optional[str] = None) -> bool:
    """
    Safely load environment variables from a .env file into os.environ.
    Uses standard library only, with zero external package dependencies.
    Existing os.environ values are not overwritten.
    """
    if filepath is None:
        # Resolve .env relative to project root (parent directory of modules/)
        filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")

    if not os.path.exists(filepath):
        return False

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = val
    return True


# Load local settings before resolving model defaults.
load_env_file()

# Model defaults with environment variable overrides.
DEFAULT_TEXT_MODEL: str = os.getenv("GEMINI_TEXT_MODEL", "gemini-3.8-flash")
if DEFAULT_TEXT_MODEL not in SUPPORTED_TEXT_MODELS:
    DEFAULT_TEXT_MODEL = "gemini-3.8-flash"

DEFAULT_IMAGE_MODEL: str = os.getenv("GEMINI_IMAGE_MODEL", "gemini-nano-banana-2.1")
if DEFAULT_IMAGE_MODEL not in SUPPORTED_IMAGE_MODELS:
    DEFAULT_IMAGE_MODEL = "gemini-nano-banana-2.1"

# Backward compatibility constants
TEXT_MODEL: str = DEFAULT_TEXT_MODEL
IMAGE_MODEL: str = DEFAULT_IMAGE_MODEL

REQUEST_TIMEOUT_MS = 60_000
REQUEST_RETRY_ATTEMPTS = 2
REQUEST_RETRY_STATUS_CODES = [408, 429, 500, 502, 503]
MAX_REQUEST_CHARS = 50_000
RATE_LIMIT_REQUESTS = 10
RATE_LIMIT_WINDOW_SECONDS = 60
DUPLICATE_REQUEST_WINDOW_SECONDS = 2


class RequestCancelledError(RuntimeError):
    """Raised when a caller cancels an active streamed request."""


def validate_credentials(api_key: Optional[str]) -> Tuple[bool, str, str]:
    """
    Validate API credentials format without triggering live external API requests.
    Returns:
        (is_valid, status_code, human_readable_message)
        status_code in: 'ready', 'missing', 'placeholder', 'malformed'
    """
    if not api_key or not api_key.strip():
        return (
            False,
            "missing",
            "Gemini API key is missing. Set GEMINI_API_KEY in the server environment or Streamlit secrets.",
        )

    clean = api_key.strip()

    # Check for known placeholder values
    placeholders = {
        "your_gemini_api_key_here",
        "your_api_key_here",
        "aizasy...",
        "placeholder",
        "api_key",
        "none",
    }
    if clean.lower() in placeholders or clean.startswith("your_"):
        return (
            False,
            "placeholder",
            "Placeholder detected. Please replace the placeholder with a valid Gemini API key.",
        )

    # Check for whitespace or quotation marks
    if any(c in clean for c in [" ", "\t", "\n", "\r", '"', "'"]):
        return (
            False,
            "malformed",
            "API key contains invalid characters (whitespace or quotes).",
        )

    # Minimum length sanity check (Google API keys are typically ~39 chars)
    if len(clean) < 15:
        return (
            False,
            "malformed",
            "API key appears incomplete or too short. Please verify the copied key.",
        )

    return (
        True,
        "configured",
        "Gemini API key is configured. Provider acceptance is verified when you make a request.",
    )


def resolve_api_key() -> Tuple[str, str]:
    """
    Resolve API credentials from server-side configuration only:
    1. Server-side environment variable (GEMINI_API_KEY)
    2. Streamlit secrets (if available)

    Returns:
        (resolved_key, source_description)
    """
    env_key = os.getenv("GEMINI_API_KEY", "").strip()
    if env_key:
        return env_key, "Environment (.env / server)"

    try:
        import streamlit as st
        secret_key = st.secrets.get("GEMINI_API_KEY", "")
    except StreamlitSecretNotFoundError:
        return "", "Not configured"

    if secret_key:
        if not isinstance(secret_key, str):
            raise TypeError("GEMINI_API_KEY in Streamlit secrets must be a string.")
        return secret_key.strip(), "Streamlit secrets"

    return "", "Not configured"


def sanitize_error_message(error: Exception, api_key: Optional[str] = None) -> str:
    """
    Preserve provider error codes and messages while redacting credentials.
    """
    status_code = getattr(error, "code", None)
    message = getattr(error, "message", None) or str(error)
    if api_key:
        message = message.replace(api_key, "[REDACTED]")
    message = re.sub(r"AIza[0-9A-Za-z_-]{20,}", "[REDACTED]", message)
    message = re.sub(
        r"(?i)(api[_ -]?key\s*[:=]\s*)[^\s,'\"}]+",
        r"\1[REDACTED]",
        message,
    )
    if status_code is not None:
        if status_code == 504:
            message = (
                f"{message} Gemini did not finish before its server deadline. "
                "Retry once; if it repeats, shorten the prompt or wait and try again."
            )
        return f"Gemini API error (HTTP {status_code}): {message}"
    if message.startswith("Gemini API error (HTTP "):
        return message
    return f"Gemini request failed: {message}"


class ModelManager:
    """Manages Google Gemini API client connection, credential validation, and inference."""

    def __init__(self):
        self._request_lock = threading.Lock()
        self._request_times: Dict[str, Deque[float]] = defaultdict(deque)
        self._duplicate_times: Dict[str, float] = {}
        self._active_requests: Set[str] = set()
        self._metrics_lock = threading.Lock()
        self._request_count = 0
        self._completed_request_count = 0
        self._total_token_count = 0
        self._latencies_ms: Deque[int] = deque(maxlen=100)

    @staticmethod
    def _validate_text_request(prompt: str) -> None:
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")
        if len(prompt) > MAX_REQUEST_CHARS:
            raise ValueError(
                f"Prompt exceeds the {MAX_REQUEST_CHARS:,}-character request limit."
            )

    @staticmethod
    def _validate_model(model: str, supported_models: List[str]) -> None:
        if model not in supported_models:
            raise ValueError("The selected Gemini model is not supported.")

    def _reserve_request(
        self,
        api_key: str,
        prompt: str,
        model: str,
        allow_recent_duplicate: bool = False,
    ) -> str:
        is_valid, _, message = validate_credentials(api_key)
        if not is_valid:
            raise RuntimeError(message)

        key_fingerprint = hashlib.sha256(api_key.strip().encode("utf-8")).hexdigest()
        prompt_fingerprint = hashlib.sha256(
            f"{model}\0{prompt}".encode("utf-8")
        ).hexdigest()
        duplicate_key = f"{key_fingerprint}:{prompt_fingerprint}"
        now = time.monotonic()
        window_start = now - RATE_LIMIT_WINDOW_SECONDS

        with self._request_lock:
            for fingerprint, timestamps in list(self._request_times.items()):
                while timestamps and timestamps[0] <= window_start:
                    timestamps.popleft()
                if not timestamps:
                    del self._request_times[fingerprint]

            for fingerprint, timestamp in list(self._duplicate_times.items()):
                if now - timestamp >= DUPLICATE_REQUEST_WINDOW_SECONDS:
                    del self._duplicate_times[fingerprint]

            previous_request = self._duplicate_times.get(duplicate_key)
            if duplicate_key in self._active_requests:
                raise RuntimeError(
                    "This identical request is already in progress. Please wait for it to finish."
                )
            if (
                not allow_recent_duplicate
                and
                previous_request is not None
                and now - previous_request < DUPLICATE_REQUEST_WINDOW_SECONDS
            ):
                raise RuntimeError(
                    "This request was just submitted. Wait briefly before submitting it again."
                )

            request_times = self._request_times[key_fingerprint]
            if len(request_times) >= RATE_LIMIT_REQUESTS:
                wait_seconds = max(
                    1,
                    int(RATE_LIMIT_WINDOW_SECONDS - (now - request_times[0])),
                )
                raise RuntimeError(
                    f"Request limit reached. Try again in about {wait_seconds} seconds."
                )

            request_times.append(now)
            self._active_requests.add(duplicate_key)

            if len(self._request_times) > 1024:
                self._request_times.pop(next(iter(self._request_times)))
            if len(self._duplicate_times) > 2048:
                self._duplicate_times.pop(next(iter(self._duplicate_times)))
        return duplicate_key

    def _release_request(self, duplicate_key: str) -> None:
        with self._request_lock:
            self._active_requests.discard(duplicate_key)
            self._duplicate_times[duplicate_key] = time.monotonic()
            if len(self._duplicate_times) > 2048:
                self._duplicate_times.pop(next(iter(self._duplicate_times)))

    def get_request_metrics(self) -> Dict[str, int]:
        """Return aggregate in-process request, token, and latency metrics."""
        with self._metrics_lock:
            average_latency = (
                round(sum(self._latencies_ms) / len(self._latencies_ms))
                if self._latencies_ms
                else 0
            )
            return {
                "requests": self._request_count,
                "completed_requests": self._completed_request_count,
                "tokens": self._total_token_count,
                "average_latency_ms": average_latency,
            }

    def _record_request_metrics(
        self,
        start_time: float,
        token_count: Optional[int],
        completed: bool,
    ) -> None:
        elapsed_ms = round((time.monotonic() - start_time) * 1000)
        with self._metrics_lock:
            self._request_count += 1
            self._completed_request_count += int(completed)
            self._latencies_ms.append(elapsed_ms)
            if token_count is not None:
                self._total_token_count += token_count

    def get_status(self, api_key: Optional[str] = None) -> Dict[str, bool]:
        """
        Backward compatible status check.
        Returns whether Gemini is ready for inference.
        """
        resolved_key = api_key if api_key is not None else resolve_api_key()[0]
        is_valid, _, _ = validate_credentials(resolved_key)
        return {"gemini_ready": is_valid}

    def get_provider_status(
        self,
        text_model: Optional[str] = None,
        image_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Return comprehensive provider, credential, and model status for UI controls.
        Does not perform live network requests.
        """
        resolved_key, source = resolve_api_key()
        is_valid, status_code, message = validate_credentials(resolved_key)

        active_text_model = text_model or DEFAULT_TEXT_MODEL
        if active_text_model not in SUPPORTED_TEXT_MODELS:
            active_text_model = SUPPORTED_TEXT_MODELS[0]

        active_image_model = image_model or DEFAULT_IMAGE_MODEL
        if active_image_model not in SUPPORTED_IMAGE_MODELS:
            active_image_model = SUPPORTED_IMAGE_MODELS[0]

        return {
            "provider": "Google Gemini",
            "is_ready": is_valid,
            "status_code": status_code,
            "status_label": "🟢 Configured" if is_valid else ("⚠️ Invalid Configuration" if status_code != "missing" else "🔑 Key Required"),
            "message": message,
            "source": source,
            "text_model": active_text_model,
            "image_model": active_image_model,
            "available_text_models": SUPPORTED_TEXT_MODELS,
            "available_image_models": SUPPORTED_IMAGE_MODELS,
        }

    def _require_client(self, api_key: str) -> genai.Client:
        """
        Validate credentials and initialize Google Gemini client.
        Raises clean RuntimeError without leaking secrets if credentials are invalid.
        """
        is_valid, status_code, message = validate_credentials(api_key)
        if not is_valid:
            raise RuntimeError(message)

        try:
            return genai.Client(
                api_key=api_key.strip(),
                http_options=types.HttpOptions(
                    timeout=REQUEST_TIMEOUT_MS,
                    retry_options=types.HttpRetryOptions(
                        attempts=REQUEST_RETRY_ATTEMPTS,
                        initial_delay=0.5,
                        max_delay=1.0,
                        http_status_codes=REQUEST_RETRY_STATUS_CODES,
                    ),
                ),
            )
        except Exception as e:
            raise RuntimeError(
                sanitize_error_message(e, api_key)
            )

    def generate_text_gemini(
        self,
        prompt: str,
        api_key: str,
        system_instruction: Optional[str] = None,
        model: str = TEXT_MODEL,
        temperature: float = 0.7,
        on_chunk: Optional[Callable[[str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
        allow_recent_duplicate: bool = False,
    ) -> str:
        """Generate text, optionally delivering streamed chunks to a caller."""
        response_parts = []
        with closing(
            self.stream_text_gemini(
                prompt=prompt,
                api_key=api_key,
                system_instruction=system_instruction,
                model=model,
                temperature=temperature,
                cancel_event=cancel_event,
                allow_recent_duplicate=allow_recent_duplicate,
            )
        ) as chunks:
            for chunk in chunks:
                response_parts.append(chunk)
                if on_chunk is not None:
                    on_chunk(chunk)

        if cancel_event is not None and cancel_event.is_set():
            raise RequestCancelledError("Gemini request was cancelled.")
        response_text = "".join(response_parts)
        if not response_text.strip():
            raise RuntimeError("Gemini returned an empty text response.")
        return response_text

    def stream_text_gemini(
        self,
        prompt: str,
        api_key: str,
        system_instruction: Optional[str] = None,
        model: str = TEXT_MODEL,
        temperature: float = 0.7,
        cancel_event: Optional[threading.Event] = None,
        allow_recent_duplicate: bool = False,
    ) -> Iterator[str]:
        """Yield text chunks and close the provider stream when consumption stops."""
        self._validate_text_request(prompt)
        self._validate_model(model, SUPPORTED_TEXT_MODELS)
        if cancel_event is not None and cancel_event.is_set():
            raise RequestCancelledError("Gemini request was cancelled.")
        duplicate_key = self._reserve_request(
            api_key,
            prompt,
            model,
            allow_recent_duplicate=allow_recent_duplicate,
        )

        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction if system_instruction else None,
        )
        client = None
        stream = None
        start_time = time.monotonic()
        token_count = None
        completed = False
        try:
            client = self._require_client(api_key)
            stream = client.models.generate_content_stream(
                model=model,
                contents=prompt,
                config=config,
            )
            stream_iterator = iter(stream)
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise RequestCancelledError("Gemini request was cancelled.")

                try:
                    chunk = next(stream_iterator)
                except StopIteration:
                    break

                usage = getattr(chunk, "usage_metadata", None)
                reported_tokens = getattr(usage, "total_token_count", None)
                if reported_tokens is not None:
                    token_count = reported_tokens

                text = getattr(chunk, "text", None)
                if text:
                    yield text
            completed = True
        except (GeneratorExit, RequestCancelledError):
            raise
        except Exception as e:
            raise RuntimeError(sanitize_error_message(e, api_key))
        finally:
            if stream is not None:
                with suppress(Exception):
                    stream.close()
            if client is not None:
                with suppress(Exception):
                    client.close()
            self._release_request(duplicate_key)
            self._record_request_metrics(start_time, token_count, completed)

    def analyze_image_gemini(
        self,
        image_bytes: bytes,
        mime_type: str,
        prompt: str,
        api_key: str,
        model: str = TEXT_MODEL,
    ) -> str:
        """Analyze an image using Google Gemini's multimodal API."""
        if not image_bytes:
            raise ValueError("Image bytes cannot be empty.")
        self._validate_text_request(prompt)
        self._validate_model(model, SUPPORTED_TEXT_MODELS)
        duplicate_key = self._reserve_request(api_key, prompt, model)

        contents = [
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            prompt,
        ]
        client = None
        try:
            client = self._require_client(api_key)
            response = client.models.generate_content(
                model=model,
                contents=contents,
            )
        except Exception as e:
            raise RuntimeError(sanitize_error_message(e, api_key))
        finally:
            if client is not None:
                with suppress(Exception):
                    client.close()
            self._release_request(duplicate_key)

        if not response.text:
            raise RuntimeError("Gemini returned an empty image-analysis response.")
        return response.text

    def generate_image_gemini(
        self,
        prompt: str,
        aspect_ratio: str,
        api_key: str,
        model: str = IMAGE_MODEL,
    ) -> bytes:
        """Generate a prompt-based image using Gemini's native image model."""
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")
        self._validate_text_request(prompt)
        self._validate_model(model, SUPPORTED_IMAGE_MODELS)
        duplicate_key = self._reserve_request(api_key, prompt, model)

        client = None
        try:
            client = self._require_client(api_key)
            interaction = client.interactions.create(
                model=model,
                input=prompt,
                response_format={
                    "type": "image",
                    "aspect_ratio": aspect_ratio,
                    "image_size": "1K",
                },
                store=False,
            )
        except Exception as e:
            raise RuntimeError(sanitize_error_message(e, api_key))
        finally:
            if client is not None:
                with suppress(Exception):
                    client.close()
            self._release_request(duplicate_key)

        if not interaction.output_image or not interaction.output_image.data:
            detail = interaction.output_text or "Gemini did not return an image."
            raise RuntimeError(f"Gemini image generation failed: {detail}")
        return base64.b64decode(interaction.output_image.data)


model_manager = ModelManager()
