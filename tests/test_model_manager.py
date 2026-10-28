import os
import unittest
from unittest.mock import patch

from google.genai.errors import APIError

from modules.model_manager import (
    DEFAULT_TEXT_MODEL,
    SUPPORTED_TEXT_MODELS,
    DUPLICATE_REQUEST_WINDOW_SECONDS,
    RATE_LIMIT_REQUESTS,
    ModelManager,
    RequestCancelledError,
    model_manager,
    resolve_api_key,
    sanitize_error_message,
    validate_credentials,
)


class ModelManagerConfigurationTests(unittest.TestCase):
    def test_api_key_is_resolved_from_server_environment(self):
        secret = "AIzaSyDUMMY0123456789abcdef0123456789"
        with patch.dict(os.environ, {"GEMINI_API_KEY": secret}):
            self.assertEqual(resolve_api_key(), (secret, "Environment (.env / server)"))

    def test_missing_and_placeholder_credentials_are_reported(self):
        self.assertEqual(validate_credentials("")[1], "missing")
        self.assertEqual(validate_credentials("your_gemini_api_key_here")[1], "placeholder")
        self.assertEqual(validate_credentials("short")[1], "malformed")

    def test_lower_latency_text_model_is_selectable(self):
        self.assertEqual(DEFAULT_TEXT_MODEL, "gemini-3.8-flash")
        self.assertIn("gemini-3.5-flash-lite", SUPPORTED_TEXT_MODELS)
        status = model_manager.get_provider_status(text_model="gemini-3.5-flash-lite")
        self.assertEqual(status["text_model"], "gemini-3.5-flash-lite")

    def test_provider_status_never_contains_the_configured_key(self):
        secret = "AIzaSyDUMMY0123456789abcdef0123456789"
        with patch.dict(os.environ, {"GEMINI_API_KEY": secret}):
            status = model_manager.get_provider_status()

        self.assertTrue(status["is_ready"])
        self.assertNotIn(secret, repr(status))

    def test_unauthorized_provider_error_is_safe(self):
        secret = "AIzaSyDUMMY0123456789abcdef0123456789"
        error = APIError(401, {"message": f"invalid API key {secret}"})

        message = sanitize_error_message(error, secret)

        self.assertIn("HTTP 401", message)
        self.assertIn("invalid API key [REDACTED]", message)
        self.assertNotIn(secret, message)

    def test_api_error_code_and_message_survive_wrapping(self):
        api_error = APIError(
            404,
            {"error": {"message": "Model is not supported", "status": "NOT_FOUND"}},
        )

        message = sanitize_error_message(RuntimeError(sanitize_error_message(api_error)))

        self.assertIn("HTTP 404", message)
        self.assertIn("Model is not supported", message)

    def test_gateway_timeout_keeps_provider_message_and_suggests_safe_retry(self):
        error = APIError(504, {"message": "Deadline expired before operation could complete."})

        message = sanitize_error_message(error)

        self.assertIn("HTTP 504", message)
        self.assertIn("Deadline expired before operation could complete.", message)
        self.assertIn("shorten the prompt", message)

    def test_generic_provider_error_redacts_credentials_and_keeps_diagnostic(self):
        secret = "AIzaSyDUMMY0123456789abcdef0123456789"

        message = sanitize_error_message(RuntimeError(f"private backend detail: {secret}"))

        self.assertNotIn(secret, message)
        self.assertIn("private backend detail", message)

    def test_missing_credentials_do_not_initialize_a_provider_client(self):
        with patch("modules.model_manager.genai.Client") as client:
            with self.assertRaisesRegex(RuntimeError, "GEMINI_API_KEY"):
                model_manager.generate_text_gemini("test prompt", api_key="")

        client.assert_not_called()

    def test_unsupported_text_model_is_rejected_before_provider_access(self):
        manager = ModelManager()
        with patch("modules.model_manager.genai.Client") as client:
            with self.assertRaisesRegex(ValueError, "not supported"):
                manager.generate_text_gemini(
                    "valid prompt",
                    "mock-provider-key-123456789",
                    model="unlisted-model",
                )

        client.assert_not_called()

    def test_request_validation_and_rate_limits_do_not_call_provider(self):
        manager = ModelManager()
        key = "mock-provider-key-123456789"
        first_request = manager._reserve_request(key, "first prompt", "mock-model")

        with self.assertRaisesRegex(RuntimeError, "already in progress"):
            manager._reserve_request(key, "first prompt", "mock-model")
        manager._release_request(first_request)

        for index in range(1, RATE_LIMIT_REQUESTS):
            manager._reserve_request(key, f"prompt {index}", "mock-model")

        with self.assertRaisesRegex(RuntimeError, "Request limit reached"):
            manager._reserve_request(key, "last prompt", "mock-model")

        with self.assertRaisesRegex(ValueError, "request limit"):
            manager._validate_text_request("a" * 50_001)

    def test_duplicate_window_is_finite(self):
        manager = ModelManager()
        key = "mock-provider-key-123456789"
        with patch("modules.model_manager.time.monotonic", return_value=100):
            first_request = manager._reserve_request(key, "same prompt", "mock-model")
            with self.assertRaisesRegex(RuntimeError, "already in progress"):
                manager._reserve_request(key, "same prompt", "mock-model")
            manager._release_request(first_request)

        with patch(
            "modules.model_manager.time.monotonic",
            return_value=100 + DUPLICATE_REQUEST_WINDOW_SECONDS + 1,
        ):
            manager._reserve_request(key, "same prompt", "mock-model")

    def test_explicit_retry_can_bypass_only_the_completed_request_cooldown(self):
        manager = ModelManager()
        key = "mock-provider-key-123456789"
        with patch("modules.model_manager.time.monotonic", return_value=100):
            request_id = manager._reserve_request(key, "same prompt", "mock-model")
            manager._release_request(request_id)
            manager._reserve_request(
                key,
                "same prompt",
                "mock-model",
                allow_recent_duplicate=True,
            )
            with self.assertRaisesRegex(RuntimeError, "already in progress"):
                manager._reserve_request(
                    key,
                    "same prompt",
                    "mock-model",
                    allow_recent_duplicate=True,
                )


if __name__ == "__main__":
    unittest.main()
