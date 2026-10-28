import unittest
import threading
from types import SimpleNamespace
from unittest.mock import patch

from google import genai

from modules.code_engine import CodeEngine
from modules.model_manager import (
    REQUEST_RETRY_ATTEMPTS,
    REQUEST_RETRY_STATUS_CODES,
    REQUEST_TIMEOUT_MS,
    ModelManager,
    RequestCancelledError,
    model_manager,
)


class CodeEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = CodeEngine()

    @patch("modules.code_engine.model_manager.generate_text_gemini")
    def test_generates_natural_language_request_as_selected_language(self, generate):
        generate.return_value = (
            "A bucket refills over time.\n"
            "```python\nclass TokenBucket:\n    pass\n```\n"
            "Time Complexity: O(1)\n"
            "```mermaid\ngraph TD\nA-->B\n```"
        )
        chunks = []
        on_chunk = chunks.append

        result = self.engine.generate_code(
            api_key="mock-key",
            prompt="Create a token bucket with burst capacity",
            language="Python",
            include_mermaid=True,
            on_chunk=on_chunk,
        )

        self.assertEqual(result["code"], "class TokenBucket:\n    pass")
        self.assertEqual(result["complexity"], "Time Complexity: O(1)")
        self.assertEqual(result["mermaid_chart"], "graph TD\nA-->B")
        self.assertIn("Create a token bucket with burst capacity", generate.call_args.kwargs["prompt"])
        self.assertEqual(generate.call_args.kwargs["api_key"], "mock-key")
        self.assertIs(generate.call_args.kwargs["on_chunk"], on_chunk)

    @patch("modules.code_engine.model_manager.generate_text_gemini")
    def test_explanation_keeps_source_code_separate(self, generate):
        generate.return_value = "The loop counts each item once."
        source = "for item in items:\n    print(item)"

        result = self.engine.explain_code("mock-key", source, "Python")

        self.assertEqual(result["code"], source)
        self.assertEqual(result["explanation"], generate.return_value)
        self.assertIn("Do not rewrite or execute it", generate.call_args.kwargs["prompt"])

    @patch("modules.code_engine.model_manager.generate_text_gemini")
    def test_debug_includes_issue_and_returns_corrected_code(self, generate):
        generate.return_value = "Fixed null handling.\n```javascript\nconst value = input ?? 0;\n```"

        result = self.engine.debug_code(
            "mock-key",
            "const value = input.value",
            "JavaScript",
            issue="input may be null",
        )

        self.assertEqual(result["code"], "const value = input ?? 0;")
        self.assertIn("input may be null", generate.call_args.kwargs["prompt"])

    @patch("modules.code_engine.model_manager.generate_text_gemini")
    def test_refactor_includes_goals_and_maps_cpp_fence(self, generate):
        generate.return_value = "Simplified the function.\n```cpp\nint answer() { return 42; }\n```"

        result = self.engine.refactor_code(
            "mock-key",
            "int answer(){return 42;}",
            "C++",
            goals="Use consistent spacing",
        )

        self.assertEqual(result["code"], "int answer() { return 42; }")
        self.assertIn("```cpp", generate.call_args.kwargs["prompt"])
        self.assertIn("Use consistent spacing", generate.call_args.kwargs["prompt"])

    @patch("modules.code_engine.model_manager.generate_text_gemini")
    def test_empty_inputs_do_not_call_provider(self, generate):
        with self.assertRaises(ValueError):
            self.engine.generate_code("mock-key", "  ")
        with self.assertRaises(ValueError):
            self.engine.explain_code("mock-key", "  ", "Python")

        generate.assert_not_called()

    def test_generated_code_is_not_executed_by_code_engine(self):
        self.assertFalse(hasattr(self.engine, "execute_python_code"))


class ModelManagerTimeoutTests(unittest.TestCase):
    def test_text_requests_stream_with_finite_timeout_and_bounded_retries(self):
        mock_client = SimpleNamespace(
            models=SimpleNamespace(
                generate_content_stream=unittest.mock.Mock(
                    return_value=iter(
                        [
                            SimpleNamespace(text="mocked ", usage_metadata=None),
                            SimpleNamespace(
                                text="response",
                                usage_metadata=SimpleNamespace(total_token_count=7),
                            ),
                        ]
                    )
                )
            ),
            close=unittest.mock.Mock(),
        )
        with patch.object(genai, "Client", return_value=mock_client) as create_client:
            result = model_manager.generate_text_gemini(
                "test prompt",
                "mock-provider-key-123456789",
            )

        self.assertEqual(result, "mocked response")
        http_options = create_client.call_args.kwargs["http_options"]
        self.assertEqual(http_options.timeout, REQUEST_TIMEOUT_MS)
        self.assertEqual(http_options.retry_options.attempts, REQUEST_RETRY_ATTEMPTS)
        self.assertEqual(
            http_options.retry_options.http_status_codes,
            REQUEST_RETRY_STATUS_CODES,
        )
        self.assertGreater(REQUEST_TIMEOUT_MS, 0)
        mock_client.close.assert_called_once()

    def test_stream_cancellation_closes_provider_stream_and_records_usage(self):
        manager = ModelManager()
        cancel_event = threading.Event()
        stream_closed = threading.Event()

        def response_chunks():
            try:
                yield SimpleNamespace(text="partial", usage_metadata=None)
                yield SimpleNamespace(text="unread", usage_metadata=None)
            finally:
                stream_closed.set()

        mock_client = SimpleNamespace(
            models=SimpleNamespace(
                generate_content_stream=unittest.mock.Mock(
                    return_value=response_chunks()
                )
            ),
            close=unittest.mock.Mock(),
        )
        with patch.object(genai, "Client", return_value=mock_client):
            with self.assertRaises(RequestCancelledError):
                manager.generate_text_gemini(
                    "cancellation test prompt",
                    "mock-provider-key-123456789",
                    cancel_event=cancel_event,
                    on_chunk=lambda _: cancel_event.set(),
                )

        self.assertTrue(stream_closed.is_set())
        mock_client.close.assert_called_once()
        metrics = manager.get_request_metrics()
        self.assertEqual(metrics["requests"], 1)
        self.assertEqual(metrics["completed_requests"], 0)

    def test_streamed_chunks_are_forwarded_to_callback(self):
        received = []
        mock_client = SimpleNamespace(
            models=SimpleNamespace(
                generate_content_stream=unittest.mock.Mock(
                    return_value=iter(
                        [
                            SimpleNamespace(text="first", usage_metadata=None),
                            SimpleNamespace(
                                text=" second",
                                usage_metadata=SimpleNamespace(total_token_count=12),
                            ),
                        ]
                    )
                )
            ),
            close=unittest.mock.Mock(),
        )
        manager = ModelManager()
        with patch.object(genai, "Client", return_value=mock_client):
            response = manager.generate_text_gemini(
                "streaming callback test",
                "mock-provider-key-123456789",
                on_chunk=received.append,
            )

        self.assertEqual(response, "first second")
        self.assertEqual(received, ["first", " second"])
        self.assertEqual(manager.get_request_metrics()["tokens"], 12)


if __name__ == "__main__":
    unittest.main()
