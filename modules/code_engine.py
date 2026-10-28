"""
Nexa AI - Gemini-powered code generation, explanation, debugging, and refactoring.
"""

import re
import time
from threading import Event
from typing import Any, Callable, Dict, Optional, Tuple

from modules.model_manager import TEXT_MODEL, model_manager


class CodeEngine:
    """Generate and improve code with Gemini without executing generated source."""

    def generate_code(
        self,
        api_key: str,
        prompt: str,
        language: str = "Python",
        complexity: str = "Production-Ready",
        include_mermaid: bool = True,
        model: Optional[str] = None,
        on_chunk: Optional[Callable[[str], None]] = None,
        cancel_event: Optional[Event] = None,
    ) -> Dict[str, Any]:
        """Generate code from a natural-language specification."""
        if not prompt.strip():
            raise ValueError("Describe the code you need.")

        task = (
            f"Implement the following software requirement: {prompt}\n"
            f"Target language: {language}. Desired implementation: {complexity}."
        )
        return self._generate_code_task(
            api_key=api_key,
            task=task,
            language=language,
            model=model,
            include_mermaid=include_mermaid,
            prompt=prompt,
            temperature=0.3,
            on_chunk=on_chunk,
            cancel_event=cancel_event,
        )

    def explain_code(
        self,
        api_key: str,
        code: str,
        language: str,
        model: Optional[str] = None,
        on_chunk: Optional[Callable[[str], None]] = None,
        cancel_event: Optional[Event] = None,
    ) -> Dict[str, Any]:
        """Explain code without modifying it."""
        if not code.strip():
            raise ValueError("Paste code to continue.")

        start_time = time.monotonic()
        active_model = model or TEXT_MODEL
        explanation = model_manager.generate_text_gemini(
            prompt=(
                f"Explain the purpose, control flow, important design choices, and "
                f"notable edge cases of this {language} code. Do not rewrite or "
                f"execute it.\n\nCode:\n```{self._fence_language(language)}\n"
                f"{code.strip()}\n```"
            ),
            api_key=api_key,
            model=active_model,
            temperature=0.2,
            on_chunk=on_chunk,
            cancel_event=cancel_event,
        )
        return {
            "prompt": code.strip(),
            "language": language,
            "code": code.strip(),
            "explanation": explanation,
            "complexity": "",
            "mermaid_chart": "",
            "model_used": active_model,
            "generation_time_ms": int((time.monotonic() - start_time) * 1000),
        }

    def debug_code(
        self,
        api_key: str,
        code: str,
        language: str,
        issue: str = "",
        model: Optional[str] = None,
        on_chunk: Optional[Callable[[str], None]] = None,
        cancel_event: Optional[Event] = None,
    ) -> Dict[str, Any]:
        """Diagnose a reported issue and return a corrected implementation."""
        instruction = (
            "Find likely bugs and provide a corrected, complete implementation. "
            "Explain each important fix."
        )
        if issue.strip():
            instruction += f"\nReported issue or error:\n{issue.strip()}"
        return self._transform_code(
            api_key, code, language, model, instruction, on_chunk, cancel_event
        )

    def refactor_code(
        self,
        api_key: str,
        code: str,
        language: str,
        goals: str = "",
        model: Optional[str] = None,
        on_chunk: Optional[Callable[[str], None]] = None,
        cancel_event: Optional[Event] = None,
    ) -> Dict[str, Any]:
        """Refactor code while preserving its existing behavior."""
        instruction = (
            "Refactor the code for readability and maintainability while preserving "
            "its observable behavior. Return the complete refactored implementation "
            "and summarize the changes."
        )
        if goals.strip():
            instruction += f"\nAdditional refactoring goals:\n{goals.strip()}"
        return self._transform_code(
            api_key, code, language, model, instruction, on_chunk, cancel_event
        )

    def _transform_code(
        self,
        api_key: str,
        code: str,
        language: str,
        model: Optional[str],
        instruction: str,
        on_chunk: Optional[Callable[[str], None]] = None,
        cancel_event: Optional[Event] = None,
    ) -> Dict[str, Any]:
        if not code.strip():
            raise ValueError("Paste code to continue.")

        task = (
            f"{instruction}\n\nTarget language: {language}.\n"
            "Put any complete implementation in a fenced code block with the "
            f"language tag {self._fence_language(language)}. Never execute the code.\n\n"
            f"Input code:\n```{self._fence_language(language)}\n{code.strip()}\n```"
        )
        return self._generate_code_task(
            api_key=api_key,
            task=task,
            language=language,
            model=model,
            include_mermaid=False,
            prompt=code.strip(),
            temperature=0.2,
            on_chunk=on_chunk,
            cancel_event=cancel_event,
        )

    def _generate_code_task(
        self,
        api_key: str,
        task: str,
        language: str,
        model: Optional[str],
        include_mermaid: bool,
        prompt: str,
        temperature: float,
        on_chunk: Optional[Callable[[str], None]] = None,
        cancel_event: Optional[Event] = None,
    ) -> Dict[str, Any]:
        start_time = time.monotonic()
        active_model = model or TEXT_MODEL
        diagram_instruction = (
            " Include a concise architecture or logic Mermaid diagram in a "
            "```mermaid code block."
            if include_mermaid
            else " Do not include a Mermaid diagram."
        )
        system_prompt = (
            "You are a senior software engineer. Follow the user's requirements "
            "and target language precisely. Return a complete implementation in "
            f"a fenced {self._fence_language(language)} code block, followed by "
            "a concise explanation and relevant complexity details."
            f"{diagram_instruction} Do not execute code."
        )
        response = model_manager.generate_text_gemini(
            prompt=f"{system_prompt}\n\nTask:\n{task}",
            api_key=api_key,
            model=active_model,
            temperature=temperature,
            on_chunk=on_chunk,
            cancel_event=cancel_event,
        )
        code, explanation, complexity_info, mermaid_chart = self._parse_llm_code_response(
            response,
            language,
        )
        return {
            "prompt": prompt,
            "language": language,
            "code": code,
            "explanation": explanation,
            "complexity": complexity_info,
            "mermaid_chart": mermaid_chart if include_mermaid else "",
            "model_used": active_model,
            "generation_time_ms": int((time.monotonic() - start_time) * 1000),
        }

    @staticmethod
    def _fence_language(language: str) -> str:
        return {
            "C++": "cpp",
            "HTML/CSS": "html",
        }.get(language, language.lower())

    @classmethod
    def _parse_llm_code_response(
        cls,
        text: str,
        language: str,
    ) -> Tuple[str, str, str, str]:
        """Extract a language-matched implementation and supporting explanation."""
        fence_pattern = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)
        blocks = fence_pattern.findall(text)
        requested_language = cls._fence_language(language).lower()
        mermaid_chart = next(
            (body.strip() for tag, body in blocks if tag.strip().lower() == "mermaid"),
            "",
        )
        code_block = next(
            (
                body.strip()
                for tag, body in blocks
                if tag.strip().lower() == requested_language
            ),
            "",
        )
        if not code_block:
            code_block = next(
                (body.strip() for tag, body in blocks if tag.strip().lower() != "mermaid"),
                "",
            )

        complexity_info = "Complexity details not provided."
        for line in text.splitlines():
            if "complexity" in line.lower() or line.strip().lower().startswith(("time:", "space:")):
                complexity_info = line.strip("* -#")
                break

        explanation = fence_pattern.sub("", text).strip()
        if mermaid_chart:
            explanation = re.sub(
                r"```mermaid[^\n]*\n.*?```",
                "",
                explanation,
                flags=re.DOTALL | re.IGNORECASE,
            ).strip()
        return code_block or text.strip(), explanation or text.strip(), complexity_info, mermaid_chart


code_engine = CodeEngine()
