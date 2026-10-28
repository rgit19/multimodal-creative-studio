"""
OmniCraft Studio - Gemini-powered text and story generation.
"""

import time
from threading import Event
from typing import Any, Callable, Dict, List, Optional

from modules.model_manager import TEXT_MODEL, model_manager


class TextEngine:
    """Generate stories, marketing copy, and explainers with Gemini."""

    def generate_story(
        self,
        api_key: str,
        seed_prompt: str,
        genre: str = "Sci-Fi Cyberpunk",
        tone: str = "Thrilling & Futuristic",
        target_audience: str = "Young Adults & Enthusiasts",
        temperature: float = 0.7,
        num_chapters: int = 3,
        model: Optional[str] = None,
        on_chunk: Optional[Callable[[str], None]] = None,
        cancel_event: Optional[Event] = None,
    ) -> Dict[str, Any]:
        """Generate a multi-chapter story and visual storyboard prompts."""
        start_time = time.time()
        active_model = model or TEXT_MODEL
        system_prompt = (
            "You are a fiction author and storyboard director. Write an engaging "
            f"story in {num_chapters} distinct chapters based on the user's premise. "
            f"Genre: {genre}. Tone: {tone}. Audience: {target_audience}. "
            "Format each chapter clearly with a chapter title, narration, and an "
            "'Image Scene Prompt' describing the visual composition."
        )
        llm_output = model_manager.generate_text_gemini(
            prompt=f"{system_prompt}\n\nStory Premise: {seed_prompt}",
            api_key=api_key,
            model=active_model,
            temperature=temperature,
            on_chunk=on_chunk,
            cancel_event=cancel_event,
        )
        chapters = self._parse_llm_story(llm_output, num_chapters, seed_prompt)
        elapsed_ms = int((time.time() - start_time) * 1000)
        first_word = seed_prompt.split()[0].title() if seed_prompt.split() else "Echoes"

        return {
            "title": f"The Chronicles of {first_word}: A {genre} Odyssey",
            "genre": genre,
            "tone": tone,
            "chapters": chapters,
            "model_used": active_model,
            "generation_time_ms": elapsed_ms,
            "full_text": "\n\n".join(
                f"### {chapter['title']}\n{chapter['text']}\n\n"
                f"*Scene Visual Prompt:* `{chapter['scene_prompt']}`"
                for chapter in chapters
            ),
        }

    def generate_marketing_campaign(
        self,
        api_key: str,
        product_name: str,
        target_audience: str,
        key_benefits: str,
        tone: str = "High-Energy & Persuasive",
        model: Optional[str] = None,
        on_chunk: Optional[Callable[[str], None]] = None,
        cancel_event: Optional[Event] = None,
    ) -> Dict[str, Any]:
        """Generate a marketing launch bundle with Gemini."""
        start_time = time.time()
        active_model = model or TEXT_MODEL
        prompt = (
            f"Create a complete marketing launch kit for {product_name}. "
            f"Target audience: {target_audience}. Core benefits: {key_benefits}. "
            f"Tone: {tone}. Include catchy headlines, Problem-Agitate-Solve ad copy, "
            "a social media post, hashtags, and an AI image banner prompt."
        )
        content = model_manager.generate_text_gemini(
            prompt=prompt,
            api_key=api_key,
            model=active_model,
            temperature=0.75,
            on_chunk=on_chunk,
            cancel_event=cancel_event,
        )
        elapsed_ms = int((time.time() - start_time) * 1000)

        return {
            "product_name": product_name,
            "content": content,
            "banner_prompt": (
                f"Commercial product advertisement for {product_name}: {key_benefits}, "
                "modern clean studio lighting, high-resolution banner"
            ),
            "model_used": active_model,
            "generation_time_ms": elapsed_ms,
        }

    def generate_educational_explainer(
        self,
        api_key: str,
        topic: str,
        level: str = "Undergraduate Student",
        include_quiz: bool = True,
        model: Optional[str] = None,
        on_chunk: Optional[Callable[[str], None]] = None,
        cancel_event: Optional[Event] = None,
    ) -> Dict[str, Any]:
        """Generate a structured educational explainer with Gemini."""
        start_time = time.time()
        active_model = model or TEXT_MODEL
        quiz_instruction = (
            "Include three multiple-choice concept-check questions with answers."
            if include_quiz
            else ""
        )
        prompt = (
            f"Explain '{topic}' for a {level}. Include an executive summary, core "
            "principles and mathematical or logical foundations, a real-world "
            f"analogy, and common misconceptions. {quiz_instruction}"
        )
        content = model_manager.generate_text_gemini(
            prompt=prompt,
            api_key=api_key,
            model=active_model,
            temperature=0.5,
            on_chunk=on_chunk,
            cancel_event=cancel_event,
        )

        return {
            "topic": topic,
            "level": level,
            "content": content,
            "model_used": active_model,
            "generation_time_ms": int((time.time() - start_time) * 1000),
        }

    @staticmethod
    def _parse_llm_story(text: str, num_chapters: int, seed: str) -> List[Dict[str, str]]:
        """Extract chapter titles, narration, and scene prompts from Gemini output."""
        raw_parts = [part.strip() for part in text.split("Chapter") if part.strip()]
        if len(raw_parts) >= 2:
            chapters = []
            for index, part in enumerate(raw_parts[:num_chapters]):
                lines = part.splitlines()
                title = lines[0].strip(": ").strip()
                body = "\n".join(lines[1:]).strip()
                scene = f"Illustration of {seed}, chapter {index + 1}: {title}, cinematic lighting"
                chapters.append(
                    {
                        "chapter_index": index + 1,
                        "title": f"Chapter {index + 1}: {title}",
                        "text": body,
                        "scene_prompt": scene,
                    }
                )
            return chapters

        return [
            {
                "chapter_index": 1,
                "title": "Comprehensive Chronicle",
                "text": text,
                "scene_prompt": f"Dramatic visual depiction of {seed}, cinematic lighting",
            }
        ]


text_engine = TextEngine()
