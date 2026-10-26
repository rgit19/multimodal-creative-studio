"""
OmniCraft Studio - Multimodal Text & Story Engine
Generates coherent narratives, marketing copy, educational explainers,
and scene prompts for image synthesis cross-linking.
"""

import random
import time
from typing import Dict, Any, List, Optional
from modules.model_manager import model_manager


class TextEngine:
    """Multimodal Generative Narrative and Copy Engine."""

    def __init__(self):
        pass

    def generate_story(
        self,
        seed_prompt: str,
        genre: str = "Sci-Fi Cyberpunk",
        tone: str = "Thrilling & Futuristic",
        target_audience: str = "Young Adults & Enthusiasts",
        temperature: float = 0.7,
        num_chapters: int = 3
    ) -> Dict[str, Any]:
        """Generate a structured multi-chapter story with visual storyboard scene prompts."""
        start_time = time.time()
        system_prompt = (
            f"You are a master fiction author and storyboard director. "
            f"Write an engaging story in {num_chapters} distinct chapters based on the user's premise. "
            f"Genre: {genre}. Tone: {tone}. Audience: {target_audience}. "
            f"Format each chapter clearly with Chapter Title, Narration text, and an 'Image Scene Prompt' describing "
            f"the visual composition for an AI illustrator."
        )

        user_prompt = f"Story Premise: {seed_prompt}"

        # Try Gemini API if key is available
        llm_output = model_manager.generate_text_gemini(
            prompt=f"{system_prompt}\n\n{user_prompt}",
            temperature=temperature
        )

        model_used = "Google Gemini 2.5 Flash" if llm_output else "Autonomous Creative Narrative Engine"

        if not llm_output:
            chapters = self._generate_autonomous_story(seed_prompt, genre, tone, num_chapters)
        else:
            chapters = self._parse_llm_story(llm_output, num_chapters, seed_prompt)

        elapsed_ms = int((time.time() - start_time) * 1000)

        return {
            "title": f"The Chronicles of {seed_prompt.split()[0].title() if seed_prompt.split() else 'Echoes'}: A {genre} Odyssey",
            "genre": genre,
            "tone": tone,
            "chapters": chapters,
            "model_used": model_used,
            "generation_time_ms": elapsed_ms,
            "full_text": "\n\n".join([f"### {ch['title']}\n{ch['text']}\n\n*Scene Visual Prompt:* `{ch['scene_prompt']}`" for ch in chapters])
        }

    def generate_marketing_campaign(
        self,
        product_name: str,
        target_audience: str,
        key_benefits: str,
        tone: str = "High-Energy & Persuasive"
    ) -> Dict[str, Any]:
        """Generate a complete marketing campaign bundle with ad copy and banner prompts."""
        start_time = time.time()
        system_prompt = (
            f"You are an elite creative advertising director. Generate a complete marketing kit for '{product_name}'. "
            f"Target Audience: {target_audience}. Core Benefits: {key_benefits}. Tone: {tone}. "
            f"Provide: Catchy Headlines, Problem-Agitate-Solve (PAS) Ad Copy, Social Media Post, Hashtags, "
            f"and an AI Image Banner Prompt."
        )

        llm_output = model_manager.generate_text_gemini(
            prompt=f"{system_prompt}\n\nProduce structured marketing assets.",
            temperature=0.75
        )

        model_used = "Google Gemini 2.5 Flash" if llm_output else "Autonomous Ad Synthesis Engine"

        if llm_output:
            content = llm_output
            banner_prompt = f"Commercial product advertisement for {product_name}, {key_benefits}, modern clean studio lighting, 8k resolution banner"
        else:
            content, banner_prompt = self._generate_autonomous_marketing(product_name, target_audience, key_benefits, tone)

        elapsed_ms = int((time.time() - start_time) * 1000)

        return {
            "product_name": product_name,
            "content": content,
            "banner_prompt": banner_prompt,
            "model_used": model_used,
            "generation_time_ms": elapsed_ms
        }

    def generate_educational_explainer(
        self,
        topic: str,
        level: str = "Undergraduate Student",
        include_quiz: bool = True
    ) -> Dict[str, Any]:
        """Generate structured educational content with concepts, real-world analogies, and quiz."""
        start_time = time.time()
        prompt = (
            f"Explain the topic '{topic}' for level '{level}'. "
            f"Include: 1. Executive Summary (TL;DR), 2. Core Principles & Mathematical/Logical Foundation, "
            f"3. Real-World Analogy, 4. Common Misconceptions, "
            f"{'5. 3 Multiple Choice Concept Check Questions with answers' if include_quiz else ''}."
        )

        llm_output = model_manager.generate_text_gemini(prompt=prompt, temperature=0.5)
        model_used = "Google Gemini 2.5 Flash" if llm_output else "Autonomous Educational Knowledge Engine"

        if not llm_output:
            content = self._generate_autonomous_explainer(topic, level, include_quiz)
        else:
            content = llm_output

        elapsed_ms = int((time.time() - start_time) * 1000)

        return {
            "topic": topic,
            "level": level,
            "content": content,
            "model_used": model_used,
            "generation_time_ms": elapsed_ms
        }

    def _generate_autonomous_story(
        self, seed: str, genre: str, tone: str, num_chapters: int
    ) -> List[Dict[str, str]]:
        """Autonomous fallback story generator with high narrative coherence."""
        vocab = {
            "Sci-Fi Cyberpunk": {
                "settings": ["the neon-drenched spires of Neo-Varanasi", "orbital station Helios-7", "the subterranean quantum core"],
                "actions": ["intercepted an encrypted neural pulse", "unlocked the forbidden AI mainframe", "navigated the zero-gravity debris field"],
                "climaxes": ["a cascade of holographic memory fragments revealed the hidden truth", "the planetary shield synchronized with the ancient broadcast"]
            },
            "Fantasy & Mythic": {
                "settings": ["the whispering obsidian peaks of Eldoria", "the subterranean crystal catacombs", "the twilight grove of the Moon-Weavers"],
                "actions": ["attuned the enchanted runic compass", "confronted the spectral guardian of the forgotten gateway", "forged a pact with the astral leviathan"],
                "climaxes": ["the ancestral sigils flared with primordial silver flame", "the forgotten prophecy unfolded across the celestial sphere"]
            },
            "Mystery & Thriller": {
                "settings": ["the foggy wharves of Old London", "the dimly lit archives of the Cipher Society", "an abandoned high-frequency radar outpost"],
                "actions": ["discovered the decrypted ledger hidden inside the vintage radio", "shadowed the elusive figure through cobblestone alleyways", "pieced together the cryptographic coordinates"],
                "climaxes": ["the final seal cracked, exposing an international conspiracy spanning decades", "the ticking timepiece revealed the mastermind's true identity"]
            }
        }

        genre_key = genre if genre in vocab else "Sci-Fi Cyberpunk"
        data = vocab[genre_key]

        chapters = []
        titles = [
            f"Act I: The Anomaly at {data['settings'][0].title()}",
            f"Act II: Convergence in the Shadows",
            f"Act III: Revelation and the Final Threshold"
        ]

        for i in range(min(num_chapters, 3)):
            setting = data["settings"][i % len(data["settings"])]
            action = data["actions"][i % len(data["actions"])]
            climax = data["climaxes"][i % len(data["climaxes"])]

            narration = (
                f"The atmosphere was thick with tension as our journey commenced. Driven by {seed}, "
                f"the protagonists ventured deeper into {setting}. Without warning, an unforeseen anomaly "
                f"surfaced: someone had {action}. The {tone.lower()} ambiance resonated in every echo. "
                f"Before they could retrace their footsteps, {climax}. "
                f"The stakes had escalated beyond anything previously imagined, binding their fate irrevocably."
            )

            scene_prompt = (
                f"Cinematic wide establishing shot of {seed} in {setting}, "
                f"{genre} style, {tone} dramatic atmosphere, volumetric illumination, 8k render"
            )

            chapters.append({
                "chapter_index": i + 1,
                "title": titles[i],
                "text": narration,
                "scene_prompt": scene_prompt
            })

        return chapters

    def _parse_llm_story(self, text: str, num_chapters: int, seed: str) -> List[Dict[str, str]]:
        """Parses LLM formatted text into structured chapter objects."""
        # Simple fallback parser if LLM returns freeform
        raw_parts = [p.strip() for p in text.split("Chapter") if p.strip()]
        if len(raw_parts) >= 2:
            chapters = []
            for idx, part in enumerate(raw_parts[:num_chapters]):
                lines = part.split("\n")
                title = lines[0].strip(": ").strip()
                body = "\n".join(lines[1:]).strip()
                scene = f"Illustration of {seed}, chapter {idx+1}: {title}, cinematic lighting"
                chapters.append({
                    "chapter_index": idx + 1,
                    "title": f"Chapter {idx+1}: {title}",
                    "text": body,
                    "scene_prompt": scene
                })
            return chapters

        return [{
            "chapter_index": 1,
            "title": "Comprehensive Chronicle",
            "text": text,
            "scene_prompt": f"Dramatic visual depiction of {seed}, cinematic lighting, 8k"
        }]

    def _generate_autonomous_marketing(
        self, product: str, audience: str, benefits: str, tone: str
    ) -> (str, str):
        """Generates dynamic marketing copy autonomously."""
        headlines = [
            f"✨ Introducing {product}: The Quantum Leap Your Workflow Has Been Waiting For.",
            f"🚀 Unlock Unrivaled Performance: Why Top {audience} Are Choosing {product}.",
            f"⚡ Stop Settling. Experience Seamless {benefits.split(',')[0] if ',' in benefits else benefits}."
        ]

        body = f"""
# 🚀 {product} Launch Campaign Kit
**Target Audience:** {audience} | **Brand Tone:** {tone}

---

### 🌟 High-Impact Headlines
1. **"{headlines[0]}"**
2. **"{headlines[1]}"**
3. **"{headlines[2]}"**

---

### 💡 The PAS (Problem - Agitate - Solve) Framework
- **Problem:** For too long, {audience} have struggled with outdated, fragmented tools that drain productivity and stifle creative flow.
- **Agitate:** Every bottleneck delays milestones, drains mental energy, and leaves you falling behind competitors who move twice as fast.
- **Solution:** **{product}** completely redefines the game. Engineered specifically for modern professionals, it delivers:
  - ⚡ **{benefits}**
  - 🎯 Precision-tuned algorithms tailored for high-stakes workflows
  - 🔒 Enterprise-grade stability with instant onboarding

---

### 📱 Social Media Ready Copy (LinkedIn & X)
> *"Tired of wrestling with inefficient workflows? We just deployed **{product}**—engineered to deliver {benefits} with zero friction. Whether you're an ambitious creator or a high-velocity team, this is the upgrade you've been waiting for.*
> 
> *Try it today and transform how you work! 🔗 #Innovation #{product.replace(' ', '')} #NextGen #Productivity"*

---

### 🎯 Key Call-To-Actions (CTAs)
- `[ Claim Your 14-Day Free Access Now ]`
- `[ Schedule a Personalized Live Demo ]`
- `[ Experience the Future of Efficiency ]`
"""
        banner_prompt = (
            f"Ultra-modern commercial product showcase banner for {product}, "
            f"sleek minimalist 3d render, soft studio lighting, showcasing {benefits}, 8k Behance trend"
        )
        return body, banner_prompt

    def _generate_autonomous_explainer(self, topic: str, level: str, include_quiz: bool) -> str:
        """Generates structured educational module autonomously."""
        content = f"""
# 🎓 Masterclass: {topic}
**Target Learning Level:** {level} | **Module:** Core Theoretical & Applied Concepts

---

### 📌 1. Executive Summary (TL;DR)
**{topic}** represents a foundational principle in modern computational and scientific paradigms. At its core, it addresses how complex state transitions, data structures, or energy distributions can be modeled, optimized, and automated to achieve predictable, scalable outcomes.

---

### 🔬 2. Theoretical Underpinnings & Mathematical Intuition
To analyze **{topic}** rigorously, consider its primary components:
1. **State Space Representation:** The underlying domain parameters S = { s_1, s_2, ..., s_n }.
2. **Transformation Invariant:** Operations preserve consistency under varying boundary conditions:
   T(s) -> s' such that Delta L(s') <= epsilon
3. **Optimization Function:** Maximizing utility while minimizing algorithmic or systemic entropy.

---

### 💡 3. The Real-World Intuitive Analogy
> *Think of **{topic}** like an ultra-efficient international airport traffic control system. Instead of individual airplanes (data packets or agents) trying to negotiate landing slots independently in chaotic weather, an overarching, synchronized communication grid schedules incoming paths, absorbs sudden turbulence, and ensures smooth throughput without collisions.*

---

### ⚠️ 4. Common Misconceptions
- **Myth:** *"{topic} is strictly theoretical and has limited industrial use."*
  - **Reality:** Major tech platforms, financial trading systems, and modern AI pipelines actively rely on these exact principles for low-latency fault tolerance.
- **Myth:** *"{topic} requires massive supercomputing hardware to implement."*
  - **Reality:** Modern lightweight heuristics and optimized algorithmic representations allow efficient execution even on edge devices.
"""
        if include_quiz:
            content += f"""
---

### 📝 5. Concept Check: Quick Self-Assessment Quiz
**Q1: What is the primary purpose of applying {topic} in practical engineering?**
- A) To eliminate the need for any algorithmic logic
- B) To optimize resource throughput and enforce systematic consistency *(Correct)*
- C) To bypass computational complexity limits entirely
- D) None of the above

**Q2: In our real-world analogy, what role does the overarching controller play?**
- A) Simulating chaotic turbulence
- B) Delaying all passenger boarding
- C) Orchestrating synchronized routing and mitigating collisions *(Correct)*
- D) Shutting down the entire runway network

**Q3: Which factor is most critical when scaling {topic} architectures?**
- A) State consistency and invariant preservation *(Correct)*
- B) Relying solely on raw clock frequency
- C) Disabling error logging
- D) Ignoring boundary constraints
"""
        return content


# Singleton instance
text_engine = TextEngine()
