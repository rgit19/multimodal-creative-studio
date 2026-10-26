# 🌟 OmniCraft AI: Multimodal Content Generation Studio

> **Mini Project:** Multimodal Content Generation using Generative AI Models  
> **Domain:** Creative Studio, Storyboarding, Software Synthesis & Digital Marketing  
> **Stack:** Streamlit, Python, Google Gemini 2.5 Flash, FLUX.1 / Stable Diffusion, Pillow, and Autonomous Fallback Engines.

---

## 📌 Executive Summary

**OmniCraft Studio** is an end-to-end Generative AI platform designed to synthesize, transform, and evaluate content across three fundamental modalities: **Text**, **Image**, and **Code**.

The system enables users to provide multimodal seed inputs (natural language descriptions, style parameters, seed images, and architectural requirements) to produce coherent, context-rich outputs. It integrates pre-trained cloud models with autonomous local procedural engines, guaranteeing 100% operational reliability whether offline or connected.

---

## 🚀 Key Functional Modules

### 1. 🎨 Visual Studio (Text-to-Image Synthesis)
- **Prompt & Negative Prompting:** Fine-grained artistic control over desired and suppressed features.
- **Style Conditioning:** Photorealistic (DSLR 8K), Cyberpunk/Neon, Anime (Studio Ghibli style), Watercolor Fantasy, Modern 3D Minimalist, and Classic Oil Painting.
- **Aspect Ratio Control:** 1:1 Square, 16:9 Banner, 9:16 Mobile Story/Reel, 4:3 Classic.
- **Backend Models:** FLUX.1-schnell / Stable Diffusion via Pollinations API + High-fidelity Autonomous Procedural Canvas Engine with Perlin harmonic wave rendering.
- **Exporting:** One-click full-resolution PNG download with generation provenance.

### 2. ✍️ Storytelling & Marketing Studio (Text-to-Text & Storyboarding)
- **Multi-Chapter Narratives:** Automatically weaves multi-act stories with dramatic conflict and tone modulation (Sci-Fi, Fantasy, Mystery).
- **Multimodal Storyboarding:** Each narrative chapter auto-generates a corresponding **Visual Scene Prompt**, which can be rendered directly in the Visual Studio with a single click!
- **Marketing Launch Kit:** Full Problem-Agitate-Solve (PAS) ad copy, LinkedIn/X posts, targeted hashtags, and commercial ad banner prompts.
- **Educational Masterclass:** Structured concept breakdown (TL;DR, intuition, real-world analogies, misconceptions, and 3-question self-assessment quiz).

### 3. 💻 Code Synthesis & Execution Sandbox (Prompt-to-Code)
- **Multi-Language Generation:** Python, JavaScript, TypeScript, SQL, HTML/CSS, and C++.
- **Architectural Diagrams:** Automatically renders visual **Mermaid flowcharts** and data topologies.
- **Algorithmic Profiling:** Generates asymptotic Big-O Time & Space complexity analysis.
- **Live Python Execution Sandbox:** Safely executes generated Python code inside an isolated memory runtime, capturing `stdout`, `stderr`, and runtime duration in milliseconds!

### 4. 👁️ Visual Reasoning & Image-to-Text Studio (Image-to-Text)
- **Flexible Seed Input:** Upload custom images (PNG, JPG, WEBP) or pick from pre-loaded product, nature, and diagram samples.
- **Task Modes:**
  1. Social Media Marketing & High-Engagement Copy.
  2. Technical Diagram & Microservice Architecture Deconstruction.
  3. Visual Story Starter (Atmospheric prologue from image mood).
  4. Accessibility & WCAG 2.2 Alt-Text Audit.
- **Feature Extraction:** Real-time perceptual metrics (perceived luminance, color temperature, dynamic contrast ratio, aspect ratio).

### 5. 💬 User Feedback Loop & Regeneration (Academic Requirement)
- **Qualitative Feedback:** One-click thumbs-up 👍 / thumbs-down 👎 recording.
- **Quantitative Evaluation:** 1-to-5 Star rating widget.
- **Active Regeneration:** One-click "🔄 Regenerate Output" button with dynamic seed variation.
- **Persistent Analytics:** Full session log stored in `data/history.json` with JSON export for submission evaluation.

---

## 🛠️ Project Structure

```plaintext
multimodal-creative-studio/
├── app.py                      # Main Streamlit Application & Multi-Tab Interface
├── modules/
│   ├── __init__.py
│   ├── model_manager.py        # Cloud & Autonomous API routing layer
│   ├── image_engine.py         # Text-to-Image & procedural canvas renderer
│   ├── text_engine.py          # Narrative, marketing copy & storyboard synthesizer
│   ├── code_engine.py          # Code generation, Mermaid diagrams & sandbox runner
│   ├── multimodal_vision.py    # Image-to-Text & visual reasoning engine
│   └── feedback_manager.py     # Feedback logging, ratings, and history export
├── assets/
│   ├── styles.css              # Custom UI enhancements (glassmorphism & badges)
│   └── sample_images/          # Pre-loaded images for instant multimodal testing
│       ├── sample_product.png
│       ├── sample_nature.png
│       └── sample_diagram.png
├── data/
│   └── history.json            # Persistent generation and feedback records
├── requirements.txt            # Python dependencies
├── README.md                   # Quick start and overview
└── PROJECT_REPORT.md           # Full academic mini-project documentation
```

---

## ⚡ Quick Start Guide

### 1. Prerequisites
- Python 3.10+ (Tested on Python 3.14)
- Pip package manager

### 2. Installation
```bash
cd /Users/rushikeshgawande/.gemini/antigravity/scratch/multimodal-creative-studio
python3 -m pip install -r requirements.txt
```

### 3. Running the Application
Launch the interactive web application:
```bash
python3 -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 4. (Optional) Configuring Live Cloud Models
- In the sidebar **🔑 Cloud API Credentials** expander:
  - Enter your **Google Gemini API Key** (`GEMINI_API_KEY`) for Gemini 2.5 Flash live inference.
  - Enter your **Hugging Face Token** (`HF_TOKEN`) for Hugging Face Hub models.
- *Note:* If keys are omitted, OmniCraft automatically operates in **Autonomous Engine Mode**, providing 100% offline functionality.

---

## 🧪 Demonstration & Test Cases

| Modality | Test Input | Expected Output |
| :--- | :--- | :--- |
| **Image** | "Cyberpunk city night with glowing holograms" + Cyberpunk style | High-res 768x768 artwork with neon palette, seed provenance, and PNG download. |
| **Story** | "Relic uncovered beneath Martian ice" + Sci-Fi | 3 structured chapters with narrative text + visual scene prompts + one-click illustration. |
| **Code** | "Token Bucket Rate Limiter in Python" | Clean code + Mermaid flowchart + Big-O analysis + live sandbox execution results. |
| **Vision** | Select `sample_product.png` + "Marketing Copy" | Perceptual visual audit (37.3 luminance, Cool palette) + Instagram copy + CTAs. |
| **Feedback** | Thumbs up + 5-Star rating on any generated artifact | Logged persistently to `data/history.json` and reflected in metrics dashboard. |

---

## 🎓 Academic Compliance
This project strictly satisfies all requirements of the **Multimodal Content Generation Mini-Project**:
- ✅ Input interface for prompt/seed across text, image, and code.
- ✅ Backend integration with pre-trained generative models (Gemini, FLUX/SD, Vision).
- ✅ Visualization/rendering interface for generated content (Markdown, PNG, Mermaid, Sandbox).
- ✅ User feedback collection (ratings, thumbs up/down) and dynamic regeneration capability.
