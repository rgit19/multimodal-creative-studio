# Nexa AI: Multimodal Creative Studio

> **Mini Project:** Multimodal Content Generation using Generative AI Models  
> **Domain:** Creative Studio, Storyboarding, Software Synthesis & Digital Marketing  
> **Stack:** Python, Streamlit, Google GenAI SDK, Gemini API, and Pillow.

---

## 📌 Executive Summary

**Nexa AI** is an end-to-end generative AI platform designed to synthesize, transform, and evaluate text, image, code, and vision tasks.

The system enables users to provide multimodal inputs (natural language descriptions, style parameters, seed images, and architectural requirements) to produce context-rich outputs. All AI-generated content is served by Google Gemini; a Gemini API key and network access are required.

---

## 🚀 Key Functional Modules

### 1. 🎨 Visual Studio (Text-to-Image Synthesis)
- **Prompt & Negative Prompting:** Fine-grained artistic control over desired and suppressed features.
- **Style Conditioning:** Photorealistic (DSLR 8K), Cyberpunk/Neon, Anime (Studio Ghibli style), Watercolor Fantasy, Modern 3D Minimalist, and Classic Oil Painting.
- **Aspect Ratio Control:** 1:1 Square, 16:9 Banner, 9:16 Mobile Story/Reel, 4:3 Classic.
- **Backend Model:** Gemini Nano Banana 2.1 native image generation, with prompt, style, negative-prompt, and aspect-ratio controls.
- **Exporting:** One-click full-resolution PNG download with generation provenance.

### 2. ✍️ Storytelling & Marketing Studio (Text-to-Text & Storyboarding)
- **Multi-Chapter Narratives:** Automatically weaves multi-act stories with dramatic conflict and tone modulation (Sci-Fi, Fantasy, Mystery).
- **Multimodal Storyboarding:** Each narrative chapter auto-generates a corresponding **Visual Scene Prompt**, which can be rendered directly in the Visual Studio with a single click!
- **Marketing Launch Kit:** Full Problem-Agitate-Solve (PAS) ad copy, LinkedIn/X posts, targeted hashtags, and commercial ad banner prompts.
- **Educational Masterclass:** Structured concept breakdown (TL;DR, intuition, real-world analogies, misconceptions, and 3-question self-assessment quiz).

### 3. 💻 Code Studio
- **Natural-Language Generation:** Describe a feature and generate code in Python, JavaScript, TypeScript, SQL, HTML/CSS, or C++.
- **Code Assistance:** Explain, debug, and refactor pasted code with optional issue details or refactoring goals.
- **Readable Output:** Syntax-highlighted code includes a copy control and source download; explanations and optional Mermaid diagrams are shown separately.
- **Safe by Default:** Generated code is displayed but is not executed by the application.
- **Architectural Diagrams:** Automatically renders visual **Mermaid flowcharts** and data topologies.
- **Algorithmic Profiling:** Generates asymptotic Big-O Time & Space complexity analysis.

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
- **Active Regeneration:** One-click "🔄 Regenerate Output" button that resubmits the current prompt.
- **Persistent Analytics:** Full session log stored in `data/history.json` with JSON export for submission evaluation.

### 6. ✦ Chat
- **Conversation UI:** Native Streamlit chat messages with Markdown rendering, streamed Gemini responses, and a compact welcome screen with writing, coding, debugging, and brainstorming suggestions.
- **Saved chats:** Conversations are stored alongside existing generation history, searchable, grouped by date, and can be reopened, renamed, deleted with confirmation, retried, or exported as Markdown.
- **Privacy and safety:** Gemini credentials stay server-side; code blocks are displayed but never executed. Copy response uses Streamlit's native code-block copy control.
- **Other workflows:** Storyboard, marketing, and educational generation remain available in the Chat workspace; Code, Images, Vision, History, and Settings remain in the main navigation.

---

## 🛠️ Project Structure

```plaintext
multimodal-creative-studio/
├── app.py                      # Main Streamlit Application & Multi-Tab Interface
├── modules/
│   ├── __init__.py
│   ├── model_manager.py        # Google Gemini API client and inference helpers
│   ├── image_engine.py         # Gemini text-to-image generation
│   ├── text_engine.py          # Narrative, marketing copy & storyboard synthesizer
│   ├── code_engine.py          # Code generation, explanation, debugging & refactoring
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
- Python 3.10 or later
- Pip package manager
- A Google Gemini API key to use generation features (the app can start without one)

### 2. Installation
```bash
cd multimodal-creative-studio-main
python3 -m pip install -r requirements.txt
```

### 3. Running the Application
Start the app from the repository root:
```bash
python3 -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 4. Configure Gemini
Create a key in [Google AI Studio](https://aistudio.google.com/apikey). For local development:

```bash
cp .env.example .env
# Edit .env and replace the placeholder GEMINI_API_KEY value.
```

For deployment, set the credential in the server environment or Streamlit secrets (`.streamlit/secrets.toml`). Never enter a key in the browser or commit `.env` or `secrets.toml`; `.gitignore` excludes both. Credential status is checked locally and does not send a test request.

| Environment variable | Required | Purpose |
| --- | --- | --- |
| `GEMINI_API_KEY` | Yes, for generation | Server-side Google Gemini credential; read from the environment, `.env`, or Streamlit secrets. |
| `GEMINI_TEXT_MODEL` | No | Default text/code/vision model. Supported values: `gemini-3.8-flash`, `gemini-3.5-flash-lite`. |
| `GEMINI_IMAGE_MODEL` | No | Default image model. Supported values: `gemini-nano-banana-2.1`, `imagen-3.0-generate-002`. |

Model selectors in the app can override the default models for the current session.

### Limitations and fallback behavior
- Without a configured key, the app starts and displays setup guidance, but generation is unavailable.
- A configured key is not proof of provider acceptance; invalid keys, unavailable models, quota errors, and provider timeouts are reported when a request is made.
- Gemini is the only provider. There is no automatic provider or model fallback; failures are surfaced to the user. A different supported model can be selected manually.
- Text and code requests stream responses. Requests have a 60-second timeout and at most two attempts for HTTP 408, 429, 500, 502, or 503; HTTP 504 is not automatically retried.
- Rate limiting permits 10 requests per minute per configured key per app process. Identical model/prompt requests are blocked while active and for two seconds after completion. Text prompts are limited to 50,000 characters.
- Streamed-text latency and provider-reported token usage are process-local aggregates. Token usage may be unavailable from the provider response.
- Closing a text stream or setting its cancellation event stops consumption and closes the provider stream. The current Streamlit interface does not provide a separate interactive cancel button.
- Generated code is displayed for review and is never executed by the application.
- Chat file attachments are not enabled. The separate Vision workflow continues to support image uploads.
- Conversation history is stored locally in `data/history.json`; it is not synchronized across separate app instances or users.
- Scrolling uses Streamlit's native chat layout; no custom browser JavaScript or third-party scrolling component is used.
- Availability and usage depend on the selected model, Google AI Studio project, region, quotas, and network access.

### Quality assurance
Validated locally on Python 3.14.6 with the project dependency constraints:

```bash
python3 -m unittest discover -s tests -v
python3 -m pip check
python3 -m py_compile app.py modules/*.py tests/*.py
git diff --check
```

Results for the final QA pass: **31 unit and Streamlit AppTest checks passed**; `pip check` reported no broken requirements; compilation and diff whitespace checks passed. Provider-dependent tests use mocks. No live Gemini calls or real-provider verification were performed. The repository has no configured or installed lint tool, so no lint run is claimed.

---

## 🧪 Demonstration & Test Cases

| Modality | Test Input | Expected Output |
| :--- | :--- | :--- |
| **Image** | "Cyberpunk city night with glowing holograms" + Cyberpunk style | Gemini-generated image with neon palette, selected aspect ratio, and PNG download. |
| **Story** | "Relic uncovered beneath Martian ice" + Sci-Fi | 3 structured chapters with narrative text + visual scene prompts + one-click illustration. |
| **Code** | "Token Bucket Rate Limiter with burst capacity" | Language-specific code, explanation, optional Mermaid flowchart, copy, and source download. |
| **Vision** | Select `sample_product.png` + "Marketing Copy" | Perceptual visual audit (37.3 luminance, Cool palette) + Instagram copy + CTAs. |
| **Feedback** | Thumbs up + 5-Star rating on any generated artifact | Logged persistently to `data/history.json` and reflected in metrics dashboard. |

---

## 🎓 Academic Compliance
This project strictly satisfies all requirements of the **Multimodal Content Generation Mini-Project**:
- ✅ Input interface for prompt/seed across text, image, and code.
- ✅ Backend integration with Google Gemini for text, image, and vision generation.
- ✅ Visualization/rendering interface for generated content (Markdown, PNG, Mermaid, syntax-highlighted code).
- ✅ User feedback collection (ratings, thumbs up/down) and dynamic regeneration capability.
