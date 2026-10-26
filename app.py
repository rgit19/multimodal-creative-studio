"""
OmniCraft Studio: Multimodal Generative AI Studio
Mini Project: Multimodal Content Generation using Generative AI Models
Supports Text-to-Image, Story & Narrative Synthesis, Code Synthesis & Execution Sandbox,
and Multimodal Vision Reasoning with Feedback Collection & Regeneration Loops.
"""

import os
import io
import json
import random
import time
from datetime import datetime
from PIL import Image
import streamlit as st

# Import core multimodal modules
from modules.feedback_manager import FeedbackManager
from modules.model_manager import model_manager, HAS_GEMINI_SDK, HAS_HF_SDK
from modules.image_engine import image_engine, STYLE_PRESETS, ASPECT_RATIOS
from modules.text_engine import text_engine
from modules.code_engine import code_engine
from modules.multimodal_vision import vision_engine

# --- Page Configuration ---
st.set_page_config(
    page_title="OmniCraft AI | Multimodal Generative Studio",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize persistent feedback manager
if "feedback_manager" not in st.session_state:
    st.session_state.feedback_manager = FeedbackManager("data/history.json")

# Initialize session state cache for active outputs
if "active_outputs" not in st.session_state:
    st.session_state.active_outputs = {
        "image": None,
        "image_meta": None,
        "image_record_id": None,
        "story": None,
        "story_record_id": None,
        "code": None,
        "code_record_id": None,
        "code_exec_result": None,
        "vision": None,
        "vision_record_id": None
    }

# Load CSS Styles
def load_custom_css():
    css_path = os.path.join(os.path.dirname(__file__), "assets", "styles.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_custom_css()

# --- Sidebar: Model Provider & Configuration ---
with st.sidebar:
    st.markdown("### ⚙️ Generative Model Hub")
    st.caption("Configure pre-trained backend integrations")

    # API Key Management
    with st.expander("🔑 Cloud API Credentials", expanded=False):
        st.markdown(
            "*(Optional)* Enter API keys for live Gemini or HuggingFace endpoints. "
            "If left blank, OmniCraft uses the high-fidelity **Autonomous Pre-trained Engine**."
        )
        gemini_input = st.text_input(
            "Google Gemini API Key",
            value=model_manager.gemini_api_key,
            type="password",
            help="Provides live access to Gemini 2.5 Flash for text, code, and vision."
        )
        hf_input = st.text_input(
            "Hugging Face Token",
            value=model_manager.hf_token,
            type="password",
            help="Enables direct queries to HuggingFace Inference API models."
        )

        if st.button("Save & Update Credentials", use_container_width=True):
            model_manager.update_keys(gemini_input, hf_input)
            st.success("API credentials updated successfully!")

    # Model Engine Status Indicators
    status = model_manager.get_status()
    st.markdown("---")
    st.markdown("#### 📡 Engine Connectivity")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        gemini_badge = "🟢 Ready" if status["gemini_ready"] else "⚪ Offline"
        st.metric("Gemini API", gemini_badge)
    with col_s2:
        hf_badge = "🟢 Ready" if status["huggingface_ready"] else "⚪ Offline"
        st.metric("Hugging Face", hf_badge)

    col_s3, col_s4 = st.columns(2)
    with col_s3:
        st.metric("Pollinations FLUX", "🟢 Active")
    with col_s4:
        st.metric("Local Engine", "🟢 100% Ready")

    st.markdown("---")
    st.markdown("#### 📊 Quick Metrics")
    stats = st.session_state.feedback_manager.get_statistics()
    st.write(f"**Total Generations:** {stats['total_generations']}")
    st.write(f"**Average User Rating:** {stats['avg_rating']} ⭐ ({stats['ratings_count']} reviews)")
    st.write(f"**Positive Feedback:** 👍 {stats['thumbs_up']} | 👎 {stats['thumbs_down']}")

    st.markdown("---")
    st.caption("OmniCraft AI Studio v1.0.0 | Academic Mini Project")


# --- Main Application Header ---
st.markdown("""
<div class="hero-header">
    <div class="hero-title">OmniCraft Multimodal Studio</div>
    <div class="hero-subtitle">
        Unified Generative AI Workbench: Visual Synthesis, Creative Storytelling, Code Generation, and Multimodal Vision Reasoning.
    </div>
</div>
""", unsafe_allow_html=True)


# --- Helper: Render Feedback Component ---
def render_feedback_widget(modality: str, record_id: str, prompt_text: str, on_regenerate_callback=None):
    """Reusable feedback, rating, and regeneration control widget."""
    if not record_id:
        return

    st.markdown("---")
    st.markdown("##### 💬 User Feedback & Regeneration")
    
    fb_col1, fb_col2, fb_col3 = st.columns([1.5, 2.5, 1.5])

    with fb_col1:
        st.write("**Was this output helpful?**")
        btn_c1, btn_c2 = st.columns(2)
        with btn_c1:
            if st.button("👍 Yes", key=f"thumb_up_{modality}_{record_id}", use_container_width=True):
                st.session_state.feedback_manager.update_feedback(record_id, feedback_type="thumbs_up")
                st.toast("Thank you for your feedback!", icon="👍")
        with btn_c2:
            if st.button("👎 No", key=f"thumb_down_{modality}_{record_id}", use_container_width=True):
                st.session_state.feedback_manager.update_feedback(record_id, feedback_type="thumbs_down")
                st.toast("Feedback recorded. We'll adjust the parameters!", icon="👎")

    with fb_col2:
        rating = st.slider(
            "Rate quality (1 - 5 stars):",
            min_value=1,
            max_value=5,
            value=5,
            key=f"star_rating_{modality}_{record_id}"
        )
        if st.button("Submit Rating", key=f"submit_rating_{modality}_{record_id}"):
            st.session_state.feedback_manager.update_feedback(record_id, rating=rating)
            st.toast(f"Rated {rating} stars! Saved.", icon="⭐")

    with fb_col3:
        st.write("**Enhanced Control:**")
        if on_regenerate_callback and st.button("🔄 Regenerate Output", key=f"regen_{modality}_{record_id}", use_container_width=True):
            on_regenerate_callback()


# --- Navigation Tabs ---
tab_image, tab_story, tab_code, tab_vision, tab_history, tab_docs = st.tabs([
    "🎨 Visual Studio (Text-to-Image)",
    "✍️ Story & Marketing (Text-to-Text)",
    "💻 Code Synthesis (Prompt-to-Code)",
    "👁️ Visual Reasoning (Image-to-Text)",
    "📊 Feedback & History Log",
    "📖 Project Report & Architecture"
])


# ==============================================================================
# TAB 1: VISUAL STUDIO (Text-to-Image)
# ==============================================================================
with tab_image:
    st.subheader("🎨 Multimodal Visual Generation & Art Studio")
    st.markdown(
        "Generate stunning illustrations, marketing banners, and conceptual artwork from natural language prompts. "
        "Integrates FLUX.1/Stable Diffusion cloud inference with autonomous procedural canvas rendering."
    )

    col_input, col_config = st.columns([2.5, 1.5])

    with col_input:
        img_prompt = st.text_area(
            "Prompt / Seed Description:",
            value="A futuristic neon city with flying vehicles, rainy reflective streets, and glowing holograms",
            height=100,
            help="Describe the scene, subject, atmosphere, and details."
        )
        img_negative = st.text_input(
            "Negative Prompt (Optional - what to avoid):",
            value="blurry, low resolution, distorted limbs, oversaturated, watermark",
            help="Concepts or artifacts you want the model to suppress."
        )

    with col_config:
        style_choice = st.selectbox("Aesthetic Style Preset:", list(STYLE_PRESETS.keys()), index=1)
        aspect_choice = st.selectbox("Aspect Ratio / Dimensions:", list(ASPECT_RATIOS.keys()), index=0)
        
        with st.expander("Advanced Diffusion Parameters", expanded=False):
            img_seed = st.number_input("Random Seed (0 for random):", min_value=0, max_value=9999999, value=0)
            img_steps = st.slider("Sampling Steps:", min_value=15, max_value=50, value=25)
            img_guidance = st.slider("Guidance Scale (CFG):", min_value=1.0, max_value=15.0, value=7.5, step=0.5)
            use_cloud = st.checkbox("Query Public FLUX/Diffusion Cloud", value=True)

    def trigger_image_generation(custom_prompt=None, seed_override=None):
        active_prompt = custom_prompt or img_prompt
        s = seed_override if seed_override is not None else (img_seed if img_seed > 0 else random.randint(1000, 999999))
        with st.spinner("✨ Synthesizing visual artwork..."):
            generated_img, meta = image_engine.generate_image(
                prompt=active_prompt,
                negative_prompt=img_negative,
                style_preset=style_choice,
                aspect_ratio_key=aspect_choice,
                seed=s,
                steps=img_steps,
                guidance_scale=img_guidance,
                prefer_cloud=use_cloud
            )
            # Record in feedback manager
            rec_id = st.session_state.feedback_manager.add_generation(
                modality="image",
                prompt=active_prompt,
                output=f"Image generated with {meta['source_model']} ({meta['width']}x{meta['height']})",
                model_name=meta["source_model"],
                parameters=meta
            )
            st.session_state.active_outputs["image"] = generated_img
            st.session_state.active_outputs["image_meta"] = meta
            st.session_state.active_outputs["image_record_id"] = rec_id

    if st.button("🚀 Generate Visual Artwork", type="primary", use_container_width=True):
        trigger_image_generation()

    # Visual Output Display
    if st.session_state.active_outputs["image"]:
        cur_img = st.session_state.active_outputs["image"]
        cur_meta = st.session_state.active_outputs["image_meta"]
        rec_id = st.session_state.active_outputs["image_record_id"]

        st.markdown("---")
        st.markdown("#### 🖼️ Generated Visual Artifact")

        # Provenance Badges
        badge_html = f"""
        <div style="margin-bottom: 12px;">
            <span class="badge-model">{cur_meta['source_model']}</span>
            <span class="badge-speed">⏱️ {cur_meta['generation_time_ms']} ms</span>
            <span class="badge-modality">📐 {cur_meta['width']}x{cur_meta['height']}</span>
            <span style="font-size: 0.85rem; color: #94a3b8; margin-left: 10px;">Seed: <code>{cur_meta['seed']}</code></span>
        </div>
        """
        st.markdown(badge_html, unsafe_allow_html=True)

        img_col1, img_col2 = st.columns([3, 1.5])
        with img_col1:
            st.image(cur_img, use_container_width=True, caption=f"Prompt: {cur_meta['prompt']}")
        with img_col2:
            st.markdown("##### 📥 Export Artifact")
            buf = io.BytesIO()
            cur_img.save(buf, format="PNG")
            st.download_button(
                label="⬇️ Download Full-Res PNG",
                data=buf.getvalue(),
                file_name=f"omnicraft_gen_{cur_meta['seed']}.png",
                mime="image/png",
                use_container_width=True
            )
            st.markdown("##### 🔍 Generation Provenance")
            st.json({
                "Model": cur_meta["source_model"],
                "Style": cur_meta["style_preset"],
                "Resolution": f"{cur_meta['width']}x{cur_meta['height']}",
                "CFG Scale": cur_meta["guidance_scale"],
                "Steps": cur_meta["steps"],
                "Seed": cur_meta["seed"]
            })

        # Feedback & Regeneration Widget
        render_feedback_widget(
            modality="image",
            record_id=rec_id,
            prompt_text=cur_meta["prompt"],
            on_regenerate_callback=lambda: trigger_image_generation(seed_override=random.randint(1000, 999999))
        )


# ==============================================================================
# TAB 2: STORY & MARKETING (Text-to-Text & Storyboard)
# ==============================================================================
with tab_story:
    st.subheader("✍️ Creative Storytelling & Marketing Copy Studio")
    st.markdown(
        "Transform seed ideas into multi-chapter narratives, marketing launch bundles, or educational modules. "
        "Includes multimodal cross-linking to auto-generate visual scene prompts."
    )

    task_type = st.radio(
        "Select Content Synthesis Mode:",
        ["Multi-Chapter Narrative & Storyboard", "Complete Marketing Launch Campaign", "Educational Masterclass Explainer"],
        horizontal=True
    )

    if task_type == "Multi-Chapter Narrative & Storyboard":
        col_s1, col_s2, col_s3 = st.columns([2.5, 1, 1])
        with col_s1:
            story_seed = st.text_input(
                "Story Premise / Seed Prompt:",
                value="An archaeologist uncovers an ancient quantum crystalline relic beneath the Martian ice"
            )
        with col_s2:
            genre_choice = st.selectbox("Genre:", ["Sci-Fi Cyberpunk", "Fantasy & Mythic", "Mystery & Thriller"])
        with col_s3:
            tone_choice = st.selectbox("Tone:", ["Thrilling & Futuristic", "Poetic & Ethereal", "Humorous & Fast-Paced", "Dark & Suspenseful"])

        num_chapters = st.slider("Number of Chapters:", min_value=1, max_value=3, value=3)

        def trigger_story_generation():
            with st.spinner("🖋️ Weaving narrative chapters and visual storyboards..."):
                res = text_engine.generate_story(
                    seed_prompt=story_seed,
                    genre=genre_choice,
                    tone=tone_choice,
                    num_chapters=num_chapters
                )
                rec_id = st.session_state.feedback_manager.add_generation(
                    modality="story",
                    prompt=story_seed,
                    output=res["full_text"],
                    model_name=res["model_used"],
                    parameters={"genre": genre_choice, "tone": tone_choice, "chapters": num_chapters}
                )
                st.session_state.active_outputs["story"] = res
                st.session_state.active_outputs["story_record_id"] = rec_id

        if st.button("📖 Generate Story & Storyboard", type="primary", use_container_width=True):
            trigger_story_generation()

        if st.session_state.active_outputs["story"]:
            story_data = st.session_state.active_outputs["story"]
            s_rec_id = st.session_state.active_outputs["story_record_id"]

            st.markdown("---")
            st.markdown(f"### {story_data['title']}")
            st.markdown(f"""
            <span class="badge-model">{story_data['model_used']}</span>
            <span class="badge-speed">⏱️ {story_data['generation_time_ms']} ms</span>
            <span class="badge-modality">📚 {len(story_data['chapters'])} Chapters</span>
            """, unsafe_allow_html=True)

            ch_tabs = st.tabs([f"Act {c['chapter_index']}: {c['title']}" for c in story_data["chapters"]])
            for idx, tab_c in enumerate(ch_tabs):
                with tab_c:
                    ch = story_data["chapters"][idx]
                    st.write(ch["text"])
                    st.info(f"🎨 **Visual Scene Prompt:** `{ch['scene_prompt']}`")
                    
                    if st.button(f"🎨 Render Illustration for Chapter {ch['chapter_index']}", key=f"render_ch_{idx}"):
                        st.session_state.active_outputs["image"] = None  # Switch focus
                        # Auto-generate scene in Visual Studio
                        generated_img, meta = image_engine.generate_image(
                            prompt=ch["scene_prompt"],
                            style_preset="Photorealistic" if genre_choice != "Fantasy & Mythic" else "Watercolor Fantasy",
                            aspect_ratio_key="16:9 Landscape (896x512)"
                        )
                        st.session_state.active_outputs["image"] = generated_img
                        st.session_state.active_outputs["image_meta"] = meta
                        st.session_state.active_outputs["image_record_id"] = st.session_state.feedback_manager.add_generation(
                            modality="image",
                            prompt=ch["scene_prompt"],
                            output="Storyboard cross-generation illustration",
                            model_name=meta["source_model"],
                            parameters=meta
                        )
                        st.success("Scene illustration synthesized! Open the '🎨 Visual Studio' tab to view and download it.")

            render_feedback_widget(
                modality="story",
                record_id=s_rec_id,
                prompt_text=story_seed,
                on_regenerate_callback=trigger_story_generation
            )

    elif task_type == "Complete Marketing Launch Campaign":
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            p_name = st.text_input("Product / Service Name:", value="AuraSound X1 Pro ANC Earbuds")
            p_audience = st.text_input("Target Audience:", value="Audiophiles, Remote Tech Professionals, Commuters")
        with m_col2:
            p_benefits = st.text_input("Core Benefits / Value Prop:", value="45dB Adaptive Active Noise Cancellation, 40-hour Battery, Lossless Spatial Audio")
            p_tone = st.selectbox("Tone:", ["High-Energy & Persuasive", "Minimalist & Elegant", "Punchy & Direct"])

        if st.button("🚀 Generate Marketing Campaign Bundle", type="primary", use_container_width=True):
            with st.spinner("Generating copy, ad frameworks, and social assets..."):
                m_res = text_engine.generate_marketing_campaign(p_name, p_audience, p_benefits, p_tone)
                m_rec_id = st.session_state.feedback_manager.add_generation(
                    modality="marketing",
                    prompt=f"{p_name} for {p_audience}",
                    output=m_res["content"],
                    model_name=m_res["model_used"]
                )
                st.session_state.active_outputs["story"] = m_res
                st.session_state.active_outputs["story_record_id"] = m_rec_id

        if st.session_state.active_outputs["story"] and "product_name" in st.session_state.active_outputs["story"]:
            m_res = st.session_state.active_outputs["story"]
            st.markdown("---")
            st.markdown(m_res["content"])
            st.info(f"🖼️ **Suggested Ad Banner Visual Prompt:** `{m_res['banner_prompt']}`")
            render_feedback_widget(
                modality="marketing",
                record_id=st.session_state.active_outputs["story_record_id"],
                prompt_text=m_res["product_name"]
            )

    else:  # Educational Masterclass Explainer
        e_col1, e_col2 = st.columns([3, 1])
        with e_col1:
            edu_topic = st.text_input("Educational Topic to Explain:", value="Transformer Architecture & Multi-Head Self-Attention")
        with e_col2:
            edu_level = st.selectbox("Target Level:", ["High School Student", "Undergraduate CS Student", "Senior ML Engineer"])

        if st.button("🎓 Generate Masterclass Explainer", type="primary", use_container_width=True):
            with st.spinner("Structuring knowledge breakdown and self-assessment quiz..."):
                e_res = text_engine.generate_educational_explainer(edu_topic, edu_level)
                e_rec_id = st.session_state.feedback_manager.add_generation(
                    modality="education",
                    prompt=edu_topic,
                    output=e_res["content"],
                    model_name=e_res["model_used"]
                )
                st.session_state.active_outputs["story"] = e_res
                st.session_state.active_outputs["story_record_id"] = e_rec_id

        if st.session_state.active_outputs["story"] and "topic" in st.session_state.active_outputs["story"]:
            e_res = st.session_state.active_outputs["story"]
            st.markdown("---")
            st.markdown(e_res["content"])
            render_feedback_widget(
                modality="education",
                record_id=st.session_state.active_outputs["story_record_id"],
                prompt_text=e_res["topic"]
            )


# ==============================================================================
# TAB 3: CODE SYNTHESIS (Prompt-to-Code & Sandbox Runner)
# ==============================================================================
with tab_code:
    st.subheader("💻 Multimodal Code Synthesis & Execution Sandbox")
    st.markdown(
        "Synthesize production-grade code, visual Mermaid architecture diagrams, and algorithmic complexity breakdowns. "
        "Test and verify Python code directly within the interactive execution sandbox."
    )

    c_col1, c_col2, c_col3 = st.columns([3, 1, 1])
    with c_col1:
        code_prompt = st.text_area(
            "Natural Language Specification / Requirement:",
            value="Implement a Token Bucket Rate Limiter with burst capacity and dynamic refill in Python",
            height=85
        )
    with c_col2:
        code_lang = st.selectbox("Target Language:", ["Python", "JavaScript", "TypeScript", "SQL", "HTML/CSS", "C++"])
    with c_col3:
        code_level = st.selectbox("Complexity Level:", ["Production-Ready", "Minimalist Clean", "Enterprise Resilient"])

    def trigger_code_generation():
        with st.spinner("⚡ Synthesizing code and architectural data flow..."):
            c_res = code_engine.generate_code(
                prompt=code_prompt,
                language=code_lang,
                complexity=code_level
            )
            c_rec_id = st.session_state.feedback_manager.add_generation(
                modality="code",
                prompt=code_prompt,
                output=c_res["code"],
                model_name=c_res["model_used"],
                parameters={"language": code_lang, "complexity": code_level}
            )
            st.session_state.active_outputs["code"] = c_res
            st.session_state.active_outputs["code_record_id"] = c_rec_id
            st.session_state.active_outputs["code_exec_result"] = None

    if st.button("⚙️ Synthesize Code Solution", type="primary", use_container_width=True):
        trigger_code_generation()

    if st.session_state.active_outputs["code"]:
        c_res = st.session_state.active_outputs["code"]
        c_rec_id = st.session_state.active_outputs["code_record_id"]

        st.markdown("---")
        st.markdown(f"""
        <span class="badge-model">{c_res['model_used']}</span>
        <span class="badge-speed">⏱️ {c_res['generation_time_ms']} ms</span>
        <span class="badge-modality">💻 {c_res['language']}</span>
        """, unsafe_allow_html=True)

        st.markdown(f"**Complexity Profile:** `{c_res['complexity']}`")

        # Code Block Presentation
        st.code(c_res["code"], language=c_res["language"].lower())

        # Architectural Flowchart Diagram
        if c_res.get("mermaid_chart"):
            st.markdown("##### 📐 Synthesized Architecture / Flowchart")
            st.markdown(f"```mermaid\n{c_res['mermaid_chart']}\n```")

        # Architectural Explanation
        with st.expander("📖 Architectural Breakdown & Logic Explanation", expanded=True):
            st.markdown(c_res["explanation"])

        # Python Execution Sandbox
        if c_res["language"].lower() == "python":
            st.markdown("##### 🧪 Interactive Execution Sandbox")
            st.caption("Execute this Python implementation inside the controlled memory runtime:")
            
            if st.button("▶️ Run in Python Sandbox", key="exec_sandbox_btn"):
                with st.spinner("Executing script in sandbox..."):
                    exec_res = code_engine.execute_python_code(c_res["code"])
                    st.session_state.active_outputs["code_exec_result"] = exec_res

            if st.session_state.active_outputs["code_exec_result"]:
                exec_data = st.session_state.active_outputs["code_exec_result"]
                if exec_data["success"]:
                    st.success(f"Execution Succeeded in {exec_data['execution_time_ms']} ms")
                    st.code(exec_data["stdout"], language="text")
                else:
                    st.error("Execution encountered an exception:")
                    st.code(exec_data["stderr"], language="text")

        render_feedback_widget(
            modality="code",
            record_id=c_rec_id,
            prompt_text=c_res["prompt"],
            on_regenerate_callback=trigger_code_generation
        )


# ==============================================================================
# TAB 4: VISUAL REASONING (Image-to-Text & Vision)
# ==============================================================================
with tab_vision:
    st.subheader("👁️ Multimodal Visual Reasoning & Marketing Vision Studio")
    st.markdown(
        "Provide an image seed (upload your own or choose a pre-loaded sample) and generate "
        "multimodal marketing campaigns, architectural deconstructions, or novel story openers."
    )

    v_col1, v_col2 = st.columns([1.5, 2])

    with v_col1:
        st.markdown("##### 1. Select Visual Input")
        image_source = st.radio("Choose Input Source:", ["Use Pre-loaded Sample Image", "Upload My Own Image"], horizontal=True)

        selected_image = None
        if image_source == "Use Pre-loaded Sample Image":
            samples = {
                "🎧 AuraSound Product Mockup": "assets/sample_images/sample_product.png",
                "🌄 Emerald Ridge Sunset Nature": "assets/sample_images/sample_nature.png",
                "🏗️ Microservice Architecture Diagram": "assets/sample_images/sample_diagram.png"
            }
            sample_choice = st.selectbox("Select Sample Image:", list(samples.keys()))
            sample_path = os.path.join(os.path.dirname(__file__), samples[sample_choice])
            if os.path.exists(sample_path):
                selected_image = Image.open(sample_path)
        else:
            uploaded_file = st.file_uploader("Upload Image (PNG, JPG, WEBP):", type=["png", "jpg", "jpeg", "webp"])
            if uploaded_file is not None:
                selected_image = Image.open(uploaded_file)

        if selected_image:
            st.image(selected_image, caption="Active Visual Input", use_container_width=True)

    with v_col2:
        st.markdown("##### 2. Multimodal Generation Directives")
        vision_task = st.selectbox(
            "Select Multimodal Generation Task:",
            [
                "Marketing Copy & Social Media",
                "Technical Diagram & Architecture Deconstruction",
                "Visual Story Starter",
                "Accessibility & Detailed Visual Audit"
            ]
        )
        custom_vision_notes = st.text_input(
            "Specific Directives or Audience Focus (Optional):",
            placeholder="e.g. Focus on tech founders; highlight portability and audio clarity"
        )

        def trigger_vision_generation():
            if selected_image is None:
                st.warning("Please select or upload an image first.")
                return

            with st.spinner("👁️ Analyzing visual features and synthesizing contextual content..."):
                v_res = vision_engine.analyze_image_and_generate(
                    image=selected_image,
                    task_mode=vision_task,
                    custom_instructions=custom_vision_notes
                )
                v_rec_id = st.session_state.feedback_manager.add_generation(
                    modality="vision",
                    prompt=f"{vision_task} on image ({v_res['features']['dimensions']})",
                    output=v_res["analysis"],
                    model_name=v_res["model_used"],
                    parameters={"features": v_res["features"]}
                )
                st.session_state.active_outputs["vision"] = v_res
                st.session_state.active_outputs["vision_record_id"] = v_rec_id

        if st.button("🔮 Analyze & Synthesize Content", type="primary", use_container_width=True):
            trigger_vision_generation()

        if st.session_state.active_outputs["vision"]:
            v_res = st.session_state.active_outputs["vision"]
            v_rec_id = st.session_state.active_outputs["vision_record_id"]

            st.markdown("---")
            st.markdown(f"""
            <span class="badge-model">{v_res['model_used']}</span>
            <span class="badge-speed">⏱️ {v_res['generation_time_ms']} ms</span>
            <span class="badge-modality">👁️ {v_res['task_mode']}</span>
            """, unsafe_allow_html=True)

            with st.expander("📊 Extracted Perceptual Visual Features", expanded=False):
                st.json(v_res["features"])

            st.markdown(v_res["analysis"])

            render_feedback_widget(
                modality="vision",
                record_id=v_rec_id,
                prompt_text=f"Visual Analysis: {vision_task}",
                on_regenerate_callback=trigger_vision_generation
            )


# ==============================================================================
# TAB 5: FEEDBACK & GENERATION HISTORY LOG
# ==============================================================================
with tab_history:
    st.subheader("📊 Feedback Analytics & Generation History Log")
    st.markdown(
        "Review persistent logs of all multimodal generations, inspect user feedback, "
        "and export data for academic evaluation or performance auditing."
    )

    stats = st.session_state.feedback_manager.get_statistics()

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Total Outputs Generated", stats["total_generations"])
    with m_col2:
        st.metric("Average Rating", f"{stats['avg_rating']} ⭐" if stats["ratings_count"] > 0 else "N/A")
    with m_col3:
        st.metric("Positive Ratings", f"👍 {stats['thumbs_up']}")
    with m_col4:
        st.metric("Adjustment Requests", f"👎 {stats['thumbs_down']}")

    st.markdown("---")

    filter_col, export_col = st.columns([2, 1])
    with filter_col:
        filter_modality = st.selectbox(
            "Filter History by Modality:",
            ["All Modalities", "image", "story", "marketing", "code", "education", "vision"]
        )
    with export_col:
        history_data = st.session_state.feedback_manager.get_history(
            modality=None if filter_modality == "All Modalities" else filter_modality
        )
        json_export = json.dumps(history_data, indent=2)
        st.download_button(
            "📥 Export History as JSON",
            data=json_export,
            file_name=f"omnicraft_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )

    if not history_data:
        st.info("No generation records found yet. Generate text, images, or code in the studio to populate this log.")
    else:
        for item in history_data:
            with st.expander(f"[{item['modality'].upper()}] {item['prompt'][:60]}... ({item['timestamp']})", expanded=False):
                st.write(f"**Record ID:** `{item['id']}` | **Model:** `{item['model_name']}`")
                st.write(f"**Rating:** {item.get('rating') or 'Unrated'} ⭐ | **Feedback:** {item.get('feedback_type') or 'None'}")
                st.write(f"**Prompt:** {item['prompt']}")
                st.markdown("**Output Preview:**")
                st.text(str(item["output"])[:400] + ("..." if len(str(item["output"])) > 400 else ""))
                if item.get("parameters"):
                    st.caption(f"Parameters: {item['parameters']}")


# ==============================================================================
# TAB 6: PROJECT REPORT & ARCHITECTURE
# ==============================================================================
with tab_docs:
    st.subheader("📖 Academic Mini-Project Specification & Architecture")
    st.markdown("""
### Multimodal Content Generation using Generative AI Models
**Project Name:** OmniCraft Studio  
**Domain Focus:** Multi-Domain Creative Studio (Storytelling, Visual Art, Code Synthesis, Marketing)  
**System Architecture:** Decoupled Multimodal AI Pipeline with Dynamic Cloud/Autonomous Failover  

---

#### 1. System Architecture Overview
```mermaid
flowchart TD
    User([User / Creator]) --> UI[Streamlit Multimodal Workbench]
    
    subgraph UI_Modules [User Interaction & Prompt Interfaces]
        P1[Visual Prompt & Style Conditioning]
        P2[Narrative Seed & Genre Control]
        P3[Code Specification & Language Selector]
        P4[Image Upload & Feature Inspector]
    end
    
    UI --> UI_Modules
    
    subgraph Model_Manager [Model Management & Routing Layer]
        Router{API Key & Network Available?}
        Router -->|Yes| Cloud[Cloud Pre-Trained Models]
        Router -->|No / Fallback| Local[Autonomous Generative Engines]
        
        Cloud --> M1[Google Gemini 2.5 Flash]
        Cloud --> M2[Pollinations / FLUX.1 Diffusion]
        Cloud --> M3[Hugging Face Hub]
        
        Local --> L1[Perlin Harmonic Procedural Visuals]
        Local --> L2[Deterministic Creative Narrative Graph]
        Local --> L3[Syntax-Verified Code Synthesizer]
        Local --> L4[Pillow Perceptual Vision Analyzer]
    end
    
    UI_Modules --> Router
    
    subgraph Execution_And_Feedback [Output Rendering & Feedback Loop]
        R1[Image Renderer & PNG Exporter]
        R2[Markdown Storyboard & Audio-Ready Narration]
        R3[Live Python Execution Sandbox]
        R4[Thumbs Up/Down & Star Rating Collector]
        R5[Persistent JSON History Store]
    end
    
    Model_Manager --> Execution_And_Feedback
    Execution_And_Feedback -.->|Regenerate with Variations| Router
```

---

#### 2. Key Functional Highlights
1. **Multimodal Input Interface:**
   - Text prompts with negative prompting and style conditioning (Photorealistic, Cyberpunk, Ghibli, Fantasy).
   - Parameter controls: Sampling steps, guidance scale (CFG), aspect ratio formatting, random seed.
   - Multimodal image upload with automatic luminance, contrast, and dominant color histogram extraction.
2. **Pre-Trained Generative Model Backend Integration:**
   - **Google Gemini 2.5 Flash** for rich text narratives, code generation, and computer vision reasoning.
   - **FLUX.1 / Stable Diffusion** for text-to-image synthesis.
   - **Autonomous Procedural Fallback Engine** guaranteeing 100% offline demonstration reliability.
3. **Output Visualization & Interactive Sandboxes:**
   - High-resolution rendering with provenance metadata (seed, latency, model tags).
   - Live **Python Execution Sandbox** capturing standard output, execution duration, and errors safely.
   - Syntax-highlighted code and visual Mermaid diagrams.
4. **Enhanced User Control & Feedback Loop:**
   - Qualitative thumbs up / thumbs down and 1-to-5 star rating system.
   - Regeneration with variation seeds and parameter adjustments.
   - Persistent generation history with JSON export.
    """)
