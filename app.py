"""
Nexa AI: Multimodal Generative Studio
Brand: Nexa AI | Tagline: Create. Code. Imagine.
Supports Chat/Story, Code Generation & Assistance, Visual Studio,
Visual Reasoning (Vision), Feedback Analytics, and Model Settings.
"""

import os
import io
import json
import time
import uuid
from datetime import datetime
from PIL import Image
from google.genai.errors import APIError
import streamlit as st

# Import core multimodal modules (preserving 100% existing functionality)
from modules.feedback_manager import FeedbackManager
from modules.model_manager import (
    model_manager,
    SUPPORTED_TEXT_MODELS,
    SUPPORTED_IMAGE_MODELS,
    DEFAULT_TEXT_MODEL,
    DEFAULT_IMAGE_MODEL,
    TEXT_MODEL,
    IMAGE_MODEL,
    resolve_api_key,
    load_env_file,
    sanitize_error_message,
)
from modules.image_engine import image_engine, STYLE_PRESETS, ASPECT_RATIOS
from modules.text_engine import text_engine
from modules.code_engine import code_engine
from modules.multimodal_vision import vision_engine

# --- Page Configuration ---
st.set_page_config(
    page_title="Nexa AI | Create. Code. Imagine.",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Auto-load .env from project root if available
load_env_file()

# Resolve credentials from server-side configuration; never put secrets in session state.
gemini_api_key, _ = resolve_api_key()

# Initialize active model selections in session state
if "selected_text_model" not in st.session_state:
    st.session_state.selected_text_model = DEFAULT_TEXT_MODEL

if "selected_image_model" not in st.session_state:
    st.session_state.selected_image_model = DEFAULT_IMAGE_MODEL

# Comprehensive Provider & Credential Status
provider_status = model_manager.get_provider_status(
    text_model=st.session_state.selected_text_model,
    image_model=st.session_state.selected_image_model,
)

# Persistent data directory relative to repository root
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DATA_FILE = os.path.join(DATA_DIR, "history.json")
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize persistent feedback manager
if "feedback_manager" not in st.session_state:
    st.session_state.feedback_manager = FeedbackManager(DATA_FILE)

# Initialize navigation state
if "nav_page" not in st.session_state:
    st.session_state.nav_page = "💬 Chat"

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
        "vision": None,
        "vision_record_id": None
    }

if "code_request_in_progress" not in st.session_state:
    st.session_state.code_request_in_progress = False

if "code_request_error" not in st.session_state:
    st.session_state.code_request_error = None

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = str(uuid.uuid4())

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "chat_request_in_progress" not in st.session_state:
    st.session_state.chat_request_in_progress = False

if "chat_request_error" not in st.session_state:
    st.session_state.chat_request_error = None

# Load Custom CSS
def load_custom_css():
    css_path = os.path.join(os.path.dirname(__file__), "assets", "styles.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_custom_css()


def create_stream_callback(placeholder):
    """Render streamed text periodically without retaining request data globally."""
    chunks = []
    last_rendered = 0.0

    def render_chunk(chunk):
        nonlocal last_rendered
        chunks.append(chunk)
        now = time.monotonic()
        if len(chunks) == 1 or now - last_rendered >= 0.15:
            placeholder.markdown("".join(chunks))
            last_rendered = now

    return render_chunk


def run_chat_request(prompt: str, retry_record_id=None):
    """Run one conversational turn through the existing Gemini text adapter."""
    prompt = prompt.strip()
    if not prompt:
        st.session_state.chat_request_error = "Enter a message before sending."
        return
    if st.session_state.chat_request_in_progress:
        st.session_state.chat_request_error = (
            "A response is already being generated. Please wait for it to finish."
        )
        return
    if not provider_status["is_ready"]:
        st.session_state.chat_request_error = provider_status["message"]
        return

    messages = st.session_state.chat_messages
    if retry_record_id is None:
        messages.append({"role": "user", "content": prompt})

    retry_index = (
        next(
            (
                index
                for index, message in enumerate(messages)
                if message.get("record_id") == retry_record_id
            ),
            None,
        )
        if retry_record_id is not None
        else None
    )
    if retry_index is not None:
        context_messages = messages[: max(0, retry_index - 1)]
    else:
        context_messages = messages[:-1]

    context = "\n".join(
        f"{message['role'].title()}: {message['content']}"
        for message in context_messages[-12:]
    )[-20_000:]
    request_prompt = (
        f"Conversation context:\n{context}\n\nNew user message:\n{prompt}"
        if context
        else prompt
    )
    if retry_index is not None:
        if retry_index == 0 or messages[retry_index - 1].get("role") != "user":
            st.session_state.chat_request_error = (
                "The saved user message for this response could not be found."
            )
            return

    st.session_state.chat_request_in_progress = True
    st.session_state.chat_request_error = None
    with st.chat_message("assistant", avatar=":material/bolt:"):
        with st.status(":shimmer[Thinking]", type="compact") as status:
            response_slot = st.empty()
            try:
                response = model_manager.generate_text_gemini(
                    prompt=request_prompt,
                    api_key=gemini_api_key,
                    model=st.session_state.selected_text_model,
                    system_instruction=(
                        "You are Nexa AI, a helpful creative and coding assistant. "
                        "Give clear, accurate answers. Use Markdown and fenced code "
                        "blocks where useful. Never claim to run code."
                    ),
                    on_chunk=create_stream_callback(response_slot),
                    allow_recent_duplicate=retry_record_id is not None,
                )
                response_slot.markdown(response)
                status.update(label="Response complete", state="complete")
                assistant_message = {
                    "role": "assistant",
                    "content": response,
                    "record_id": retry_record_id,
                }
                if retry_record_id is None:
                    title = next(
                        (
                            conversation["title"]
                            for conversation in st.session_state.feedback_manager.get_chat_conversations()
                            if conversation["id"] == st.session_state.current_chat_id
                        ),
                        prompt.replace("\n", " ")[:48].strip() or "New chat",
                    )
                    turn_index = sum(
                        1
                        for item in messages
                        if item.get("role") == "user"
                    ) - 1
                    record_id = st.session_state.feedback_manager.add_chat_turn(
                        conversation_id=st.session_state.current_chat_id,
                        title=title,
                        turn_index=turn_index,
                        prompt=prompt,
                        response=response,
                        model_name=st.session_state.selected_text_model,
                    )
                    assistant_message["record_id"] = record_id
                    messages.append(assistant_message)
                else:
                    st.session_state.feedback_manager.update_chat_turn(
                        retry_record_id,
                        response,
                        st.session_state.selected_text_model,
                    )
                    messages[retry_index] = assistant_message
            except (APIError, RuntimeError, TimeoutError, ValueError) as error:
                status.update(label="Response failed", state="error")
                st.session_state.chat_request_error = sanitize_error_message(
                    error, gemini_api_key
                )
                st.error(st.session_state.chat_request_error)
            finally:
                st.session_state.chat_request_in_progress = False


# --- Reusable Feedback & Regeneration Component ---
def render_feedback_widget(modality: str, record_id: str, prompt_text: str, on_regenerate_callback=None):
    """Reusable feedback, rating, and regeneration control widget."""
    if not record_id:
        return

    st.markdown('<div class="feedback-box">', unsafe_allow_html=True)
    st.markdown("##### 💬 Feedback & Regeneration")

    fb_col1, fb_col2, fb_col3 = st.columns([1.5, 2.5, 1.5])

    with fb_col1:
        st.write("**Was this helpful?**")
        btn_c1, btn_c2 = st.columns(2)
        with btn_c1:
            if st.button("👍 Yes", key=f"thumb_up_{modality}_{record_id}", width="stretch"):
                st.session_state.feedback_manager.update_feedback(record_id, feedback_type="thumbs_up")
                st.toast("Thank you for your feedback!", icon="👍")
        with btn_c2:
            if st.button("👎 No", key=f"thumb_down_{modality}_{record_id}", width="stretch"):
                st.session_state.feedback_manager.update_feedback(record_id, feedback_type="thumbs_down")
                st.toast("Feedback recorded. We'll adjust parameters!", icon="👎")

    with fb_col2:
        rating = st.slider(
            "Quality rating (1-5 stars):",
            min_value=1,
            max_value=5,
            value=5,
            key=f"star_rating_{modality}_{record_id}"
        )
        if st.button("Submit Rating", key=f"submit_rating_{modality}_{record_id}"):
            st.session_state.feedback_manager.update_feedback(record_id, rating=rating)
            st.toast(f"Rated {rating} stars! Saved.", icon="⭐")

    with fb_col3:
        st.write("**Need an alternative?**")
        if on_regenerate_callback and st.button("🔄 Retry / Regenerate", key=f"regen_{modality}_{record_id}", width="stretch"):
            on_regenerate_callback()

    st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# SIDEBAR: BRAND, NAVIGATION, PROVIDER STATUS, MODEL CONTROLS & METRICS
# ==============================================================================
with st.sidebar:
    st.markdown("""
    <div class="nexa-sidebar-brand">
        <div class="nexa-sidebar-title">✦ Nexa AI</div>
        <div class="nexa-sidebar-tagline">Your creative assistant</div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.nav_page == "💬 Chat":
        if st.button(
            "＋  New chat",
            key="new_chat_button",
            type="primary",
            width="stretch",
            disabled=st.session_state.chat_request_in_progress,
        ):
            st.session_state.current_chat_id = str(uuid.uuid4())
            st.session_state.chat_messages = []
            st.session_state.chat_request_error = None
            st.rerun()

        st.text_input(
            "Search conversations",
            key="chat_search_query",
            placeholder="Search chats",
        )
        conversations = st.session_state.feedback_manager.get_chat_conversations()
        search_query = st.session_state.get("chat_search_query", "").strip().lower()
        if search_query:
            conversations = [
                conversation
                for conversation in conversations
                if search_query
                in f"{conversation['title']} {conversation['preview']}".lower()
            ]

        today = datetime.now().astimezone().date()
        date_groups = {"Today": [], "Previous 7 days": [], "Earlier": []}
        for conversation in conversations:
            try:
                updated = datetime.fromisoformat(conversation["updated_at"]).date()
            except (TypeError, ValueError):
                updated = today
            if updated == today:
                date_groups["Today"].append(conversation)
            elif (today - updated).days < 7:
                date_groups["Previous 7 days"].append(conversation)
            else:
                date_groups["Earlier"].append(conversation)

        for group, items in date_groups.items():
            if not items:
                continue
            st.caption(group)
            for conversation in items:
                if st.button(
                    conversation["title"][:52],
                    key=f"open_chat_{conversation['id']}",
                    type=(
                        "primary"
                        if conversation["id"] == st.session_state.current_chat_id
                        else "secondary"
                    ),
                    width="stretch",
                    help=conversation["preview"],
                ):
                    st.session_state.current_chat_id = conversation["id"]
                    st.session_state.chat_messages = (
                        st.session_state.feedback_manager.get_chat_messages(
                            conversation["id"]
                        )
                    )
                    st.session_state.chat_request_error = None
                    st.rerun()

        active_conversation = next(
            (
                conversation
                for conversation in st.session_state.feedback_manager.get_chat_conversations()
                if conversation["id"] == st.session_state.current_chat_id
            ),
            None,
        )
        if active_conversation:
            with st.expander("Conversation actions"):
                new_title = st.text_input(
                    "Rename conversation",
                    value=active_conversation["title"],
                    key=f"rename_chat_title_{active_conversation['id']}",
                )
                if st.button(
                    "Save name",
                    key=f"save_chat_title_{active_conversation['id']}",
                    width="stretch",
                ):
                    title = new_title.strip()
                    if title:
                        st.session_state.feedback_manager.rename_chat_conversation(
                            active_conversation["id"], title[:80]
                        )
                        st.rerun()
                    else:
                        st.warning("Enter a conversation name.")
                confirm_delete = st.checkbox(
                    "Confirm deleting this conversation",
                    key=f"confirm_delete_chat_{active_conversation['id']}",
                )
                if st.button(
                    "Delete conversation",
                    key=f"delete_chat_{active_conversation['id']}",
                    disabled=not confirm_delete,
                    width="stretch",
                ):
                    st.session_state.feedback_manager.delete_chat_conversation(
                        active_conversation["id"]
                    )
                    st.session_state.current_chat_id = str(uuid.uuid4())
                    st.session_state.chat_messages = []
                    st.session_state.chat_request_error = None
                    st.rerun()
        elif not conversations:
            st.caption("Your saved conversations will appear here.")

        with st.expander("Help"):
            st.write(
                "Send a message to chat with the selected Gemini model. "
                "Use the model selectors below to change models. Conversations "
                "are stored in the local generation history file."
            )

    nav_options = [
        "💬 Chat",
        "💻 Code",
        "🎨 Images",
        "👁️ Vision",
        "📊 History",
        "⚙️ Settings"
    ]

    current_idx = 0
    if st.session_state.nav_page in nav_options:
        current_idx = nav_options.index(st.session_state.nav_page)

    selected_nav = st.radio(
        "Navigation",
        nav_options,
        index=current_idx,
        label_visibility="collapsed"
    )
    st.session_state.nav_page = selected_nav

    st.markdown("---")

    # Clear Provider Status & Model Controls
    with st.expander("Provider and models", expanded=False):
        badge_class = "badge-status-online" if provider_status["is_ready"] else "badge-status-offline"
        st.markdown(
            f"**Provider:** Google Gemini <span class='{badge_class}'>{provider_status['status_label']}</span>",
            unsafe_allow_html=True
        )
        st.caption(f"Source: `{provider_status['source']}`")

        # Compact Model Selectors
        st.markdown("###### Model Selection")
        sel_text = st.selectbox(
            "Text & Code Model",
            SUPPORTED_TEXT_MODELS,
            index=SUPPORTED_TEXT_MODELS.index(st.session_state.selected_text_model)
            if st.session_state.selected_text_model in SUPPORTED_TEXT_MODELS else 0,
            key="sb_text_model_selector",
            help="Select the Gemini model for storytelling, code generation, and visual reasoning."
        )
        st.session_state.selected_text_model = sel_text

        sel_img = st.selectbox(
            "Native Image Model",
            SUPPORTED_IMAGE_MODELS,
            index=SUPPORTED_IMAGE_MODELS.index(st.session_state.selected_image_model)
            if st.session_state.selected_image_model in SUPPORTED_IMAGE_MODELS else 0,
            key="sb_image_model_selector",
            help="Select the Gemini model for text-to-image synthesis."
        )
        st.session_state.selected_image_model = sel_img

        if not provider_status["is_ready"]:
            st.info("Configure GEMINI_API_KEY in the server environment or Streamlit secrets.")
        else:
            st.caption("Credentials are loaded server-side and are never shown in the browser.")

    st.markdown("---")

    # Quick Metrics
    stats = st.session_state.feedback_manager.get_statistics()
    s_col1, s_col2 = st.columns(2)
    with s_col1:
        st.metric("Generations", stats["total_generations"])
    with s_col2:
        st.metric("Avg Rating", f"{stats['avg_rating']} ⭐" if stats["ratings_count"] > 0 else "N/A")

    request_metrics = model_manager.get_request_metrics()
    st.caption(
        "Streamed text requests this process: "
        f"{request_metrics['requests']} • Avg latency: "
        f"{request_metrics['average_latency_ms']} ms • "
        f"Reported tokens: {request_metrics['tokens']:,}"
    )

    st.markdown("---")
    st.markdown("---")
    st.caption("Local workspace · Nexa AI")


# ==============================================================================
# VIEW 1: CHAT (Storytelling, Marketing Copy & Education)
# ==============================================================================
if st.session_state.nav_page == "💬 Chat":
    chat_workspace = st.segmented_control(
        "Chat workspace",
        ["Conversation", "Story studio", "Marketing campaign", "Learning"],
        default="Conversation",
        key="chat_workspace_selector",
        width="stretch",
    )

    if chat_workspace == "Conversation":
        active_conversation = next(
            (
                conversation
                for conversation in st.session_state.feedback_manager.get_chat_conversations()
                if conversation["id"] == st.session_state.current_chat_id
            ),
            None,
        )
        title = active_conversation["title"] if active_conversation else "New chat"
        title_col, export_col = st.columns([5, 1], vertical_alignment="center")
        with title_col:
            st.markdown(f"### {title}")
            st.caption(
                f"Google Gemini · {st.session_state.selected_text_model}"
                if provider_status["is_ready"]
                else "Gemini credentials are not configured"
            )
        with export_col:
            if st.session_state.chat_messages:
                export_markdown = "\n\n".join(
                    f"## {'You' if message['role'] == 'user' else 'Nexa AI'}\n\n"
                    f"{message['content']}"
                    for message in st.session_state.chat_messages
                    if message.get("content")
                )
                st.download_button(
                    "Export",
                    data=export_markdown,
                    file_name="nexa_conversation.md",
                    mime="text/markdown",
                    width="stretch",
                )

        if not provider_status["is_ready"]:
            st.warning(
                "Chat is ready, but generation needs a server-side Gemini API key. "
                "Set `GEMINI_API_KEY` in your environment or Streamlit secrets."
            )

        if not st.session_state.chat_messages:
            st.markdown(
                """
                <section class="chat-welcome">
                    <div class="chat-welcome-mark">✦</div>
                    <h1>What would you like to create today?</h1>
                    <p>Write, code, debug, or explore an idea with Nexa AI.</p>
                </section>
                """,
                unsafe_allow_html=True,
            )
            suggestions = [
                ("Writing", "Help me write a vivid opening for a science-fiction story."),
                ("Coding", "Design a small Python utility to validate email addresses."),
                ("Debugging", "Help me diagnose a bug. Ask me for the code and error details."),
                ("Brainstorming", "Brainstorm creative names and concepts for a sustainable brand."),
            ]
            suggestion_columns = st.columns(2)
            for index, (label, suggestion) in enumerate(suggestions):
                with suggestion_columns[index % 2]:
                    if st.button(
                        f"{label}  ·  {suggestion}",
                        key=f"chat_suggestion_{index}",
                        width="stretch",
                    ):
                        st.session_state.chat_composer = suggestion
                        st.rerun()

        retry_request = None
        for index, message in enumerate(st.session_state.chat_messages):
            role = message["role"]
            with st.chat_message(
                "user" if role == "user" else "assistant",
                avatar=None if role == "user" else ":material/bolt:",
            ):
                st.markdown(message["content"])
                if role == "assistant":
                    with st.expander("Copy response"):
                        st.code(message["content"], language="markdown")
            if (
                role == "assistant"
                and index == len(st.session_state.chat_messages) - 1
                and st.button("Retry response", key=f"retry_chat_{message.get('record_id')}")
            ):
                retry_request = (
                    st.session_state.chat_messages[index - 1]["content"],
                    message.get("record_id"),
                )

        if st.session_state.chat_request_error:
            st.error(st.session_state.chat_request_error)

        if retry_request:
            run_chat_request(*retry_request)

        submitted_message = st.chat_input(
            "Message Nexa AI...",
            key="chat_composer",
            max_chars=20_000,
            disabled=st.session_state.chat_request_in_progress,
            submit_mode="disable",
        )
        if submitted_message:
            run_chat_request(submitted_message)

        st.stop()

    workspace_to_legacy_mode = {
        "Story studio": "Narrative & Storyboard",
        "Marketing campaign": "Marketing Campaign",
        "Learning": "Educational Explainer",
    }
    st.markdown("""
    <div class="nexa-header">
        <div class="nexa-title">Chat & Story Studio</div>
        <p class="nexa-subtitle">Synthesize multi-act narratives, market-ready copy, and educational deep-dives.</p>
    </div>
    """, unsafe_allow_html=True)

    if not provider_status["is_ready"]:
        st.warning(f"⚠️ {provider_status['message']}")

    chat_mode = st.radio(
        "Content Type",
        ["Narrative & Storyboard", "Marketing Campaign", "Educational Explainer"],
        index=[
            "Narrative & Storyboard",
            "Marketing Campaign",
            "Educational Explainer",
        ].index(workspace_to_legacy_mode.get(chat_workspace, "Narrative & Storyboard")),
        key="creative_content_type",
        horizontal=True,
        label_visibility="collapsed"
    )

    # --- Mode A: Multi-Chapter Narrative & Storyboard ---
    if chat_mode == "Narrative & Storyboard":
        story_prompt = st.text_area(
            "Prompt",
            value="An archaeologist uncovers an ancient quantum crystalline relic beneath the Martian ice",
            placeholder="e.g. An astronaut discovers an ancient alien library hidden inside a moon crater...",
            height=85,
            help="Provide your core story premise or scenario."
        )

        col_c1, col_c2, col_c3 = st.columns([2, 2, 1.2])
        with col_c1:
            genre_choice = st.selectbox(
                "Genre",
                ["Sci-Fi Cyberpunk", "Fantasy & Mythic", "Mystery & Thriller"],
                index=0
            )
        with col_c2:
            tone_choice = st.selectbox(
                "Tone",
                ["Thrilling & Futuristic", "Poetic & Ethereal", "Humorous & Fast-Paced", "Dark & Suspenseful"],
                index=0
            )
        with col_c3:
            num_chapters = st.selectbox("Chapters", [1, 2, 3], index=2)

        def trigger_story_generation():
            if not story_prompt.strip():
                st.warning("Please enter a story prompt to begin.")
                return
            if not provider_status["is_ready"]:
                st.error(f"Cannot generate: {provider_status['message']}")
                return

            stream_preview = st.empty()
            stream_preview.caption("Receiving story response…")
            with st.spinner(f"✨ Nexa AI is crafting your narrative using {st.session_state.selected_text_model}..."):
                try:
                    res = text_engine.generate_story(
                        api_key=gemini_api_key,
                        seed_prompt=story_prompt.strip(),
                        genre=genre_choice,
                        tone=tone_choice,
                        num_chapters=num_chapters,
                        model=st.session_state.selected_text_model,
                        on_chunk=create_stream_callback(stream_preview),
                    )
                except (APIError, RuntimeError, ValueError) as error:
                    stream_preview.empty()
                    st.error(f"Story generation failed: {sanitize_error_message(error)}")
                    return

                stream_preview.empty()
                rec_id = st.session_state.feedback_manager.add_generation(
                    modality="story",
                    prompt=story_prompt.strip(),
                    output=res["full_text"],
                    model_name=res["model_used"],
                    parameters={"genre": genre_choice, "tone": tone_choice, "chapters": num_chapters, "model": res["model_used"]}
                )
                st.session_state.active_outputs["story"] = res
                st.session_state.active_outputs["story_record_id"] = rec_id

        # Actions row
        act_col1, act_col2, act_col3, _ = st.columns([1.5, 1.2, 1, 3])
        with act_col1:
            if st.button("🚀 Generate Story", type="primary", width="stretch"):
                trigger_story_generation()
        with act_col2:
            if st.button("🔄 Retry", width="stretch"):
                trigger_story_generation()
        with act_col3:
            if st.button("🗑️ Reset", width="stretch"):
                st.session_state.active_outputs["story"] = None
                st.session_state.active_outputs["story_record_id"] = None
                st.rerun()

        # Render Story Output
        if st.session_state.active_outputs["story"]:
            story_data = st.session_state.active_outputs["story"]
            s_rec_id = st.session_state.active_outputs["story_record_id"]

            st.markdown("---")
            st.markdown(f"### {story_data['title']}")

            badge_html = f"""
            <div style="margin-bottom: 14px;">
                <span class="badge-model">{story_data['model_used']}</span>
                <span class="badge-speed">⏱️ {story_data['generation_time_ms']} ms</span>
                <span class="badge-modality">📚 {len(story_data['chapters'])} Acts</span>
            </div>
            """
            st.markdown(badge_html, unsafe_allow_html=True)

            ch_tabs = st.tabs([f"Act {c['chapter_index']}: {c['title']}" for c in story_data["chapters"]])
            for idx, tab_c in enumerate(ch_tabs):
                with tab_c:
                    ch = story_data["chapters"][idx]
                    st.write(ch["text"])
                    st.info(f"🎨 **Scene Visual Prompt:** `{ch['scene_prompt']}`")

                    if st.button(f"🎨 Render Illustration for Chapter {ch['chapter_index']}", key=f"render_ch_{idx}"):
                        if not provider_status["is_ready"]:
                            st.error(f"Cannot generate: {provider_status['message']}")
                        else:
                            with st.spinner(f"Synthesizing scene artwork with {st.session_state.selected_image_model}..."):
                                try:
                                    generated_img, meta = image_engine.generate_image(
                                        api_key=gemini_api_key,
                                        prompt=ch["scene_prompt"],
                                        style_preset="Photorealistic" if genre_choice != "Fantasy & Mythic" else "Watercolor Fantasy",
                                        aspect_ratio_key="16:9 Landscape",
                                        model=st.session_state.selected_image_model,
                                    )
                                except (APIError, RuntimeError, ValueError, OSError) as error:
                                    st.error(f"Image generation failed: {sanitize_error_message(error)}")
                                    continue

                                img_rec_id = st.session_state.feedback_manager.add_generation(
                                    modality="image",
                                    prompt=ch["scene_prompt"],
                                    output=f"Image generated with {meta['source_model']} ({meta['width']}x{meta['height']})",
                                    model_name=meta["source_model"],
                                    parameters=meta
                                )
                                st.session_state.active_outputs["image"] = generated_img
                                st.session_state.active_outputs["image_meta"] = meta
                                st.session_state.active_outputs["image_record_id"] = img_rec_id
                                st.session_state.nav_page = "🎨 Images"
                                st.rerun()

            # Actions: Download
            down_col, _ = st.columns([2, 3])
            with down_col:
                st.download_button(
                    "⬇️ Download Story (.md)",
                    data=story_data["full_text"],
                    file_name=f"nexa_story_{int(time.time())}.md",
                    mime="text/markdown",
                    width="stretch"
                )

            render_feedback_widget(
                modality="story",
                record_id=s_rec_id,
                prompt_text=story_prompt,
                on_regenerate_callback=trigger_story_generation
            )

    # --- Mode B: Complete Marketing Launch Campaign ---
    elif chat_mode == "Marketing Campaign":
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            p_name = st.text_input("Product / Service Name", value="AuraSound X1 Pro ANC Earbuds", placeholder="e.g. CloudScale AI Platform")
            p_audience = st.text_input("Target Audience", value="Audiophiles, Remote Tech Professionals, Commuters", placeholder="e.g. Startup founders, DevOps engineers")
        with m_col2:
            p_benefits = st.text_input("Core Value Proposition", value="45dB Adaptive Active Noise Cancellation, 40-hour Battery, Lossless Spatial Audio", placeholder="e.g. Zero-config deployment, 99.99% uptime")
            p_tone = st.selectbox("Tone", ["High-Energy & Persuasive", "Minimalist & Elegant", "Punchy & Direct"])

        def trigger_marketing_generation():
            if not p_name.strip():
                st.warning("Please enter a product name.")
                return
            if not provider_status["is_ready"]:
                st.error(f"Cannot generate: {provider_status['message']}")
                return

            stream_preview = st.empty()
            stream_preview.caption("Receiving campaign response…")
            with st.spinner(f"✨ Nexa AI is synthesizing your marketing kit with {st.session_state.selected_text_model}..."):
                try:
                    m_res = text_engine.generate_marketing_campaign(
                        api_key=gemini_api_key,
                        product_name=p_name,
                        target_audience=p_audience,
                        key_benefits=p_benefits,
                        tone=p_tone,
                        model=st.session_state.selected_text_model,
                        on_chunk=create_stream_callback(stream_preview),
                    )
                except (APIError, RuntimeError, ValueError) as error:
                    stream_preview.empty()
                    st.error(f"Marketing generation failed: {sanitize_error_message(error)}")
                    return

                stream_preview.empty()
                m_rec_id = st.session_state.feedback_manager.add_generation(
                    modality="marketing",
                    prompt=f"{p_name} for {p_audience}",
                    output=m_res["content"],
                    model_name=m_res["model_used"]
                )
                st.session_state.active_outputs["story"] = m_res
                st.session_state.active_outputs["story_record_id"] = m_rec_id

        act_m1, act_m2, act_m3, _ = st.columns([1.5, 1.2, 1, 3])
        with act_m1:
            if st.button("🚀 Generate Campaign", type="primary", width="stretch"):
                trigger_marketing_generation()
        with act_m2:
            if st.button("🔄 Retry", key="retry_mkt", width="stretch"):
                trigger_marketing_generation()
        with act_m3:
            if st.button("🗑️ Reset", key="reset_mkt", width="stretch"):
                st.session_state.active_outputs["story"] = None
                st.session_state.active_outputs["story_record_id"] = None
                st.rerun()

        if st.session_state.active_outputs["story"] and "product_name" in st.session_state.active_outputs["story"]:
            m_res = st.session_state.active_outputs["story"]
            st.markdown("---")
            st.markdown(m_res["content"])
            st.info(f"🖼️ **Suggested Ad Banner Prompt:** `{m_res['banner_prompt']}`")

            d_col, _ = st.columns([2, 3])
            with d_col:
                st.download_button(
                    "⬇️ Download Marketing Kit (.txt)",
                    data=m_res["content"],
                    file_name=f"nexa_marketing_{int(time.time())}.txt",
                    mime="text/plain",
                    width="stretch"
                )

            render_feedback_widget(
                modality="marketing",
                record_id=st.session_state.active_outputs["story_record_id"],
                prompt_text=m_res["product_name"]
            )

    # --- Mode C: Educational Explainer ---
    else:
        e_col1, e_col2 = st.columns([3, 1.5])
        with e_col1:
            edu_topic = st.text_input("Educational Topic", value="Transformer Architecture & Multi-Head Self-Attention", placeholder="e.g. Quantum Computing, Raft Consensus...")
        with e_col2:
            edu_level = st.selectbox("Target Level", ["High School Student", "Undergraduate CS Student", "Senior ML Engineer"])

        def trigger_edu_generation():
            if not edu_topic.strip():
                st.warning("Please enter a topic.")
                return
            if not provider_status["is_ready"]:
                st.error(f"Cannot generate: {provider_status['message']}")
                return

            stream_preview = st.empty()
            stream_preview.caption("Receiving explainer response…")
            with st.spinner(f"✨ Structuring knowledge masterclass with {st.session_state.selected_text_model}..."):
                try:
                    e_res = text_engine.generate_educational_explainer(
                        api_key=gemini_api_key,
                        topic=edu_topic,
                        level=edu_level,
                        model=st.session_state.selected_text_model,
                        on_chunk=create_stream_callback(stream_preview),
                    )
                except (APIError, RuntimeError, ValueError) as error:
                    stream_preview.empty()
                    st.error(f"Education generation failed: {sanitize_error_message(error)}")
                    return

                stream_preview.empty()
                e_rec_id = st.session_state.feedback_manager.add_generation(
                    modality="education",
                    prompt=edu_topic,
                    output=e_res["content"],
                    model_name=e_res["model_used"]
                )
                st.session_state.active_outputs["story"] = e_res
                st.session_state.active_outputs["story_record_id"] = e_rec_id

        act_e1, act_e2, act_e3, _ = st.columns([1.5, 1.2, 1, 3])
        with act_e1:
            if st.button("🎓 Generate Explainer", type="primary", width="stretch"):
                trigger_edu_generation()
        with act_e2:
            if st.button("🔄 Retry", key="retry_edu", width="stretch"):
                trigger_edu_generation()
        with act_e3:
            if st.button("🗑️ Reset", key="reset_edu", width="stretch"):
                st.session_state.active_outputs["story"] = None
                st.session_state.active_outputs["story_record_id"] = None
                st.rerun()

        if st.session_state.active_outputs["story"] and "topic" in st.session_state.active_outputs["story"]:
            e_res = st.session_state.active_outputs["story"]
            st.markdown("---")
            st.markdown(e_res["content"])

            d_col, _ = st.columns([2, 3])
            with d_col:
                st.download_button(
                    "⬇️ Download Explainer (.md)",
                    data=e_res["content"],
                    file_name=f"nexa_explainer_{int(time.time())}.md",
                    mime="text/markdown",
                    width="stretch"
                )

            render_feedback_widget(
                modality="education",
                record_id=st.session_state.active_outputs["story_record_id"],
                prompt_text=e_res["topic"]
            )


# ==============================================================================
# VIEW 2: CODE (Generation, Explanation, Debugging & Refactoring)
# ==============================================================================
elif st.session_state.nav_page == "💻 Code":
    st.markdown("""
    <div class="nexa-header">
        <div class="nexa-title">Code Studio</div>
        <p class="nexa-subtitle">Generate, explain, debug, and refactor code in your chosen language.</p>
    </div>
    """, unsafe_allow_html=True)

    if not provider_status["is_ready"]:
        st.warning(f"⚠️ {provider_status['message']}")

    code_operation = st.radio(
        "Code task",
        ["Generate", "Explain", "Debug", "Refactor"],
        horizontal=True,
        key="code_operation_selector",
    )

    code_lang = st.selectbox(
        "Language",
        ["Python", "JavaScript", "TypeScript", "SQL", "HTML/CSS", "C++"],
        key="code_language_selector",
    )
    code_prompt = ""
    source_code = ""
    operation_notes = ""
    if code_operation == "Generate":
        code_prompt = st.text_area(
            "Describe the code you need",
            value="Implement a Token Bucket Rate Limiter with burst capacity and dynamic refill",
            placeholder="Describe the behavior, constraints, inputs, and expected outputs...",
            height=100,
            key="code_generation_prompt",
            help="Describe the feature in natural language; the selected language is used for the implementation.",
        )
    else:
        source_code = st.text_area(
            "Code to review",
            placeholder="Paste the code you want to explain, debug, or refactor...",
            height=220,
            key="code_source_input",
        )
        if code_operation in ("Debug", "Refactor"):
            note_label = "Observed bug or error (optional)" if code_operation == "Debug" else "Refactoring goals (optional)"
            operation_notes = st.text_area(
                note_label,
                height=70,
                key="code_operation_notes",
            )

    c_col1, c_col2 = st.columns([1.5, 1.5])
    with c_col1:
        code_level = st.selectbox(
            "Implementation style",
            ["Production-Ready", "Minimalist Clean", "Enterprise Resilient"],
            key="code_implementation_style",
            disabled=code_operation != "Generate",
        )
    with c_col2:
        include_mermaid = st.checkbox(
            "Include architecture diagram",
            value=True,
            key="code_include_mermaid",
            disabled=code_operation != "Generate",
        )

    def trigger_code_generation():
        if st.session_state.code_request_in_progress:
            st.session_state.code_request_error = "A code request is already in progress. Please wait for it to finish."
            return
        request_input = code_prompt.strip() if code_operation == "Generate" else source_code.strip()
        if not request_input:
            st.session_state.code_request_error = (
                "Describe the code you need."
                if code_operation == "Generate"
                else "Paste code to continue."
            )
            return
        if not provider_status["is_ready"]:
            st.session_state.code_request_error = f"Cannot continue: {provider_status['message']}"
            return

        st.session_state.code_request_in_progress = True
        st.session_state.code_request_error = None
        stream_preview = st.empty()
        stream_preview.caption("Receiving code response…")
        try:
            with st.spinner(f"{loading_label} with {st.session_state.selected_text_model}..."):
                if code_operation == "Generate":
                    c_res = code_engine.generate_code(
                        api_key=gemini_api_key,
                        prompt=request_input,
                        language=code_lang,
                        complexity=code_level,
                        include_mermaid=include_mermaid,
                        model=st.session_state.selected_text_model,
                        on_chunk=create_stream_callback(stream_preview),
                    )
                elif code_operation == "Explain":
                    c_res = code_engine.explain_code(
                        api_key=gemini_api_key,
                        code=request_input,
                        language=code_lang,
                        model=st.session_state.selected_text_model,
                        on_chunk=create_stream_callback(stream_preview),
                    )
                elif code_operation == "Debug":
                    c_res = code_engine.debug_code(
                        api_key=gemini_api_key,
                        code=request_input,
                        language=code_lang,
                        issue=operation_notes,
                        model=st.session_state.selected_text_model,
                        on_chunk=create_stream_callback(stream_preview),
                    )
                else:
                    c_res = code_engine.refactor_code(
                        api_key=gemini_api_key,
                        code=request_input,
                        language=code_lang,
                        goals=operation_notes,
                        model=st.session_state.selected_text_model,
                        on_chunk=create_stream_callback(stream_preview),
                    )

                stream_preview.empty()
                c_rec_id = st.session_state.feedback_manager.add_generation(
                    modality="code",
                    prompt=request_input,
                    output=c_res["code"],
                    model_name=c_res["model_used"],
                    parameters={
                        "operation": code_operation,
                        "language": code_lang,
                        "style": code_level if code_operation == "Generate" else None,
                        "model": c_res["model_used"],
                    },
                )
                st.session_state.active_outputs["code"] = c_res
                st.session_state.active_outputs["code_record_id"] = c_rec_id
        except (APIError, RuntimeError, TimeoutError, ValueError) as error:
            stream_preview.empty()
            st.session_state.code_request_error = (
                f"{code_operation} failed: {sanitize_error_message(error)}"
            )
        finally:
            st.session_state.code_request_in_progress = False

    action_label = {
        "Generate": "Generate code",
        "Explain": "Explain code",
        "Debug": "Debug code",
        "Refactor": "Refactor code",
    }[code_operation]
    loading_label = {
        "Generate": "Generating code",
        "Explain": "Explaining code",
        "Debug": "Debugging code",
        "Refactor": "Refactoring code",
    }[code_operation]
    act_c1, act_c2, act_c3, _ = st.columns([1.5, 1.2, 1, 3])
    with act_c1:
        if st.button(
            action_label,
            type="primary",
            width="stretch",
            disabled=st.session_state.code_request_in_progress,
        ):
            trigger_code_generation()
    with act_c2:
        if st.button(
            "Retry",
            key="retry_code",
            width="stretch",
            disabled=st.session_state.code_request_in_progress,
        ):
            trigger_code_generation()
    with act_c3:
        if st.button("Reset", key="reset_code", width="stretch"):
            st.session_state.active_outputs["code"] = None
            st.session_state.active_outputs["code_record_id"] = None
            st.session_state.code_request_error = None
            st.rerun()

    if st.session_state.code_request_error:
        st.error(st.session_state.code_request_error)

    if st.session_state.active_outputs["code"]:
        c_res = st.session_state.active_outputs["code"]
        c_rec_id = st.session_state.active_outputs["code_record_id"]

        st.markdown("---")
        badge_html = f"""
        <div style="margin-bottom: 12px;">
            <span class="badge-model">{c_res['model_used']}</span>
            <span class="badge-speed">⏱️ {c_res['generation_time_ms']} ms</span>
            <span class="badge-modality">💻 {c_res['language']}</span>
        </div>
        """
        st.markdown(badge_html, unsafe_allow_html=True)

        highlight_language = {"HTML/CSS": "html", "C++": "cpp"}.get(
            c_res["language"], c_res["language"].lower()
        )
        if c_res["code"]:
            st.code(c_res["code"], language=highlight_language)

            ext_map = {
                "Python": "py",
                "JavaScript": "js",
                "TypeScript": "ts",
                "SQL": "sql",
                "HTML/CSS": "html",
                "C++": "cpp",
            }
            file_ext = ext_map.get(c_res["language"], "txt")
            st.download_button(
                label=f"Download {c_res['language']} source",
                data=c_res["code"],
                file_name=f"nexa_solution_{int(time.time())}.{file_ext}",
                mime="text/plain",
                width="content",
            )

        if c_res.get("mermaid_chart"):
            st.markdown("##### Architecture and data flow")
            st.markdown(f"```mermaid\n{c_res['mermaid_chart']}\n```")

        if c_res.get("explanation"):
            st.markdown("##### Explanation")
            st.markdown(c_res["explanation"])
        if c_res.get("complexity"):
            st.caption(c_res["complexity"])

        render_feedback_widget(
            modality="code",
            record_id=c_rec_id,
            prompt_text=c_res["prompt"],
            on_regenerate_callback=trigger_code_generation,
        )


# ==============================================================================
# VIEW 3: IMAGES (Visual Studio Text-to-Image with Gemini)
# ==============================================================================
elif st.session_state.nav_page == "🎨 Images":
    st.markdown("""
    <div class="nexa-header">
        <div class="nexa-title">Visual Generation Studio</div>
        <p class="nexa-subtitle">Create high-resolution artwork and graphics with Gemini's native image model.</p>
    </div>
    """, unsafe_allow_html=True)

    if not provider_status["is_ready"]:
        st.warning(f"⚠️ {provider_status['message']}")

    img_prompt = st.text_area(
        "Prompt",
        value="A futuristic neon city with flying vehicles, rainy reflective streets, and glowing holograms",
        placeholder="e.g. A serene Japanese zen garden with cherry blossoms, moss stones, and morning mist...",
        height=85,
        help="Describe what you want to see, including subjects, environment, and mood."
    )

    col_i1, col_i2 = st.columns(2)
    with col_i1:
        style_choice = st.selectbox("Style Preset", list(STYLE_PRESETS.keys()), index=1)
    with col_i2:
        aspect_choice = st.selectbox("Dimensions / Aspect Ratio", list(ASPECT_RATIOS.keys()), index=0)

    with st.expander("Advanced Settings (Negative Prompt)", expanded=False):
        img_negative = st.text_input(
            "Negative Prompt (elements to suppress)",
            value="blurry, low resolution, distorted limbs, oversaturated, watermark"
        )

    def trigger_image_generation():
        if not img_prompt.strip():
            st.warning("Please enter a visual prompt to begin.")
            return
        if not provider_status["is_ready"]:
            st.error(f"Cannot generate: {provider_status['message']}")
            return

        with st.spinner(f"🎨 Nexa AI is synthesizing visual artwork with {st.session_state.selected_image_model}..."):
            try:
                generated_img, meta = image_engine.generate_image(
                    api_key=gemini_api_key,
                    prompt=img_prompt.strip(),
                    negative_prompt=img_negative,
                    style_preset=style_choice,
                    aspect_ratio_key=aspect_choice,
                    model=st.session_state.selected_image_model,
                )
            except (APIError, RuntimeError, ValueError, OSError) as error:
                st.error(f"Image generation failed: {sanitize_error_message(error)}")
                return

            rec_id = st.session_state.feedback_manager.add_generation(
                modality="image",
                prompt=img_prompt.strip(),
                output=f"Image generated with {meta['source_model']} ({meta['width']}x{meta['height']})",
                model_name=meta["source_model"],
                parameters=meta
            )
            st.session_state.active_outputs["image"] = generated_img
            st.session_state.active_outputs["image_meta"] = meta
            st.session_state.active_outputs["image_record_id"] = rec_id

    act_i1, act_i2, act_i3, _ = st.columns([1.5, 1.2, 1, 3])
    with act_i1:
        if st.button("🎨 Generate Image", type="primary", width="stretch"):
            trigger_image_generation()
    with act_i2:
        if st.button("🔄 Retry", key="retry_img", width="stretch"):
            trigger_image_generation()
    with act_i3:
        if st.button("🗑️ Reset", key="reset_img", width="stretch"):
            st.session_state.active_outputs["image"] = None
            st.session_state.active_outputs["image_meta"] = None
            st.session_state.active_outputs["image_record_id"] = None
            st.rerun()

    if st.session_state.active_outputs["image"]:
        cur_img = st.session_state.active_outputs["image"]
        cur_meta = st.session_state.active_outputs["image_meta"]
        rec_id = st.session_state.active_outputs["image_record_id"]

        st.markdown("---")
        badge_html = f"""
        <div style="margin-bottom: 12px;">
            <span class="badge-model">{cur_meta['source_model']}</span>
            <span class="badge-speed">⏱️ {cur_meta['generation_time_ms']} ms</span>
            <span class="badge-modality">📐 {cur_meta['width']}x{cur_meta['height']} ({cur_meta['aspect_ratio']})</span>
        </div>
        """
        st.markdown(badge_html, unsafe_allow_html=True)

        col_img_l, col_img_r = st.columns([3, 1.8])
        with col_img_l:
            st.image(cur_img, width="stretch", caption=f"Prompt: {cur_meta['prompt']}", alt="Generated Artwork")
        with col_img_r:
            st.markdown("##### 📥 Export Artwork")
            buf = io.BytesIO()
            cur_img.save(buf, format="PNG")
            st.download_button(
                label="⬇️ Download Full-Res PNG",
                data=buf.getvalue(),
                file_name=f"nexa_art_{rec_id}.png",
                mime="image/png",
                width="stretch"
            )
            st.markdown("##### 🔍 Generation Provenance")
            st.json({
                "Model": cur_meta["source_model"],
                "Style": cur_meta["style_preset"],
                "Resolution": f"{cur_meta['width']}x{cur_meta['height']}",
                "Aspect Ratio": cur_meta["aspect_ratio"],
                "Generation Time": f"{cur_meta['generation_time_ms']} ms"
            })

        render_feedback_widget(
            modality="image",
            record_id=rec_id,
            prompt_text=cur_meta["prompt"],
            on_regenerate_callback=trigger_image_generation
        )


# ==============================================================================
# VIEW 4: VISION (Visual Reasoning & Image-to-Text)
# ==============================================================================
elif st.session_state.nav_page == "👁️ Vision":
    st.markdown("""
    <div class="nexa-header">
        <div class="nexa-title">Visual Reasoning & Vision</div>
        <p class="nexa-subtitle">Analyze visual inputs with Gemini to extract attributes, generate copy, deconstruct architectures, or start stories.</p>
    </div>
    """, unsafe_allow_html=True)

    if not provider_status["is_ready"]:
        st.warning(f"⚠️ {provider_status['message']}")

    v_col1, v_col2 = st.columns([1.6, 2])

    with v_col1:
        st.markdown("##### 1. Select Visual Seed")
        image_source = st.radio("Input Source", ["Pre-loaded Sample", "Upload Image"], horizontal=True, label_visibility="collapsed")

        selected_image = None
        if image_source == "Pre-loaded Sample":
            samples = {
                "🎧 AuraSound Product Mockup": "assets/sample_images/sample_product.png",
                "🌄 Emerald Ridge Sunset Nature": "assets/sample_images/sample_nature.png",
                "🏗️ Microservice Architecture Diagram": "assets/sample_images/sample_diagram.png"
            }
            sample_choice = st.selectbox("Sample Image", list(samples.keys()))
            sample_path = os.path.join(os.path.dirname(__file__), samples[sample_choice])
            if os.path.exists(sample_path):
                selected_image = Image.open(sample_path)
        else:
            uploaded_file = st.file_uploader("Upload Image (PNG, JPG, WEBP)", type=["png", "jpg", "jpeg", "webp"])
            if uploaded_file is not None:
                selected_image = Image.open(uploaded_file)

        if selected_image:
            st.image(selected_image, caption="Active Visual Input", width="stretch", alt="Selected Input")

    with v_col2:
        st.markdown("##### 2. Directives & Task")
        vision_task = st.selectbox(
            "Task Mode",
            [
                "Marketing Copy & Social Media",
                "Technical Diagram & Architecture Deconstruction",
                "Visual Story Starter",
                "Accessibility & Detailed Visual Audit"
            ]
        )
        custom_vision_notes = st.text_input(
            "Instructions (optional)",
            placeholder="e.g. Focus on tech founders; highlight portability and audio clarity"
        )

        def trigger_vision_generation():
            if selected_image is None:
                st.warning("Please select or upload an image first.")
                return
            if not provider_status["is_ready"]:
                st.error(f"Cannot generate: {provider_status['message']}")
                return

            with st.spinner(f"👁️ Analyzing visual features with {st.session_state.selected_text_model}..."):
                try:
                    v_res = vision_engine.analyze_image_and_generate(
                        api_key=gemini_api_key,
                        image=selected_image,
                        task_mode=vision_task,
                        custom_instructions=custom_vision_notes,
                        model=st.session_state.selected_text_model,
                    )
                except (APIError, RuntimeError, ValueError, OSError) as error:
                    st.error(f"Vision analysis failed: {sanitize_error_message(error)}")
                    return

                v_rec_id = st.session_state.feedback_manager.add_generation(
                    modality="vision",
                    prompt=f"{vision_task} on image ({v_res['features']['dimensions']})",
                    output=v_res["analysis"],
                    model_name=v_res["model_used"],
                    parameters={"features": v_res["features"], "model": v_res["model_used"]}
                )
                st.session_state.active_outputs["vision"] = v_res
                st.session_state.active_outputs["vision_record_id"] = v_rec_id

        act_v1, act_v2, act_v3 = st.columns([1.5, 1.2, 1])
        with act_v1:
            if st.button("🔮 Analyze Visual", type="primary", width="stretch"):
                trigger_vision_generation()
        with act_v2:
            if st.button("🔄 Retry", key="retry_vis", width="stretch"):
                trigger_vision_generation()
        with act_v3:
            if st.button("🗑️ Reset", key="reset_vis", width="stretch"):
                st.session_state.active_outputs["vision"] = None
                st.session_state.active_outputs["vision_record_id"] = None
                st.rerun()

    if st.session_state.active_outputs["vision"]:
        v_res = st.session_state.active_outputs["vision"]
        v_rec_id = st.session_state.active_outputs["vision_record_id"]

        st.markdown("---")
        badge_html = f"""
        <div style="margin-bottom: 12px;">
            <span class="badge-model">{v_res['model_used']}</span>
            <span class="badge-speed">⏱️ {v_res['generation_time_ms']} ms</span>
            <span class="badge-modality">👁️ {v_res['task_mode']}</span>
        </div>
        """
        st.markdown(badge_html, unsafe_allow_html=True)

        with st.expander("📊 Extracted Perceptual Features", expanded=False):
            st.json(v_res["features"])

        st.markdown(v_res["analysis"])

        down_col, _ = st.columns([2, 3])
        with down_col:
            st.download_button(
                "⬇️ Download Analysis (.md)",
                data=v_res["analysis"],
                file_name=f"nexa_vision_{int(time.time())}.md",
                mime="text/markdown",
                width="stretch"
            )

        render_feedback_widget(
            modality="vision",
            record_id=v_rec_id,
            prompt_text=f"Visual Analysis: {vision_task}",
            on_regenerate_callback=trigger_vision_generation
        )


# ==============================================================================
# VIEW 5: HISTORY (Feedback & Generation Log)
# ==============================================================================
elif st.session_state.nav_page == "📊 History":
    st.markdown("""
    <div class="nexa-header">
        <div class="nexa-title">Generation History & Feedback</div>
        <p class="nexa-subtitle">Review generation provenance, inspect user ratings, and export records for auditing.</p>
    </div>
    """, unsafe_allow_html=True)

    stats = st.session_state.feedback_manager.get_statistics()

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Total Outputs", stats["total_generations"])
    with m_col2:
        st.metric("Average Rating", f"{stats['avg_rating']} ⭐" if stats["ratings_count"] > 0 else "N/A")
    with m_col3:
        st.metric("Helpful 👍", stats["thumbs_up"])
    with m_col4:
        st.metric("Adjustments 👎", stats["thumbs_down"])

    st.markdown("---")

    filter_col, export_col, clear_col = st.columns([2, 1.5, 1])
    with filter_col:
        filter_modality = st.selectbox(
            "Filter by Modality",
            ["All Modalities", "image", "story", "marketing", "code", "education", "vision"],
            label_visibility="collapsed"
        )
    with export_col:
        history_data = st.session_state.feedback_manager.get_history(
            modality=None if filter_modality == "All Modalities" else filter_modality
        )
        json_export = json.dumps(history_data, indent=2)
        st.download_button(
            "📥 Export History (JSON)",
            data=json_export,
            file_name=f"nexa_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            width="stretch"
        )
    with clear_col:
        if st.button("🗑️ Clear Log", width="stretch"):
            st.session_state.feedback_manager.clear_history()
            st.toast("History log cleared.", icon="🧹")
            st.rerun()

    if not history_data:
        st.info("No generation records found. Synthesize text, code, or images to populate this log.")
    else:
        for item in history_data:
            with st.expander(f"[{item['modality'].upper()}] {item['prompt'][:60]}... ({item['timestamp']})", expanded=False):
                st.write(f"**ID:** `{item['id']}` | **Model:** `{item['model_name']}`")
                st.write(f"**Rating:** {item.get('rating') or 'Unrated'} ⭐ | **Feedback:** {item.get('feedback_type') or 'None'}")
                st.write(f"**Prompt:** {item['prompt']}")
                st.markdown("**Output Preview:**")
                st.text(str(item["output"])[:350] + ("..." if len(str(item["output"])) > 350 else ""))
                if item.get("parameters"):
                    st.caption(f"Parameters: {item['parameters']}")


# ==============================================================================
# VIEW 6: SETTINGS (Provider Configuration & Model Hub)
# ==============================================================================
elif st.session_state.nav_page == "⚙️ Settings":
    st.markdown("""
    <div class="nexa-header">
        <div class="nexa-title">Settings & Provider Configuration</div>
        <p class="nexa-subtitle">Check provider readiness, configure server-side credentials, and select models.</p>
    </div>
    """, unsafe_allow_html=True)

    # Section 1: Provider Status & Security Architecture
    st.markdown("#### 📡 Provider & Credential Status")
    status_card_class = "badge-status-online" if provider_status["is_ready"] else "badge-status-offline"

    stat_col1, stat_col2, stat_col3 = st.columns(3)
    with stat_col1:
        st.markdown(f"**Provider**<br>Google Gemini API", unsafe_allow_html=True)
    with stat_col2:
        st.markdown(f"**Status**<br><span class='{status_card_class}'>{provider_status['status_label']}</span>", unsafe_allow_html=True)
    with stat_col3:
        st.markdown(f"**Credential Source**<br>`{provider_status['source']}`", unsafe_allow_html=True)

    if not provider_status["is_ready"]:
        st.warning(f"⚠️ {provider_status['message']}")
    else:
        st.success(f"✅ {provider_status['message']}")

    st.markdown("---")

    # Section 2: Server-side credential configuration
    with st.expander("Server-side credential configuration", expanded=not provider_status["is_ready"]):
        st.markdown("""
        Set `GEMINI_API_KEY` in the server environment or in `.streamlit/secrets.toml`.
        For local development, copy `.env.example` to `.env` and replace its placeholder.
        Credentials are not accepted in the browser or stored in session state.
        """)
        st.caption("A configured key is not validated with a live request until you start a generation.")

    st.markdown("---")

    # Section 3: Model Selection & Registry
    st.markdown("#### 🎛️ Model Selection Controls")
    m_sel_col1, m_sel_col2 = st.columns(2)

    with m_sel_col1:
        st.markdown("##### 📝 Text, Code & Vision Model")
        new_text_model = st.selectbox(
            "Active Text Model",
            SUPPORTED_TEXT_MODELS,
            index=SUPPORTED_TEXT_MODELS.index(st.session_state.selected_text_model)
            if st.session_state.selected_text_model in SUPPORTED_TEXT_MODELS else 0,
            key="settings_text_model_selector",
            help="Default is gemini-3.8-flash. Configurable via GEMINI_TEXT_MODEL environment variable."
        )
        st.session_state.selected_text_model = new_text_model
        st.caption(f"Currently active: `{st.session_state.selected_text_model}`")

    with m_sel_col2:
        st.markdown("##### 🎨 Native Image Model")
        new_img_model = st.selectbox(
            "Active Image Model",
            SUPPORTED_IMAGE_MODELS,
            index=SUPPORTED_IMAGE_MODELS.index(st.session_state.selected_image_model)
            if st.session_state.selected_image_model in SUPPORTED_IMAGE_MODELS else 0,
            key="settings_image_model_selector",
            help="Default is gemini-nano-banana-2.1. Configurable via GEMINI_IMAGE_MODEL environment variable."
        )
        st.session_state.selected_image_model = new_img_model
        st.caption(f"Currently active: `{st.session_state.selected_image_model}`")

    st.markdown("---")

    # Section 4: Architecture & Security Overview
    st.markdown("#### 🏛️ Security & Architecture")
    st.markdown("""
    * **Zero Secret Leakage:** Keys are stored server-side and never serialized into history logs, browser DOM, or error traces.
    * **No browser credential entry:** Provider status shows the credential source without displaying any part of the key.
    * **Git Safety:** Both `.env` and `.streamlit/secrets.toml` are strictly ignored in `.gitignore`, while `.env.example` serves as a clean template.
    """)
