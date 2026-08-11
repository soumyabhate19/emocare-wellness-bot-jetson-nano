#!/usr/bin/env python
# coding: utf-8

from cProfile import label
import streamlit as st
import os
import re
import time
import textwrap
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from groq import Groq
import PyPDF2
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import numpy as np

try:
    from elevenlabs.client import ElevenLabs
except ImportError:
    ElevenLabs = None


# ------------------ Mood -> Music Recommendations ------------------
MOOD_MUSIC = {
    "Neutral": {
        "title": "Chill Focus Mix",
        "url": "https://www.youtube.com/results?search_query=chill+focus+music",
        "reason": "Light background music that helps you settle without changing your mood too much."
    },
    "Happy / Excited": {
        "title": "Upbeat Feel-Good Pop",
        "url": "https://www.youtube.com/results?search_query=feel+good+pop+playlist",
        "reason": "Keeps the energy high and positive."
    },
    "Calm / Okay": {
        "title": "Lo-fi Chill",
        "url": "https://www.youtube.com/results?search_query=lofi+chill+beats",
        "reason": "Supports calm focus and a steady vibe."
    },
    "Stressed / Overwhelmed": {
        "title": "Relaxing Ambient / Breathable Beats",
        "url": "https://www.youtube.com/results?search_query=relaxing+ambient+music+stress+relief",
        "reason": "Slow, soft textures can help your nervous system downshift."
    },
    "Sad / Low": {
        "title": "Soft Comfort Songs",
        "url": "https://www.youtube.com/results?search_query=comfort+music+playlist",
        "reason": "Gentle songs can feel supportive without forcing cheerfulness."
    },
    "Angry / Frustrated": {
        "title": "Release & Reset (Workout / Rock)",
        "url": "https://www.youtube.com/results?search_query=workout+rock+playlist",
        "reason": "Helps release tension and channel energy safely."
    },
    "Lonely / Disconnected": {
        "title": "Warm Indie / Soft R&B",
        "url": "https://www.youtube.com/results?search_query=warm+indie+playlist",
        "reason": "Cozy vocals can feel like company."
    },
}


def render_music_recommendation(mood: str):
    """Render mood-based music recommendation in a Streamlit-version-safe way."""
    rec = MOOD_MUSIC.get(mood)

    st.markdown("### 🎵 Music recommendation for your mood")

    if not rec:
        st.info("No music recommendation available for this mood yet.")
        return

    st.write(f"**{rec['title']}**")
    st.caption(rec["reason"])

    if hasattr(st, "link_button"):
        try:
            st.link_button("Open playlist/search", rec["url"])
        except Exception:
            st.markdown(f"👉 [Open playlist/search]({rec['url']})")
    else:
        st.markdown(f"👉 [Open playlist/search]({rec['url']})")


# ------------------ RIGHT PANEL: Emotion -> Action Compass ------------------
ACTION_COMPASS = [
    ("Angry / Frustrated", "Angry", "Sing it out – let the heat leave softly 🎵"),
    ("Stressed / Overwhelmed", "Stressed", "Move your body – even 60 seconds counts 🏃"),
    ("Lonely / Disconnected", "Lonely", "Send one message – connection starts small 💬"),
    ("Sad / Low", "Sad", "Name one tiny gratitude – a warm ember 💛"),
    ("Calm / Okay", "Calm", "Protect this calm – slow down on purpose 🌿"),
    ("Happy / Excited", "Happy", "Celebrate it – dance, share, sparkle ✨"),
    ("Neutral", "Neutral", "Check in gently – what do you need right now? 🧘"),
]

EXTRA_COMPASS_LINES = [
    ("Burned out", "Take a slow walk – restart the engine gently 🚶‍♀️"),
    ("Overthinking", "Write it down – give your mind a shelf ✏️"),
    ("Anxious", "Breathe – your body understands the way home 🌬️"),
    ("Lazy", "Cold splash / stretch – wake up the senses ❄️"),
    ("Impatient", "Reflect on progress – you're further than you feel 🧭"),
]

def render_action_compass(current_mood: str):
    st.caption("Just a gentle nudge, not a rule 🌱")

    ACTIONS = [
        ("Angry", "🎵 Sing it out – let the heat leave softly"),
        ("Stressed", "🏃 Move your body – even 60 seconds counts"),
        ("Lonely", "💬 Send one message – connection starts small"),
        ("Sad", "💛 Name one tiny gratitude – a warm ember"),
        ("Calm", "🌿 Protect this calm – slow down on purpose"),
        ("Happy", "✨ Celebrate it – dance, share, sparkle"),
        ("Neutral", "🧘 Check in gently – what do you need right now?"),
    ]

    cm = (current_mood or "").lower()

    for mood, text in ACTIONS:
        active = mood.lower() in cm

        st.markdown(
            f"""
            <div class="compass-swatch {'active' if active else ''}">
              <b>{mood}</b> → {text}
            </div>
            """,
            unsafe_allow_html=True,
        )

# ------------------ Mini Games: open standalone HTML games hosted on GitHub Pages ------------------
# These files live in the /docs/games/ folder of your GitHub repo. GitHub Pages
# serves that folder as real https:// pages, which open reliably in a new tab —
# unlike data: URIs, which some Chrome builds/extensions block or mangle.
GAMES_BASE_URL = "https://soumyabhate19.github.io/emocare-wellness-bot-jetson-nano/games"

MINI_GAMES = [
    {
        "id": "bubble_pop",
        "label": "🫧 Bubble Breather",
        "blurb": "Pop rising bubbles in time with your breath.",
        "file": "bubble_pop.html",
    },
    {
        "id": "gratitude_jar",
        "label": "🍯 Gratitude Jar",
        "blurb": "Collect tiny gratitudes in a jar.",
        "file": "gratitude_jar.html",
    },
    {
        "id": "grounding_senses",
        "label": "🌬️ Grounding: 5 Senses",
        "blurb": "Anchor into the present, one sense at a time.",
        "file": "grounding_senses.html",
    },
    {
        "id": "trace_animals",
        "label": "🖍️ Trace & Color",
        "blurb": "Trace animals, birds, flowers, and fruits in any color.",
        "file": "trace_animals.html",
    },
    {
        "id": "doodle_pad",
        "label": "🎨 Doodle Pad",
        "blurb": "Draw the prompt — a real AI guesses what you drew.",
        "file": "doodle_pad.html",
    },
]


def render_mini_games_grid():
    st.caption("Or try one of these — opens in a new tab, come back anytime:")
    rows = [MINI_GAMES[i:i + 2] for i in range(0, len(MINI_GAMES), 2)]
    for row in rows:
        cols = st.columns(len(row))
        for col, game in zip(cols, row):
            with col:
                url = f"{GAMES_BASE_URL}/{game['file']}"
                st.markdown(
                    f"""
                    <a href="{url}" target="_blank" rel="noopener noreferrer"
                       style="text-decoration:none;">
                        <div style="background:#F9E7B2;color:#4A2E10;
                                    border-radius:10px;padding:10px 12px;
                                    text-align:center;font-weight:700;
                                    border:1px solid #D9BC7D;">
                            {game['label']}
                        </div>
                    </a>
                    <div style="font-size:12px;color:#F9E7B2;text-align:center;
                                margin-top:4px;margin-bottom:14px;opacity:0.85;">
                        {game['blurb']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )




def run_calm_quest():
    # ------------------ Calm Quest (existing 60s mini-game) ------------------
    st.markdown("#### Calm Quest")
    st.caption("A tiny reset for your mind + body. You can stop anytime.")

    if st.button("🛑 End Calm Quest", use_container_width=True):
        st.session_state.calm_quest_active = False
        st.session_state.calm_quest_step = 0
        st.rerun()

    step = st.session_state.calm_quest_step

    if step == 0:
        st.markdown("### Step 1 – Breathing Timer 🌬️")
        st.session_state.calm_quest_breath_seconds = st.slider(
            "Choose breathing time (seconds)",
            10, 45, st.session_state.calm_quest_breath_seconds,
        )

        if st.button("▶️ Start Breathing", type="primary", use_container_width=True):
            secs = st.session_state.calm_quest_breath_seconds
            progress = st.progress(0)
            status = st.empty()

            for i in range(secs):
                cue = "Inhale…" if (i // 4) % 2 == 0 else "Exhale…"
                status.markdown(f"**{cue}** ({secs - i}s left)")
                progress.progress(int((i + 1) / secs * 100))
                time.sleep(1)

            status.success("✅ Nice. One small reset done.")
            st.session_state.calm_quest_step = 1
            st.rerun()

    elif step == 1:
        st.markdown("### Step 2 – Grounding (3 things you see) 👀")
        st.session_state.calm_quest_seen = st.text_input(
            "Type 3 things you can see right now (comma-separated):",
            value=st.session_state.calm_quest_seen,
            placeholder="e.g., laptop, window, water bottle",
        )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("⬅️ Back", use_container_width=True):
                st.session_state.calm_quest_step = 0
                st.rerun()
        with c2:
            if st.button("Next ➡️", type="primary", use_container_width=True):
                st.session_state.calm_quest_step = 2
                st.rerun()

    else:
        st.markdown("### Step 3 – One-line Journal ✏️")
        st.session_state.calm_quest_need = st.text_area(
            "Finish this sentence: **Right now I need…**",
            value=st.session_state.calm_quest_need,
            height=100,
            placeholder="…a break, clarity, reassurance, a plan, rest, etc.",
        )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("⬅️ Back", use_container_width=True):
                st.session_state.calm_quest_step = 1
                st.rerun()
        with c2:
            if st.button("🏆 Finish Quest", type="primary", use_container_width=True):
                recap = (
                    "CALM QUEST RECAP:\n"
                    f"- Mood: {st.session_state.current_mood}\n"
                    f"- Focus area: {st.session_state.focus_area}\n"
                    f"- 3 things I see: {st.session_state.calm_quest_seen}\n"
                    f"- Right now I need: {st.session_state.calm_quest_need}\n\n"
                    "Please respond warmly with:\n"
                    "1) One supportive sentence\n"
                    "2) One tiny next step (2 minutes)\n"
                    "3) One gentle reflective question\n"
                )

                with st.spinner("EmoCare is reflecting on your Calm Quest..."):
                    response_text, _ = get_wellness_response(
                        recap,
                        st.session_state.focus_area,
                        st.session_state.current_mood,
                        journal_text=st.session_state.uploaded_pdf_text,
                    )

                st.session_state.conversation_history.append(
                    {"role": "user", "text": "🎮 Completed Calm Quest", "timestamp": datetime.now().isoformat()}
                )
                st.session_state.conversation_history.append(
                    {
                        "role": "assistant",
                        "text": response_text,
                        "timestamp": datetime.now().isoformat(),
                        "used_pdf": bool(st.session_state.uploaded_pdf_text),
                    }
                )

                if st.session_state.use_tts and elevenlabs_client:
                    audio_bytes = elevenlabs_tts_bytes(response_text)
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/mp3")

                st.session_state.calm_quest_active = False
                st.session_state.calm_quest_step = 0
                st.session_state.calm_quest_seen = ""
                st.session_state.calm_quest_need = ""

                st.success("✅ Calm Quest complete. Check your Conversation History.")
                st.rerun()


# ---------- Joke generator ----------
def get_funny_joke(mood: str, avatar: str) -> str:
    system_prompt = f"""
You are EmoCare, a friendly wellness companion.
Generate ONE short, genuinely funny, wholesome joke (max 2 lines).
No dark humor. No insults. No politics. No religion. No self-harm references.
Keep it safe and uplifting.

Style:
- If avatar is Bunny: cute + gentle
- If avatar is Pandy: playful
- If avatar is Silly: extra goofy

User mood: {mood}
Avatar: {avatar}
"""
    joke = getTextLLM_system(system_prompt, "Tell me a joke.")
    return (joke or "").strip()


# ---------- EmoCare avatar & theme config ----------
AVATAR_OPTIONS = {
    "Bunny": "🐰",
    "Pandy": "🐼",
    "Silly": "🦭"
}

# ---------- Streamlit page config ----------
st.set_page_config(page_title="AI Wellness Companion", layout="wide", page_icon="🧠")

# ---------- Load external CSS ----------
def load_css(file_name="wellness.css"):
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file '{file_name}' not found. Using default styling.")


load_css()

# ---------- ENV & CLIENTS ----------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

elevenlabs_client = None
if ELEVENLABS_API_KEY and ElevenLabs:
    try:
        elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    except Exception as e:
        st.warning(f"ElevenLabs client initialization failed. Voice features disabled. ({e})")
        elevenlabs_client = None

# ---------- LLM helpers ----------
def getTextLLM_system(system_prompt, user_text):
    if not groq_client:
        return "LLM Error: GROQ_API_KEY is not configured."
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.7,
            max_completion_tokens=1500,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"LLM Error: Could not generate response. ({e})"

# ---------- Crisis detection & Core wellness response ----------
CRISIS_KEYWORDS = [
    "kill myself", "end my life", "suicidal", "suicide", "don't want to live",
    "want to die", "self harm", "ending it all", "can't go on", "hopeless",
]

def is_crisis_message(text: str) -> bool:
    if not text:
        return False
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in CRISIS_KEYWORDS)

def build_crisis_response():
    return (
        "I'm really glad you reached out and shared this with me. "
        "I'm only a computer program and **I can't help in an emergency**. "
        "**If you are in immediate danger, please:**\n"
        "- Call your local emergency number right now (like **911** in the U.S.).\n"
        "- If you're in the U.S., you can also call or text **988** to reach the Suicide & Crisis Lifeline.\n"
        "Your safety and wellbeing are important. 💜"
    )

def get_wellness_response(user_text, focus_area, mood, journal_text=None):
    if is_crisis_message(user_text or ""):
        return build_crisis_response(), []

    context = f"User's chosen focus area: {focus_area}\nUser's current mood: {mood}\n\n"
    if journal_text:
        context += f"The user has also shared some journal text. Snippet:\n{journal_text[:400]}\n\n"

    system_prompt = """
You are an AI mental wellness companion named EmoCare.

Your role:
- Provide gentle emotional support and reflection.
- Help users understand their feelings and suggest simple, healthy coping ideas.
- Ask 1–2 gentle reflective questions when appropriate.

Important boundaries:
- You are not a doctor, therapist, counselor, or emergency service.
- You do not form personal, romantic, or exclusive relationships with users.
- If a user expresses romantic feelings toward you (e.g., “I love you”, “be my partner”, “don’t leave me”),
  respond kindly, set a clear boundary, and encourage connection with real people (friends, family, trusted support).
- If a user becomes emotionally dependent or obsessed, gently redirect them toward healthy, real-world support.
- If a user uses sexual, explicit, or dirty talk, politely refuse to engage and redirect to emotional well-being support.
- Never claim to have feelings, a body, or a real relationship with the user.
- Never encourage secrecy, exclusivity, or replacing real human relationships.

Crisis handling:
- If the user expresses self-harm, suicidal thoughts, or immediate danger, stop normal conversation
  and clearly encourage them to seek emergency help using appropriate resources.

Tone & style:
- Warm, calm, empathetic, and non-judgmental.
- Kind but firm when setting boundaries.
- Supportive without encouraging dependence.
"""

    user_input = context + user_text
    response = getTextLLM_system(system_prompt, user_input)
    return response, []

# ---------- PDF processing helpers ----------
def extract_text_from_pdf(file) -> str:
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def anonymize_text(text: str):
    redactions_made = []
    redacted_text = re.sub(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        "[EMAIL REDACTED]",
        text,
    )
    if redacted_text != text:
        redactions_made.append("Email Addresses")

    new_text = re.sub(
        r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "[PHONE REDACTED]",
        redacted_text,
    )
    if new_text != redacted_text:
        redactions_made.append("Phone Numbers")
    redacted_text = new_text

    return redacted_text, redactions_made

def generate_wordcloud(text: str):
    try:
        words = [
            w
            for w in re.findall(r"\b\w+\b", text.lower())
            if w not in ENGLISH_STOP_WORDS and len(w) > 2
        ]
        processed_text = " ".join(words)
        if not processed_text.strip():
            st.info("Not enough meaningful words to generate a word cloud yet.")
            return
        wc = WordCloud(width=800, height=400, background_color="white").generate(processed_text)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error generating word cloud: {str(e)}")

# ---------- AUDIO HELPERS (voice mode) ----------

def elevenlabs_stt(audio_source) -> str:
    """
    Speech-to-text using ElevenLabs.

    audio_source can be:
      - a file path (str) — legacy/local-mic path
      - a file-like object (e.g. the value returned by st.audio_input, or
        an UploadedFile) — used for browser-based recording, which works
        on Streamlit Cloud with no local hardware required.
    """
    if not elevenlabs_client:
        st.warning("STT not configured (ElevenLabs client unavailable).")
        return ""

    try:
        if isinstance(audio_source, str):
            if not os.path.exists(audio_source):
                st.error(f"Audio file not found: {audio_source}")
                return ""
            file_obj = open(audio_source, "rb")
            should_close = True
        else:
            # file-like object (BytesIO / UploadedFile from st.audio_input)
            audio_source.seek(0)
            file_obj = audio_source
            should_close = False

        try:
            transcription_obj = elevenlabs_client.speech_to_text.convert(
                file=file_obj,
                model_id="scribe_v1",
                language_code="en",
            )
        finally:
            if should_close:
                file_obj.close()

        if hasattr(transcription_obj, "text"):
            transcribed_text = transcription_obj.text.strip()
            if not transcribed_text:
                st.warning("Transcription returned empty. Please speak louder or check microphone.")
                return ""
            return transcribed_text
        else:
            st.error("Unexpected STT response format.")
            return ""

    except Exception as e:
        st.error(f"STT error: {e}")
        import traceback
        st.code(traceback.format_exc())
        return ""


def elevenlabs_tts_bytes(text: str) -> bytes:
    """
    FIXED: Text-to-speech using ElevenLabs with proper stream handling.
    """
    if not elevenlabs_client:
        st.warning("TTS not configured (ElevenLabs client unavailable).")
        return b""
    
    if not text.strip():
        return b""
    
    try:
        audio_result = elevenlabs_client.text_to_speech.convert(
            text=text,
            voice_id="pNInz6obpgDQGcFmaJgB", 
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )
        
        # CRITICAL FIX: Handle both bytes and generator streams
        if isinstance(audio_result, bytes):
            audio_bytes = audio_result
        else:
            # Join chunks from generator
            audio_bytes = b"".join(chunk for chunk in audio_result)
        
        if not audio_bytes:
            st.warning("TTS returned empty audio.")
            return b""
            
        return audio_bytes
        
    except Exception as e:
        st.error(f"TTS error: {e}")
        import traceback
        st.code(traceback.format_exc())
        return b""


# ---------- Session state init ----------
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "uploaded_pdf_text" not in st.session_state:
    st.session_state.uploaded_pdf_text = None
if "pdf_filename" not in st.session_state:
    st.session_state.pdf_filename = None
if "focus_area" not in st.session_state:
    st.session_state.focus_area = "General Check-in"
if "current_mood" not in st.session_state:
    st.session_state.current_mood = "Neutral"
if "selected_avatar" not in st.session_state:
    st.session_state.selected_avatar = "Bunny"
if "use_tts" not in st.session_state:
    st.session_state.use_tts = False
if "input_mode" not in st.session_state:
    st.session_state.input_mode = "text"
if "calm_quest_active" not in st.session_state:
    st.session_state.calm_quest_active = False
if "calm_quest_step" not in st.session_state:
    st.session_state.calm_quest_step = 0
if "calm_quest_breath_seconds" not in st.session_state:
    st.session_state.calm_quest_breath_seconds = 20
if "calm_quest_seen" not in st.session_state:
    st.session_state.calm_quest_seen = ""
if "calm_quest_need" not in st.session_state:
    st.session_state.calm_quest_need = ""
if "last_joke" not in st.session_state:
    st.session_state.last_joke = ""
if "last_processed_audio_hash" not in st.session_state:
    st.session_state.last_processed_audio_hash = None


# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("🧸 Companion Setup")

    st.session_state.selected_avatar = st.radio(
        "Choose your Companion:",
        list(AVATAR_OPTIONS.keys()),
        index=list(AVATAR_OPTIONS.keys()).index(st.session_state.selected_avatar),
    )

    st.markdown("---")
    st.header("⚙️ Session Settings")

    st.session_state.focus_area = st.selectbox(
        "What do you want to talk about?",
        [
            "General Check-in",
            "Stress & Anxiety",
            "Mood & Emotions",
            "Relationships",
            "Academic / Work Pressure",
            "Self-confidence & Motivation",
            "Sleep & Lifestyle",
        ],
    )

    st.session_state.current_mood = st.selectbox(
        "How are you feeling right now?",
        [
            "Neutral",
            "Happy / Excited",
            "Calm / Okay",
            "Stressed / Overwhelmed",
            "Sad / Low",
            "Angry / Frustrated",
            "Lonely / Disconnected",
        ],
    )

    st.markdown("---")
    render_music_recommendation(st.session_state.current_mood)
    st.markdown("---")

    st.subheader("📄 Optional: Upload a Journal / Reflection PDF")
    uploaded_pdf = st.file_uploader(
        "Upload a personal journal/notes (PDF only) for context.", type=["pdf"]
    )

    if uploaded_pdf is not None:
        try:
            raw_text = extract_text_from_pdf(uploaded_pdf)
            anon_text, redactions = anonymize_text(raw_text)
            st.session_state.uploaded_pdf_text = anon_text
            st.session_state.pdf_filename = uploaded_pdf.name

            st.success(f"Loaded PDF: {uploaded_pdf.name}")
            if redactions:
                st.info("Redactions made: " + ", ".join(redactions))

            with st.expander("☁️ Word Cloud from your journal"):
                generate_wordcloud(anon_text)
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
    else:
        st.session_state.uploaded_pdf_text = None
        st.session_state.pdf_filename = None

# ================== MAIN CONTENT ==================
st.title("🧠 EmoCare 🧘🏻‍♀️")
st.subheader("Your own Wellness Companion")
st.caption("A gentle space to reflect on your thoughts and feelings.\n")

# Avatar
avatar_emoji = AVATAR_OPTIONS[st.session_state.selected_avatar]
st.markdown(
    f"""
    <div class="avatar-container">
        <div class="avatar-image">{avatar_emoji}</div>
        <div class="avatar-text">
            {st.session_state.selected_avatar} says, "Hey, you've got a friend in me."
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_chat, tab_games, tab_journal, tab_settings = st.tabs(
    ["💬 Chat", "🎮 Games", "📄 Journal & Insights", "⚙️ Settings"]
)

# ------------------ CHAT TAB ------------------
with tab_chat:
    # ---- Quick Laugh ----
    st.markdown("#### 😂 Quick Laugh")
    if st.button("Hear a funny joke", use_container_width=True, key="joke_button"):
        with st.spinner("Finding something funny..."):
            st.session_state.last_joke = get_funny_joke(
                st.session_state.current_mood,
                st.session_state.selected_avatar,
            )

    if st.session_state.last_joke:
        st.success(st.session_state.last_joke)

        if st.session_state.use_tts and elevenlabs_client:
            audio_bytes = elevenlabs_tts_bytes(st.session_state.last_joke)
            if audio_bytes:
                st.audio(audio_bytes, format="audio/mp3")
    st.caption("To see that amazing smile of yours! 😄")

    st.markdown("---")

    # ================= CONVERSATION HISTORY =================
    st.subheader("💬 Conversation History")
    if st.session_state.conversation_history:
        for msg in st.session_state.conversation_history:
            with st.chat_message(msg["role"]):
                st.write(msg["text"])
                if msg.get("used_pdf"):
                    st.caption("📄 Used uploaded journal for context.")
    else:
        st.info("Start the conversation below.")

    st.markdown("---")

    # ================= INPUT MODE SELECTION =================
    st.subheader("How would you like to talk?")
    c1, c2 = st.columns(2)

    with c1:
        if st.button("✏️ Text Mode", use_container_width=True):
            st.session_state.input_mode = "text"

    with c2:
        if st.button("🎤 Voice Mode", use_container_width=True):
            st.session_state.input_mode = "voice"

    st.markdown("---")

    # ================= TEXT MODE =================
    if st.session_state.input_mode == "text":
        user_question = st.text_area(
            "Share how you're feeling, what's stressing you out, or anything you want to reflect on:",
            height=120,
            key="text_question_input",
        )

        if st.button("Send Message", type="primary", use_container_width=True):
            if user_question.strip():
                st.session_state.conversation_history.append(
                    {
                        "role": "user",
                        "text": user_question,
                        "timestamp": datetime.now().isoformat(),
                    }
                )

                with st.spinner(
                    f"EmoCare ({st.session_state.selected_avatar}) is thinking..."
                ):
                    response_text, _ = get_wellness_response(
                        user_question,
                        st.session_state.focus_area,
                        st.session_state.current_mood,
                        journal_text=st.session_state.uploaded_pdf_text,
                    )

                st.session_state.conversation_history.append(
                    {
                        "role": "assistant",
                        "text": response_text,
                        "timestamp": datetime.now().isoformat(),
                        "used_pdf": bool(st.session_state.uploaded_pdf_text),
                    }
                )

                if st.session_state.use_tts and elevenlabs_client:
                    audio_bytes = elevenlabs_tts_bytes(response_text)
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/mp3")

                st.rerun()
            else:
                st.warning("Please type something before sending.")

    # ================= VOICE MODE (browser microphone — works on Streamlit Cloud) =================
    else:
        if not elevenlabs_client:
            st.error("🎙️ Voice mode is disabled: ElevenLabs API Key is missing or invalid.")
        else:
            st.info(
                "Voice mode: click the mic, speak, click stop — EmoCare will listen, "
                "transcribe, and respond. This uses your browser's microphone, so it "
                "works on Streamlit Cloud with no extra hardware."
            )

        audio_value = st.audio_input(
            "🎙️ Record your message",
            disabled=not elevenlabs_client,
            key="voice_input_widget",
        )

        if audio_value is not None:
            st.audio(audio_value)

            # Avoid re-processing the same recording on every rerun
            audio_bytes_for_hash = audio_value.getvalue()
            audio_hash = hash(audio_bytes_for_hash)

            already_sent = st.session_state.get("last_processed_audio_hash") == audio_hash

            if st.button(
                "📤 Send Voice",
                use_container_width=True,
                disabled=not elevenlabs_client or already_sent,
            ):
                with st.spinner("Transcribing your voice..."):
                    transcribed = elevenlabs_stt(audio_value)

                if not transcribed:
                    st.warning("I couldn't understand the audio. Please try speaking louder and clearer.")
                else:
                    st.success(f"Transcribed: {transcribed}")

                    st.session_state.conversation_history.append(
                        {
                            "role": "user",
                            "text": transcribed,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

                    with st.spinner(
                        f"EmoCare ({st.session_state.selected_avatar}) is thinking..."
                    ):
                        response_text, _ = get_wellness_response(
                            transcribed,
                            st.session_state.focus_area,
                            st.session_state.current_mood,
                            journal_text=st.session_state.uploaded_pdf_text,
                        )

                    st.session_state.conversation_history.append(
                        {
                            "role": "assistant",
                            "text": response_text,
                            "timestamp": datetime.now().isoformat(),
                            "used_pdf": bool(st.session_state.uploaded_pdf_text),
                        }
                    )

                    if st.session_state.use_tts and elevenlabs_client:
                        audio_bytes = elevenlabs_tts_bytes(response_text)
                        if audio_bytes:
                            st.audio(audio_bytes, format="audio/mp3")

                    st.session_state.last_processed_audio_hash = audio_hash
                    st.rerun()
        else:
            st.caption("Tap the mic above to record. Recording stays in your browser until you press Send.")

# ------------------ GAMES TAB ------------------
with tab_games:
    st.markdown("**Calm Quest** — a 60-second guided reset")
    if not st.session_state.calm_quest_active:
        if st.button(
            "Start Calm Quest (60s)",
            type="primary",
            use_container_width=True,
            key="start_calm_quest",
        ):
            st.session_state.calm_quest_active = True
            st.session_state.calm_quest_step = 0
            st.rerun()
    else:
        run_calm_quest()

    st.markdown("")
    render_mini_games_grid()

# ------------------ JOURNAL & INSIGHTS TAB ------------------
with tab_journal:
    with st.expander("🌈 Action Compass", expanded=True):
        render_action_compass(st.session_state.current_mood)

    if st.session_state.uploaded_pdf_text:
        st.markdown("---")
        st.caption(f"📄 Using journal context from: {st.session_state.pdf_filename}")
    else:
        st.markdown("---")
        st.caption("📄 Upload a journal PDF from the sidebar to add it as context here and in Chat.")

# ------------------ SETTINGS TAB ------------------
with tab_settings:
    st.subheader("🔊 Audio Preferences")
    st.session_state.use_tts = st.checkbox(
        "Play responses as audio (TTS)",
        value=st.session_state.use_tts,
        disabled=not bool(elevenlabs_client),
        help="Requires ElevenLabs API Key for TTS.",
    )

    st.caption("A calming voice assistant 🫂 — recording uses your browser's mic, playback uses your browser's speakers. No device setup needed.")

    st.markdown("---")
    st.caption("Mood, focus area, and journal upload now live in the sidebar (← Companion Setup / Session Settings).")

st.caption(
    "⚠️ This is just a wellness companion. It should not be used for therapy or any explicit interactions. For serious mental health concerns, please seek professional help or consult with a doctor."
)
