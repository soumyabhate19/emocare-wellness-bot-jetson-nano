#!/usr/bin/env python
# coding: utf-8

from cProfile import label
import streamlit as st
import streamlit.components.v1 as components
try:
    from streamlit_float import float_init
    FLOAT_OK = True
except ImportError:
    FLOAT_OK = False
import os
import re
import time
import random
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


def flatten_html(html: str) -> str:
    """Strips leading whitespace from every line and joins into one line.
    Markdown treats lines indented 4+ spaces as a code block (raw text,
    not rendered HTML) — this avoids that entirely, regardless of how the
    HTML is indented in the Python source for readability."""
    return " ".join(line.strip() for line in html.strip().splitlines())


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


def build_action_compass_fixed_html(current_mood: str) -> str:
    """Builds the ENTIRE Action Compass panel (header, caption, all swatches)
    as one combined HTML string, so it can be rendered via a single
    st.markdown call wrapped in a position:fixed div with a direct inline
    style — the same reliable pattern used for the navbar, avoiding any
    dependency on Streamlit's own internal container selectors."""
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

    swatches_html = ""
    for mood, text in ACTIONS:
        active = mood.lower() in cm
        bg = "rgba(249,231,178,0.22)" if active else "rgba(249,231,178,0.10)"
        swatches_html += f"""
            <div style="background:{bg};border-radius:14px;padding:12px 14px;
                        margin-bottom:10px;color:#F9E7B2;line-height:1.5;">
              <b>{mood}</b> → {text}
            </div>
        """

    return f"""
        <div style="font-size:18px;font-weight:800;color:#F9E7B2;margin-bottom:4px;">
            🌈 Action Compass
        </div>
        <div style="font-size:13px;color:#FDF6E3;opacity:0.85;margin-bottom:12px;">
            Just a gentle nudge, not a rule 🌱
        </div>
        {swatches_html}
    """

# ------------------ Mini Games: open standalone HTML games hosted on GitHub Pages ------------------
# These files live in the /docs/games/ folder of your GitHub repo. GitHub Pages
# serves that folder as real https:// pages, which open reliably in a new tab —
# unlike data: URIs, which some Chrome builds/extensions block or mangle.
GAMES_BASE_URL = "https://soumyabhate19.github.io/emocare-wellness-bot-jetson-nano"

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
        "file": "trace_it.html",
    },
    {
        "id": "doodle_pad",
        "label": "🎨 Doodle Pad",
        "blurb": "Draw the prompt.",
        "file": "doodle_pad.html",
    },
    {
        "id": "counting_stars",
        "label": "✨ Counting Stars",
        "blurb": "Tap each star before it fades.",
        "file": "counting_stars.html",
    },
]


def render_mini_games_grid():
    st.caption("Or try one of these mini games")
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


# ------------------ Dinosaur Run: embedded inline (not a new tab) ------------------
DINO_GAME_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "games", "dino_run.html"
)
# Fallback URL (matches the same GAMES_BASE_URL pattern used by the other
# games above) — used if the local games/dino_run.html file isn't present
# in this deployment, so the embed doesn't depend on getting local file
# placement exactly right.
DINO_GAME_FALLBACK_URL = f"{GAMES_BASE_URL}/dino_run.html"


@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_dino_html_fallback() -> Optional[str]:
    try:
        import urllib.request
        with urllib.request.urlopen(DINO_GAME_FALLBACK_URL, timeout=8) as resp:
            return resp.read().decode("utf-8")
    except Exception:
        return None


def render_dino_run():
    if os.path.exists(DINO_GAME_PATH):
        with open(DINO_GAME_PATH, "r", encoding="utf-8") as f:
            dino_html = f.read()
        components.html(dino_html, height=250, scrolling=False)
        return

    # Local file missing — fall back to the GitHub Pages hosted copy.
    dino_html = _fetch_dino_html_fallback()
    if dino_html:
        components.html(dino_html, height=250, scrolling=False)
    else:
        st.warning(
            f"Dinosaur Run file not found locally at {DINO_GAME_PATH}, "
            f"and the fallback ({DINO_GAME_FALLBACK_URL}) couldn't be reached either."
        )




def run_calm_quest():
    # ------------------ Calm Quest: all 3 steps visible at once, unlocked in order ------------------
    st.caption("A tiny reset for your mind + body. All 3 steps are shown below — complete them in order.")

    if st.button("🛑 End Calm Quest", use_container_width=True):
        st.session_state.calm_quest_active = False
        st.session_state.calm_quest_step = 0
        st.session_state.calm_quest_seen = ""
        st.session_state.calm_quest_need = ""
        st.rerun()

    progress = st.session_state.calm_quest_step  # 0=not started, 1=breathing done, 2=grounding done, 3=complete

    # ---- Completion popup (persists until dismissed) ----
    if progress >= 3:
        st.markdown(
            """
            <div style="background:#F9E7B2;color:#4A2E10;border-radius:14px;
                        padding:18px 22px;text-align:center;margin-bottom:16px;">
                <div style="font-size:20px;font-weight:800;">💛 Hope you feel better now.</div>
                <div style="font-size:14px;margin-top:6px;">You showed up for yourself today — that counts.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Done", type="primary", use_container_width=True, key="cq_dismiss"):
            st.session_state.calm_quest_active = False
            st.session_state.calm_quest_step = 0
            st.session_state.calm_quest_seen = ""
            st.session_state.calm_quest_need = ""
            st.rerun()
        return

    st.markdown("---")

    # ---- Step 1: Breathing ----
    st.markdown("### Step 1 – Breathing Timer 🌬️")
    st.session_state.calm_quest_breath_seconds = st.slider(
        "Choose breathing time (seconds)",
        10, 45, st.session_state.calm_quest_breath_seconds,
        disabled=progress >= 1,
        key="cq_breath_slider",
    )

    if progress < 1:
        if st.button("▶️ Start Breathing", type="primary", use_container_width=True, key="cq_start_breath"):
            secs = st.session_state.calm_quest_breath_seconds
            bar = st.progress(0)
            status = st.empty()

            for i in range(secs):
                cue = "Inhale…" if (i // 4) % 2 == 0 else "Exhale…"
                status.markdown(f"**{cue}** ({secs - i}s left)")
                bar.progress(int((i + 1) / secs * 100))
                time.sleep(1)

            status.success("✅ Breathing complete.")
            st.session_state.calm_quest_step = 1
            st.rerun()
    else:
        st.success("✅ Breathing complete.")

    st.markdown("---")

    # ---- Step 2: Grounding ----
    st.markdown("### Step 2 – Grounding (3 things you see) 👀")
    if progress < 1:
        st.info("Complete Step 1 first.")

    st.session_state.calm_quest_seen = st.text_input(
        "Type 3 things you can see right now (comma-separated):",
        value=st.session_state.calm_quest_seen,
        placeholder="e.g., laptop, window, water bottle",
        disabled=(progress < 1 or progress >= 2),
        key="cq_seen_input",
    )

    if progress == 1:
        if st.button("Next ➡️", type="primary", use_container_width=True, key="cq_next_ground"):
            st.session_state.calm_quest_step = 2
            st.rerun()
    elif progress >= 2:
        st.success("✅ Grounding complete.")

    st.markdown("---")

    # ---- Step 3: Journal ----
    st.markdown("### Step 3 – One-line Journal ✏️")
    if progress < 2:
        st.info("Complete Step 2 first.")

    st.session_state.calm_quest_need = st.text_area(
        "Finish this sentence: **Right now I need…**",
        value=st.session_state.calm_quest_need,
        height=100,
        placeholder="…a break, clarity, reassurance, a plan, rest, etc.",
        disabled=(progress < 2 or progress >= 3),
        key="cq_need_input",
    )

    if progress == 2:
        if st.button("🏆 Finish Quest", type="primary", use_container_width=True, key="cq_finish"):
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

            st.session_state.calm_quest_step = 3
            st.balloons()
            st.rerun()


# ---------- Joke generator ----------
# ---------- Curated jokes (quality-controlled — not AI-generated on the fly) ----------
# LLMs asked to "be funny" reliably produce corny dad-jokes, so these are hand-picked,
# clean, wholesome one-liners. One shared pool — not tied to which companion is chosen.
JOKES = [
    "I told my plant a joke about photosynthesis. It didn't laugh, but it did grow a little that day.",
    "Why do bunnies never argue? Because they always hop to a compromise.",
    "I'm not lazy, I'm just on energy-saving mode. Like a phone, but cuter.",
    "My bed and I have a special bond. It's the longest relationship I've ever committed to.",
    "I asked the universe for a sign. It sent me a parking ticket. Thanks, universe.",
    "Some days I'm a productivity powerhouse. Today I'm a productivity... house.",
    "Why did the carrot blush? Because it saw the salad dressing.",
    "I told my houseplant I loved it. It didn't say it back. We're working through it.",
    "My favorite yoga pose is called 'lying very still and hoping nobody asks me to do anything.'",
    "I'm basically a soft pretzel — a little twisted, but ultimately harmless and worth keeping around.",
    "Fun fact: bunnies can't burp. Honestly, kind of jealous. Must be nice to keep secrets that well.",
    "I've started saying 'I'm not procrastinating, I'm marinating.' It hasn't fooled anyone yet.",
    "I'm 70% water and 30% strong opinions about snacks.",
    "My gym routine is called 'bear crawl to the fridge and back.'",
    "I don't do mornings. Mornings do me, and honestly, we should talk about consent.",
    "Being an adult is just googling how to do things while pretending you already know.",
    "I'm not arguing, I'm just explaining why I'm right in a slightly louder voice.",
    "I put the 'pro' in procrastinate. Everything else is still a work in progress.",
    "My relationship status: in a committed partnership with my blanket.",
    "I run on two things: snacks and spite. Mostly snacks.",
    "I'm not a morning person or a night owl. I'm some kind of tired all-day pigeon.",
    "I told myself I'd stop talking to myself. Now we're not speaking.",
    "I have the focus of a Roomba that just found a sock.",
    "Panda fact: pandas sleep up to 10 hours a day. I consider this a personal goal, not a fun fact.",
    "I tried to catch some fog earlier. I mist.",
    "A skeleton walked into a bar and ordered a beer and a mop.",
    "I'm reading a book about anti-gravity. It's impossible to put down.",
    "I used to be a banker, but I lost interest.",
    "My blanket fort has better structural integrity than most of my life decisions.",
    "I told my computer I needed a break. Now it won't stop sending me vacation ads.",
    "I'm not clumsy, the floor just hates me sometimes.",
    "Why don't scientists trust atoms? Because they make up literally everything.",
    "I put googly eyes on the fridge. Now every snack decision feels supervised.",
    "I tried yoga. Turns out 'downward dog' is not just a suggestion, it's a warning.",
    "I named my Wi-Fi 'Tell My Wi-Fi Love Her' so when it disconnects it says 'Tell My Wi-Fi Love Her has disconnected.'",
    "I asked my dog what's two minus two. He said nothing.",
]


def get_funny_joke(mood: str, avatar: str) -> str:
    """Returns a joke from a shuffled 'bag' so nothing repeats until every
    joke in the pool has been shown once in this session."""
    bag = st.session_state.get("joke_bag") or []

    if not bag:
        bag = JOKES.copy()
        random.shuffle(bag)

    joke = bag.pop()
    st.session_state.joke_bag = bag
    return joke


# ---------- EmoCare avatar & theme config ----------
AVATAR_OPTIONS = {
    "Bunny": "🐰",
    "Pandy": "🐼",
    "Silly": "🦭"
}

# ---------- Streamlit page config ----------
st.set_page_config(page_title="AI Wellness Companion", layout="wide", page_icon="🧠")

if FLOAT_OK:
    float_init()

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
- If a user expresses romantic feelings toward you (e.g., "I love you", "be my partner", "don't leave me"),
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

# ---- Navbar: jump-links to each section (plain anchor scrolling) ----
# position:fixed with left:0/right:0 spans the FULL browser viewport on its
# own layer, completely outside the column layout below — so it can never
# overlap the right panel (Action Compass), regardless of column widths.
# Placed here, before the columns are created, so it's unambiguously not
# nested inside either one.
st.markdown(
    flatten_html(
        """
        <div class="emocare-navbar" style="
            position:fixed; top:3.5rem; left:0; right:0; z-index:999;
            background:rgba(102,50,31,0.82);
            backdrop-filter:blur(6px); -webkit-backdrop-filter:blur(6px);
            padding:12px 24px;
            display:flex; justify-content:center; gap:30px; flex-wrap:wrap;
            align-items:center;
            box-shadow:0 4px 14px rgba(0,0,0,0.25);
        ">
            <a class="emocare-navbar-link" href="#calm-quest-section" style="color:#F9E7B2;text-decoration:none;font-weight:700;font-size:14px;">Calm Quest</a>
            <a class="emocare-navbar-link" href="#games-section" style="color:#F9E7B2;text-decoration:none;font-weight:700;font-size:14px;">Mini Games</a>
            <a class="emocare-navbar-link" href="#dino-section" style="color:#F9E7B2;text-decoration:none;font-weight:700;font-size:14px;">Dinosaur Run</a>
            <a class="emocare-navbar-link" href="#quick-laugh-section" style="color:#F9E7B2;text-decoration:none;font-weight:700;font-size:14px;">Quick Laugh</a>
            <a class="emocare-navbar-link" href="#chatbot-section" style="color:#F9E7B2;text-decoration:none;font-weight:700;font-size:14px;">Chatbot</a>
            <a class="emocare-navbar-link" href="#contact-section" style="color:#F9E7B2;text-decoration:none;font-weight:700;font-size:14px;">Contact Me</a>
        </div>
        """
    )
    + '<div class="emocare-navbar-spacer" style="height:52px;"></div>',
    unsafe_allow_html=True,
)

# ---- Action Compass: fixed panel on the right, same pattern as the navbar ----
# One combined HTML string in a single div with a direct inline
# position:fixed style — not relying on any Streamlit-generated selector,
# so it isn't affected by Streamlit's own container/overflow quirks.
st.markdown(
    flatten_html(
        f"""
        <div class="emocare-fixed-compass" style="
            position:fixed; top:8.5rem; right:1.5rem; z-index:998;
            width:260px; max-height:calc(100vh - 10rem); overflow-y:auto;
            background:#66321F; border-radius:14px; padding:16px;
            box-shadow:0 10px 24px rgba(0,0,0,0.25);
        ">
            {build_action_compass_fixed_html(st.session_state.current_mood)}
        </div>
        """
    ),
    unsafe_allow_html=True,
)

# ================== MAIN CONTENT: CENTER + RIGHT PANEL ==================
center_col, right_col = st.columns([2.7, 1.0], gap="large")

# ------------------ CENTER PANEL ------------------
with center_col:
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

    # ---- Calm Quest (its own section) ----
    st.markdown('<div id="calm-quest-section"></div>', unsafe_allow_html=True)
    st.markdown("### 🧘 Calm Quest")
    st.caption("A 60-second guided reset — breathing, grounding, and reflection.")

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

    st.markdown("---")

    # ---- Games (separate section) ----
    st.markdown('<div id="games-section"></div>', unsafe_allow_html=True)
    st.markdown("### 🎮 Mini Games")
    render_mini_games_grid()

    st.markdown("---")

    # ---- Dinosaur Run (embedded inline, not a new tab) ----
    st.markdown('<div id="dino-section"></div>', unsafe_allow_html=True)
    st.markdown("### 🦖 Dinosaur Run")
    st.caption("The classic offline dino game — jump the cacti, don't stop running.")
    render_dino_run()

    st.markdown("---")

    # ---- Quick Laugh ----
    st.markdown('<div id="quick-laugh-section"></div>', unsafe_allow_html=True)
    st.markdown("#### 😂 Quick Laugh")
    if st.button("Hear a funny joke", use_container_width=True, key="joke_button"):
        with st.spinner("Finding something funny..."):
            st.session_state.last_joke = get_funny_joke(
                st.session_state.current_mood,
                st.session_state.selected_avatar,
            )

    if st.session_state.last_joke:
        st.success(st.session_state.last_joke)
    st.caption("To see that amazing smile of yours! 😄")

    st.markdown("---")

    # ================= CONVERSATION HISTORY =================
    st.markdown('<div id="chatbot-section"></div>', unsafe_allow_html=True)
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

    # ================= TEXT INPUT =================
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

            st.rerun()
        else:
            st.warning("Please type something before sending.")

    st.caption(
        "⚠️ This is just a wellness companion. It should not be used for therapy or any explicit interactions. For serious mental health concerns, please seek professional help or consult with a doctor."
    )

    st.markdown("---")

    # ---- Contact Me (anchor + footer-style info) ----
    st.markdown('<div id="contact-section"></div>', unsafe_allow_html=True)
    st.markdown(
        flatten_html(
            """
            <div style="text-align:center;padding:18px 10px 6px;">
                <div style="display:flex;flex-wrap:wrap;justify-content:center;
                            align-items:center;gap:26px;margin-bottom:16px;">

                    <a href="mailto:soumyabhate19@gmail.com"
                       style="display:flex;align-items:center;gap:7px;
                              color:#F9E7B2;text-decoration:none;font-size:14px;">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                            <path d="M2 5.5A2.5 2.5 0 0 1 4.5 3h15A2.5 2.5 0 0 1 22 5.5v13a2.5 2.5 0 0 1-2.5 2.5h-15A2.5 2.5 0 0 1 2 18.5v-13zm2.2.3 7.4 5.9a.7.7 0 0 0 .8 0l7.4-5.9a.6.6 0 0 0-.4-1.1H4.6a.6.6 0 0 0-.4 1.1z"/>
                        </svg>
                        soumyabhate19@gmail.com
                    </a>

                    <a href="https://linkedin.com/in/soumyabhate19" target="_blank" rel="noopener noreferrer"
                       style="display:flex;align-items:center;gap:7px;
                              color:#F9E7B2;text-decoration:none;font-size:14px;">
                        <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                            <path d="M14.82 0H1.18C.53 0 0 .53 0 1.18v13.64C0 15.47.53 16 1.18 16h13.64c.65 0 1.18-.53 1.18-1.18V1.18C16 .53 15.47 0 14.82 0zM4.75 13.4H2.4V6h2.35v7.4zM3.58 5.02a1.36 1.36 0 1 1 0-2.72 1.36 1.36 0 0 1 0 2.72zM13.6 13.4h-2.35V9.8c0-.86-.02-1.97-1.2-1.97-1.2 0-1.39.94-1.39 1.9v3.67H6.31V6h2.26v1.01h.03c.31-.59 1.08-1.2 2.22-1.2 2.38 0 2.82 1.57 2.82 3.6v3.99z"/>
                        </svg>
                        soumyabhate19
                    </a>

                    <a href="https://github.com/soumyabhate19" target="_blank" rel="noopener noreferrer"
                       style="display:flex;align-items:center;gap:7px;
                              color:#F9E7B2;text-decoration:none;font-size:14px;">
                        <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                            <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>
                        </svg>
                        soumyabhate19
                    </a>

                </div>
                <hr style="border-color:rgba(249,231,178,0.3);margin:14px 0;">
                <div style="display:flex;justify-content:space-between;
                            flex-wrap:wrap;gap:8px;color:#FDF6E3;
                            font-size:13px;opacity:0.85;padding:0 4px;">
                    <span>Baltimore, MD</span>
                    <span>Soumya Bhate · EmoCare © 2026</span>
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

# ------------------ RIGHT PANEL ------------------
# (Action Compass now renders as a fixed panel above, outside the column
# layout — right_col is left empty here just to preserve center_col's width.)
with right_col:
    pass
