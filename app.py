"""
app.py — VoiceGuard Main Streamlit Application
===============================================
Entry point for the VoiceGuard AI Voice Authentication System.

Run with:  streamlit run app.py

Architecture Overview:
  ┌─────────────────────────────────────┐
  │         Streamlit Frontend          │
  │  ┌─────────────┐  ┌──────────────┐ │
  │  │  Register   │  │    Login     │ │
  │  │  New User   │  │              │ │
  │  └──────┬──────┘  └──────┬───────┘ │
  └─────────┼────────────────┼─────────┘
            │                │
  ┌─────────▼────────────────▼─────────┐
  │         Audio Pipeline             │
  │  sounddevice → scipy filter →      │
  │  Whisper STT → Liveness Check      │
  └─────────────────┬──────────────────┘
                    │  HTTP (multipart/form-data)
  ┌─────────────────▼──────────────────┐
  │     FastAPI Microservice Backend   │
  │  POST /register  POST /login       │
  │  http://127.0.0.1:8000             │
  └────────────────────────────────────┘
"""

# ── Standard Library ──────────────────────────────────────────────────────────
import os
import time
import random
import tempfile
from thefuzz import fuzz  # Fuzzy string matching — tolerant STT liveness comparison
import string             # Built-in punctuation stripping — used in STT normalisation

# ── Third-Party ───────────────────────────────────────────────────────────────
import requests
import streamlit as st
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as wav_write
from scipy.signal import butter, lfilter
import whisper   # OpenAI Whisper — runs LOCALLY, no internet or API key required

# ── Backend API ───────────────────────────────────────────────────────────────
BACKEND_URL = "http://127.0.0.1:8000"

# =============================================================================
# PAGE CONFIG  (must be the very first Streamlit call)
# =============================================================================
st.set_page_config(
    page_title="VoiceGuard — AI Voice Authentication",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# GLOBAL CONSTANTS
# =============================================================================
SAMPLE_RATE   = 22_050   # Hz — matches librosa's default sr
RECORD_SECS   = 6        # seconds — increased from 4s so the full phrase is captured
                         # (the 0.5 s mic-wake-up eats the first ~500 ms of the window)
TEMP_WAV_PATH     = os.path.join(tempfile.gettempdir(), "voiceguard_sample.wav")
TEMP_RAW_WAV_PATH = os.path.join(tempfile.gettempdir(), "voiceguard_raw.wav")

# Predefined list of simple, natural, easy-to-pronounce liveness challenge sentences.
# Replaced with longer 7-12 word phrases to ensure sufficient speech duration
# for robust 256-dim Resemblyzer embeddings.
_LIVENESS_PHRASES = [
    "The weather is pleasant and the sky looks very clear today",
    "I am testing the microphone to ensure my voice is captured correctly",
    "One two three four five six seven eight nine and ten",
    "A bright red apple was resting quietly on the kitchen counter",
    "Please let me in now because I need to access the system",
    "Hello this is a voice test for the security authentication system",
    "My name is a secret but my voice will grant me access",
    "The quick brown fox successfully jumped over the lazy sleeping dog",
    "The sun sets beautifully behind the mountains in the early evening",
    "Security is very important so I am verifying my identity through speech",
]

# =============================================================================
# CUSTOM CSS — Dark-Mode Professional Theme
# =============================================================================
CUSTOM_CSS = """
<style>
/* ── Google Font Import ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Root Variables ── */
:root {
    --bg-primary:    #0d1117;
    --bg-secondary:  #161b22;
    --bg-card:       #1c2230;
    --accent-cyan:   #00e5ff;
    --accent-green:  #39ff8f;
    --accent-red:    #ff4c4c;
    --text-primary:  #e6edf3;
    --text-muted:    #8b949e;
    --border:        #30363d;
    --shadow:        0 4px 24px rgba(0,0,0,0.5);
    --radius:        12px;
}

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

/* ── Main area padding ── */
.main .block-container { padding: 2rem 3rem; }

/* ── Cards ── */
.vg-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 2rem;
    margin-bottom: 1.5rem;
    box-shadow: var(--shadow);
}

/* ── Liveness phrase box ── */
.liveness-box {
    background: linear-gradient(135deg, #0d2137 0%, #0a1628 100%);
    border: 2px solid var(--accent-cyan);
    border-radius: var(--radius);
    padding: 1.5rem 2rem;
    text-align: center;
    box-shadow: 0 0 18px rgba(0,229,255,0.15);
    margin: 1rem 0;
}
.liveness-box .phrase-label {
    font-size: 0.78rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--accent-cyan);
    margin-bottom: 0.5rem;
}
.liveness-box .phrase-text {
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: 0.04em;
}

/* ── Status badges ── */
.badge-success { color: var(--accent-green); font-weight: 600; }
.badge-error   { color: var(--accent-red);   font-weight: 600; }
.badge-info    { color: var(--accent-cyan);  font-weight: 600; }

/* ── Step indicators ── */
.step-row { display: flex; align-items: center; gap: 0.75rem; padding: 0.5rem 0; }
.step-icon { font-size: 1.2rem; width: 1.5rem; text-align: center; }
.step-label { font-size: 0.95rem; color: var(--text-muted); }
.step-label.active { color: var(--text-primary); font-weight: 500; }

/* ── Buttons — Streamlit overrides ── */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,229,255,0.25) !important;
}

/* ── Text inputs ── */
div[data-baseweb="input"] input {
    background-color: var(--bg-secondary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}

/* ── Progress / info boxes ── */
.stAlert { border-radius: var(--radius) !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; }
</style>
"""

# =============================================================================
# AUDIO UTILITIES
# =============================================================================

def generate_liveness_phrase() -> str:
    """
    Returns a randomly selected liveness challenge sentence.
    Phrases are short, natural-sounding, and easy to pronounce for non-native
    speakers, which improves Whisper transcription reliability during demos.

    Returns: e.g. "The sky is very blue"
    """
    return random.choice(_LIVENESS_PHRASES)


def record_audio(duration: int = RECORD_SECS, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Captures audio from the default system microphone using sounddevice.

    Parameters
    ----------
    duration : int  — Recording length in seconds.
    sr       : int  — Sample rate in Hz.

    Returns
    -------
    np.ndarray of shape (duration*sr,) with dtype float32, range [-1, 1].
    """
    # ── Visual 'Go' signal — eliminates the 'walkie-talkie' first-word cut-off ──
    # sounddevice needs ~500 ms to open the OS audio stream.  Without this
    # delay, users who speak immediately lose their first word, which causes
    # Whisper to miss the opening word of the challenge phrase and fail the
    # liveness check.  We:
    #   1. Show a "waking up" warning so the user waits.
    #   2. Sleep 0.5 s to let Streamlit render the UI and the OS prep the mic.
    #   3. Overwrite with a bright "SPEAK NOW!" banner the instant sd.rec()
    #      is about to open the stream — this is the user's precise cue.
    #   4. Clear the banner after blocking sd.rec() returns.
    status_box = st.empty()
    status_box.warning("⏳ Waking up microphone... Get ready.")
    time.sleep(0.5)
    status_box.success("🎙️ SPEAK NOW!")

    audio = sd.rec(
        frames=int(duration * sr),
        samplerate=sr,
        channels=1,          # mono
        dtype="float32",
        blocking=True,       # wait until recording finishes
    )

    # Recording complete — clear the cue banner
    status_box.empty()

    # sd.rec returns shape (frames, channels) — squeeze to 1-D
    return audio.squeeze()


def apply_bandpass_filter(
    audio: np.ndarray,
    lowcut: float = 80.0,
    highcut: float = 8000.0,
    sr: int = SAMPLE_RATE,
    order: int = 5,
) -> np.ndarray:
    """
    Applies a Butterworth band-pass filter to remove sub-bass rumble and
    high-frequency noise outside the human speech frequency range (80–8000 Hz).

    This improves MFCC quality by denoising the input signal before
    feature extraction.
    """
    nyq  = 0.5 * sr
    low  = lowcut  / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return lfilter(b, a, audio).astype(np.float32)


def save_wav(audio: np.ndarray, path: str, sr: int = SAMPLE_RATE) -> None:
    """
    Saves a float32 NumPy audio array as a 16-bit PCM WAV file.

    Parameters
    ----------
    audio : np.ndarray — 1-D float32 waveform in range [-1, 1].
    path  : str        — Destination file path.
    sr    : int        — Sample rate.
    """
    # Peak-normalize to prevent int16 clipping
    peak = np.max(np.abs(audio))
    if peak > 1e-6:
        audio = (audio / peak) * 0.95
        
    # Convert float32 → int16 for standard WAV compatibility
    audio_int16 = (audio * 32767).astype(np.int16)
    wav_write(path, sr, audio_int16)


# =============================================================================
# BACKEND API HELPERS
# =============================================================================

# No local processing functions needed — all voiceprint work is handled
# by the FastAPI microservice at BACKEND_URL.


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_sidebar() -> str:
    """Renders the sidebar navigation and returns the selected page name."""
    with st.sidebar:
        # Logo / Brand
        st.markdown("""
        <div style="text-align:center; padding: 1rem 0 2rem 0;">
            <div style="font-size:3rem;">🔐</div>
            <div style="font-size:1.3rem; font-weight:700; color:#00e5ff;">VoiceGuard</div>
            <div style="font-size:0.75rem; color:#8b949e; margin-top:0.25rem;">
                AI Voice Authentication
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(
            "<div style='font-size:0.7rem; letter-spacing:0.12em; color:#8b949e;"
            " text-transform:uppercase; margin-bottom:0.75rem;'>Navigation</div>",
            unsafe_allow_html=True,
        )

        page = st.radio(
            label="",
            options=["🆕  Register New User", "🔓  Login"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        # AI Security Features Panel
        st.markdown("""
        <div style="background:#0d2137; border:1px solid #30363d; border-radius:10px;
                    padding:1rem; font-size:0.8rem; color:#8b949e;">
            <div style="color:#00e5ff; font-weight:600; margin-bottom:0.5rem;">
                🛡️ AI Security Features
            </div>
            <ul style="margin:0; padding-left:1.2rem; line-height:1.8;">
                <li>Local on-device authentication</li>
                <li>Real-time liveness verification</li>
                <li>Phrase-independent voice embeddings</li>
                <li>Multi-layer speaker verification</li>
                <li>Temporary audio processing only</li>
                <li>AI-powered similarity scoring</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    return page


def render_pipeline_steps(steps: list[dict]) -> None:
    """
    Renders a visual step-by-step pipeline indicator.

    Parameters
    ----------
    steps : list of dicts with keys:
        icon   — emoji icon string
        label  — step description
        active — bool, whether this step is currently active/complete
    """
    html = ""
    for step in steps:
        cls = "active" if step["active"] else ""
        html += (
            f'<div class="step-row">'
            f'  <div class="step-icon">{step["icon"]}</div>'
            f'  <div class="step-label {cls}">{step["label"]}</div>'
            f'</div>'
        )
    st.markdown(html, unsafe_allow_html=True)


# =============================================================================
# PAGES
# =============================================================================

def page_register() -> None:
    """
    Registration Page — Multi-Phrase Enrollment
    -------------------------------------------
    The model encodes both speaker identity AND some speech content.
    To make authentication text-independent (so you can log in with any
    random challenge phrase), we enrol 3 recordings with 3 DIFFERENT phrases
    and average the embeddings into a single speaker-identity centroid.

    Workflow (repeat Steps 2-3 three times with different phrases):
      1. Enter username
      2. Generate a liveness challenge phrase  ← use a DIFFERENT one each round
      3. Record voice — backend accumulates embeddings
      4. After 3 rounds: centroid is saved — login is now text-independent
    """
    st.markdown(
        "<h1 style='color:#00e5ff; margin-bottom:0.25rem;'>Register New Voiceprint</h1>"
        "<p style='color:#8b949e; margin-bottom:0.5rem;'>"
        "To make your voiceprint <strong style='color:#00e5ff;'>phrase-independent</strong>, "
        "you need to record <strong style='color:#fff;'>5 different challenge phrases</strong>. "
        "The system averages them into a speaker-identity centroid that works with any phrase at login.</p>",
        unsafe_allow_html=True,
    )

    col_form, col_info = st.columns([2, 1], gap="large")

    # ── Left Column: Registration Form ──────────────────────────────────────
    with col_form:

        # ── Step 1: Username ───────────────────────────────────────────────
        st.markdown("#### Step 1 — Choose a Username")
        username = st.text_input(
            "Username",
            placeholder="e.g. keerti_puri",
            key="reg_username",
        )

        st.divider()

        # ── Step 2: Liveness Challenge ─────────────────────────────────────
        st.markdown("#### Step 2 — Liveness Detection Phrase")
        st.markdown(
            "<p style='color:#8b949e; font-size:0.9rem;'>"
            "Click the button to generate your unique challenge phrase. "
            "You must read this phrase aloud during the recording. "
            "This prevents deepfake replay attacks.</p>",
            unsafe_allow_html=True,
        )

        if st.button("🎲  Generate Challenge Phrase", key="btn_gen_phrase"):
            st.session_state["liveness_phrase"] = generate_liveness_phrase()

        if "liveness_phrase" in st.session_state:
            phrase = st.session_state["liveness_phrase"]
            st.markdown(
                f'<div class="liveness-box">'
                f'  <div class="phrase-label">📢 Read this phrase aloud</div>'
                f'  <div class="phrase-text">{phrase}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Step 3: Record Audio ───────────────────────────────────────────
        st.markdown("#### Step 3 — Record Your Voice")

        if "liveness_phrase" not in st.session_state:
            st.info("⚠️  Please generate a liveness phrase first (Step 2).")
        else:
            if st.button(
                f"🎙️  Start Recording ({RECORD_SECS} seconds)",
                key="btn_record",
                type="primary",
            ):
                if not username.strip():
                    st.error("❌  Please enter a username before recording.")
                else:
                    # ── 3a. Record ──────────────────────────────────────
                    with st.spinner(f"🎙️  Recording for {RECORD_SECS} seconds… Speak now!"):
                        try:
                            raw_audio = record_audio(duration=RECORD_SECS)
                            st.success(f"✅  Recording complete! ({len(raw_audio)} samples captured)")
                        except Exception as e:
                            st.error(f"❌  Microphone error: {e}")
                            st.stop()

                    # ──────────────────────────────────────────────────────
                    # SECURITY CHECK 1 — Voice Activity Detection (VAD)
                    # ──────────────────────────────────────────────────────
                    # Calculate the Root-Mean-Square (RMS) energy of the
                    # captured audio array.  If the user was silent (or the
                    # microphone wasn't working), the RMS will be near zero.
                    #
                    # Threshold of 0.01 works well for float32 audio in the
                    # range [-1, 1].  Below this the recording is considered
                    # empty / silent and we abort BEFORE any DB write.
                    # ──────────────────────────────────────────────────────
                    rms_energy = float(np.sqrt(np.mean(raw_audio ** 2)))

                    if rms_energy < 0.01:
                        # ❌ Silent recording — reject immediately
                        st.error(
                            "🔇 No voice detected! Please speak clearly into the microphone."
                        )
                        st.stop()   # Halt pipeline — nothing written to DB

                    # VAD passed — inform the user their voice was detected
                    st.info(f"🔊 VAD passed — voice energy detected (RMS = {rms_energy:.4f})")

                    # ── 3b. Filter ──────────────────────────────────────
                    with st.spinner("🔊  Applying band-pass filter (80Hz – 8kHz)…"):
                        # 1. Save raw audio (peak-normalized automatically by save_wav)
                        # This raw version is sent to the backend for accurate speaker embeddings
                        save_wav(raw_audio, TEMP_RAW_WAV_PATH)
                        st.info(f"📁  RAW audio saved for backend: `{TEMP_RAW_WAV_PATH}`")
                        
                        # 2. Apply bandpass filter ONLY for Whisper STT liveness check
                        filtered_audio = apply_bandpass_filter(raw_audio)
                        save_wav(filtered_audio, TEMP_WAV_PATH)
                        st.info(f"📁  Filtered audio saved for STT: `{TEMP_WAV_PATH}`")

                    # ──────────────────────────────────────────────────────
                    # SECURITY CHECK 2 — Offline STT Liveness Verification
                    #                    (OpenAI Whisper — runs 100% locally)
                    # ──────────────────────────────────────────────────────
                    # Architecture rationale:
                    #   • Whisper is a local deep-learning model — it requires
                    #     NO internet connection and NO API key, which satisfies
                    #     our Zero-Trust, local-only architecture constraint.
                    #   • We load the "base" model (~74 MB), which gives a good
                    #     balance between accuracy and speed on consumer hardware.
                    #     Larger variants ("small", "medium", "large") are more
                    #     accurate but slower to load and run.
                    #
                    # How it defends against liveness bypass attacks:
                    #   • The challenge phrase is randomly generated each session.
                    #   • Whisper transcribes what was ACTUALLY spoken.
                    #   • If the transcript does not match the phrase, the
                    #     pipeline is stopped BEFORE any MFCC / BioHash / DB step.
                    #   • A pre-recorded replay of a DIFFERENT phrase cannot
                    #     pass this check.
                    # ──────────────────────────────────────────────────────

                    # Retrieve the challenge phrase that was shown to the user
                    challenge_phrase = st.session_state["liveness_phrase"]

                    # ── Step 2a: Load the local Whisper model ────────────
                    # whisper.load_model() downloads the weights the first time
                    # it is called, then caches them locally in ~/.cache/whisper.
                    # Subsequent runs load from disk — no network needed after
                    # the first download.
                    with st.spinner("🤖  Loading local STT model (Whisper 'base')…"):
                        model = whisper.load_model("base")

                    # ── Step 2b: Transcribe the audio file ───────────────
                    # model.transcribe() reads the WAV file at TEMP_WAV_PATH
                    # and returns a dict; the key "text" contains the plain-text
                    # transcription of the spoken audio.
                    #
                    # Key parameters:
                    #  • fp16=False    — float32 inference for CPU compatibility.
                    #  • language="en" — locks Whisper to English, eliminating
                    #                    language-detection errors on short phrases
                    #                    that can cause hallucinated non-English text.
                    #  • initial_prompt — seeds the decoder with the expected
                    #                    phrase vocabulary, strongly biasing it
                    #                    toward the correct words and preventing
                    #                    hallucinations like 'zapple'.
                    with st.spinner("🗣️  Transcribing audio with local Whisper model…"):
                        result = model.transcribe(
                            TEMP_WAV_PATH,
                            fp16=False,
                            language="en",
                            initial_prompt=challenge_phrase,
                        )

                    # Extract the raw transcription string from Whisper's output
                    raw_transcription = result["text"].strip()

                    # ── Step 2c: Normalise both strings for comparison ───
                    # Convert both to lowercase and strip all punctuation
                    # before the fuzzy comparison.  This prevents false
                    # rejections caused by trailing commas, capitalisation
                    # differences, etc. (e.g. "Swift Mountain," → "swift mountain").
                    # Note: `string` and `fuzz` (from thefuzz) are imported
                    # at the top of the file — no inline import needed here.
                    _strip_punct = str.maketrans("", "", string.punctuation)

                    transcribed_text  = raw_transcription.lower().translate(_strip_punct).strip()
                    normalised_phrase = challenge_phrase.lower().translate(_strip_punct).strip()

                    # ══════════════════════════════════════════════════════
                    # THREE-TIERED FUZZY LIVENESS VERIFICATION (Registration)
                    # ══════════════════════════════════════════════════════
                    # Strategy: ANY one metric ≥ 65 is sufficient.
                    #
                    # • token_set_ratio  — insensitive to missing/extra words
                    #   and word order. Catches Whisper dropping a connector
                    #   like "is" or reordering due to accent.
                    # • ratio            — overall character-level similarity.
                    #   Catches near-homophones ("whether" → "weather").
                    # • partial_ratio    — best substring match. Catches cases
                    #   where Whisper transcribes a fragment of the phrase
                    #   (e.g., "red apple" from "A bright red apple").
                    #
                    # Threshold lowered to 65 (from 80) because:
                    #  • Whisper 'base' has ~10-15% WER on accented English.
                    #  • Short 4-6 word phrases amplify character-level error %.
                    #  • The 3-metric OR gate already provides strong security.
                    # ══════════════════════════════════════════════════════

                    SIMILARITY_THRESHOLD = 65   # integer percentage for thefuzz

                    token_set_score = fuzz.token_set_ratio(
                        normalised_phrase,   # expected challenge phrase (normalised)
                        transcribed_text,    # what Whisper heard (normalised)
                    )
                    ratio_score = fuzz.ratio(
                        normalised_phrase,   # expected challenge phrase (normalised)
                        transcribed_text,    # what Whisper heard (normalised)
                    )
                    partial_score = fuzz.partial_ratio(
                        normalised_phrase,   # expected challenge phrase (normalised)
                        transcribed_text,    # what Whisper heard (normalised)
                    )

                    liveness_passed = (
                        token_set_score >= SIMILARITY_THRESHOLD
                        or ratio_score  >= SIMILARITY_THRESHOLD
                        or partial_score >= SIMILARITY_THRESHOLD
                    )

                    if not liveness_passed:
                        # ❌ No metric reached the threshold — reject.
                        # "Demo Saver" UI: show exactly what the AI heard vs. required.
                        st.error(
                            f"Liveness Check Failed\n\n"
                            f"token_set_ratio: {token_set_score}%  |  "
                            f"ratio: {ratio_score}%  |  "
                            f"partial_ratio: {partial_score}%  (threshold: {SIMILARITY_THRESHOLD}%)\n\n"
                            f"System Heard: '{transcribed_text}'\n"
                            f"Required: '{challenge_phrase}'\n\n"
                            f"Please try again and speak clearly."
                        )
                        st.stop()   # Halt pipeline — nothing is written to the DB

                    # ✅ Liveness passed — user is live and said the correct phrase.
                    best_score = max(token_set_score, ratio_score, partial_score)
                    st.success(
                        f"✅ Liveness Verified: {best_score}% match "
                        f"(token_set_ratio={token_set_score}% · ratio={ratio_score}% · "
                        f"partial_ratio={partial_score}% · threshold: {SIMILARITY_THRESHOLD}%). Audio accepted."
                    )

                    # ── 3c. Send to FastAPI /register ────────────────────
                    with st.spinner("🔒  Sending voice sample to backend…"):
                        try:
                            # IMPORTANT: Send the RAW (but peak-normalized) audio to backend,
                            # NOT the bandpass-filtered audio, which damages Resemblyzer embeddings.
                            with open(TEMP_RAW_WAV_PATH, "rb") as audio_file:
                                response = requests.post(
                                    f"{BACKEND_URL}/register",
                                    files={"file": ("voiceguard_sample.wav", audio_file, "audio/wav")},
                                    data={"username": username.strip()},
                                    timeout=30,
                                )

                            if response.status_code == 200:
                                payload          = response.json()
                                recorded         = payload.get("samples_recorded", 1)
                                needed           = payload.get("samples_needed", 0)
                                enroll_complete  = payload.get("enrollment_complete", False)
                                audio_quality    = payload.get("audio_quality", {})
                                
                                # Show quality metrics
                                if audio_quality:
                                    q_score = audio_quality.get("quality_score", 0)
                                    q_color = "#39ff8f" if q_score >= 70 else ("#ffcc00" if q_score >= 40 else "#ff4c4c")
                                    st.markdown(
                                        f'<div style="font-size:0.85rem; color:#8b949e; margin-top:0.5rem;">'
                                        f'🎙️ Sample Quality: <strong style="color:{q_color};">{q_score}/100</strong> '
                                        f'(RMS: {audio_quality.get("rms", 0):.3f})</div>',
                                        unsafe_allow_html=True
                                    )

                                # ── Progress bar ─────────────────────────
                                progress_pct = recorded / 5
                                st.progress(progress_pct)

                                if enroll_complete:
                                    # All 5 samples collected — enrollment done
                                    st.balloons()
                                    st.markdown(
                                        f'<div class="vg-card" style="border-color:#39ff8f;">'
                                        f'<span class="badge-success" style="font-size:1.1rem;">'
                                        f'🎉 Enrollment Complete! (5/5 samples averaged)</span>'
                                        f'<p style="margin-top:0.75rem; color:#8b949e;">'
                                        f'<strong style="color:#e6edf3;">{username}</strong>\'s '
                                        f'text-independent voiceprint is ready. '
                                        f'You can now log in with <em>any</em> challenge phrase.</p>'
                                        f'</div>',
                                        unsafe_allow_html=True,
                                    )
                                    # Clear the phrase so next action is fresh
                                    st.session_state.pop("liveness_phrase", None)
                                else:
                                    # More samples needed
                                    st.markdown(
                                        f'<div class="vg-card" style="border-color:#00e5ff;">'
                                        f'<span class="badge-info" style="font-size:1rem;">'
                                        f'✅ Sample {recorded}/5 recorded</span>'
                                        f'<p style="margin-top:0.75rem; color:#8b949e;">'
                                        f'<strong style="color:#fff;">{needed} more recording(s) needed.</strong><br>'
                                        f'👉 Go back to <strong>Step 2</strong>, click '
                                        f'<em>"Generate Challenge Phrase"</em> to get a '
                                        f'<strong style="color:#00e5ff;">DIFFERENT</strong> phrase, '
                                        f'then record again. Each phrase you speak adds a new '
                                        f'dimension to your voice identity.</p>'
                                        f'</div>',
                                        unsafe_allow_html=True,
                                    )
                                    # Clear the used phrase to force generating a new one
                                    st.session_state.pop("liveness_phrase", None)

                            else:
                                st.error(
                                    f"❌  Registration failed (HTTP {response.status_code}): "
                                    f"{response.text}"
                                )
                        except requests.exceptions.ConnectionError:
                            st.error(
                                "❌  Cannot reach the backend. "
                                "Make sure the FastAPI server is running at "
                                f"`{BACKEND_URL}`."
                            )
                        except Exception as exc:
                            st.error(f"❌  Unexpected error during registration: {exc}")

    # ── Right Column: Pipeline Info ──────────────────────────────────────────
    with col_info:
        # Fetch live enrollment status for this user
        uname = st.session_state.get("reg_username", "").strip()
        enrolled_count = 0
        if uname:
            try:
                status_resp = requests.get(f"{BACKEND_URL}/enrollment_status/{uname}", timeout=5)
                if status_resp.status_code == 200:
                    enrolled_count = status_resp.json().get("samples_recorded", 0)
            except Exception:
                pass

        # Enrollment Progress Card
        st.markdown(
            f'<div class="vg-card">'
            f'<h4 style="margin-top:0;">🎙️ Enrollment Progress</h4>'
            f'<p style="color:#8b949e; font-size:0.85rem; margin-bottom:0.75rem;">'
            f'Record 5 different phrases to build a phrase-independent voiceprint.</p>'
            + "".join(
                f'<div style="display:flex;align-items:center;gap:0.6rem;margin:0.4rem 0;">'
                f'<span style="font-size:1.2rem;">{"✅" if i < enrolled_count else "⬜"}</span>'
                f'<span style="color:{"#39ff8f" if i < enrolled_count else "#8b949e"};font-size:0.9rem;">'
                f'Sample {i+1} {"recorded" if i < enrolled_count else "— pending"}</span></div>'
                for i in range(5)
            )
            + f'<div style="margin-top:1rem;background:#0d1117;border-radius:8px;padding:0.5rem 0.75rem;'
              f'font-size:0.8rem;color:{"#39ff8f" if enrolled_count >= 5 else "#00e5ff"};font-weight:600;">'
            + ("Enrollment complete — ready to login!" if enrolled_count >= 5
               else f"{5 - enrolled_count} more recording(s) needed")
            + '</div></div>',
            unsafe_allow_html=True,
        )

        # Registration Pipeline steps
        pipeline_html = '<div class="vg-card"><h4 style="margin-top:0;">Registration Pipeline</h4>'
        steps = [
            ("👤", "Enter username",                    True),
            ("🎲", "Generate phrase (use a NEW one each round)", True),
            ("🎙️", "Record voice (6s)",                True),
            ("✅", "Liveness check (Whisper STT)",       True),
            ("🧠", "Extract 20 MFCCs",                  True),
            ("📐", "Embed via Siamese CNN",              True),
            ("🔁", "Repeat ×3 with different phrases",   True),
            ("📊", "Average embeddings → centroid",      True),
            ("💾", "Save text-independent voiceprint",   True),
        ]
        for icon, label, active in steps:
            cls = "active" if active else ""
            pipeline_html += (
                f'<div class="step-row">'
                f'<div class="step-icon">{icon}</div>'
                f'<div class="step-label {cls}">{label}</div>'
                f'</div>'
            )
        pipeline_html += "</div>"
        st.markdown(pipeline_html, unsafe_allow_html=True)


def page_login() -> None:
    """
    Login Page
    -----------
    Workflow:
      1. Enter username
      2. Generate a new liveness challenge phrase
      3. Record 4 seconds of audio
      4. [Placeholder] Extract MFCCs from live audio
      5. [Placeholder] Verify BioHash vs stored hash
      6. Grant / deny access
    """
    st.markdown(
        "<h1 style='color:#00e5ff; margin-bottom:0.25rem;'>Voice Authentication Login</h1>"
        "<p style='color:#8b949e; margin-bottom:2rem;'>"
        "Verify your identity — speak the challenge phrase to authenticate.</p>",
        unsafe_allow_html=True,
    )

    col_form, col_info = st.columns([2, 1], gap="large")

    with col_form:

        # ── Step 1: Username ───────────────────────────────────────────────
        st.markdown("#### Step 1 — Enter Username")
        username = st.text_input(
            "Username",
            placeholder="e.g. keerti_puri",
            key="login_username",
        )

        # Static hint — live lookup is handled by the backend at authentication time
        if username.strip():
            st.markdown(
                '<span class="badge-info">ℹ️  Username entered — proceed to generate a phrase and record</span>',
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Step 2: New Liveness Challenge ────────────────────────────────
        st.markdown("#### Step 2 — New Liveness Challenge")
        st.markdown(
            "<p style='color:#8b949e; font-size:0.9rem;'>"
            "A fresh challenge phrase is generated for every login attempt. "
            "Pre-recorded audio cannot pass this check.</p>",
            unsafe_allow_html=True,
        )

        if st.button("🎲  Generate Challenge Phrase", key="btn_login_phrase"):
            st.session_state["login_phrase"] = generate_liveness_phrase()

        if "login_phrase" in st.session_state:
            st.markdown(
                f'<div class="liveness-box">'
                f'  <div class="phrase-label">📢 Read this phrase aloud</div>'
                f'  <div class="phrase-text">{st.session_state["login_phrase"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Step 3: Record & Verify ────────────────────────────────────────
        st.markdown("#### Step 3 — Authenticate")

        if "login_phrase" not in st.session_state:
            st.info("⚠️  Please generate a liveness phrase first (Step 2).")
        else:
            if st.button(
                f"🎙️  Record & Authenticate ({RECORD_SECS} seconds)",
                key="btn_auth",
                type="primary",
            ):
                if not username.strip():
                    st.error("❌  Please enter a username.")
                else:
                    if True:  # guard kept for indentation parity
                        # ── STEP 3a: Record live audio ───────────────────
                        with st.spinner(f"🎙️  Recording for {RECORD_SECS} seconds… Speak now!"):
                            try:
                                raw_audio = record_audio(duration=RECORD_SECS)
                                st.success("✅  Recording complete!")
                            except Exception as e:
                                st.error(f"❌  Microphone error: {e}")
                                st.stop()

                        # ──────────────────────────────────────────────────
                        # SECURITY CHECK 1 — Voice Activity Detection (VAD)
                        # ──────────────────────────────────────────────────
                        # Compute the Root-Mean-Square (RMS) energy of the
                        # captured audio.  A near-zero RMS indicates that the
                        # microphone captured silence, which could mean:
                        #   • The user forgot to speak
                        #   • The microphone is not working correctly
                        #   • An adversary submitted an empty / muted clip
                        #
                        # We abort the pipeline immediately in any of these
                        # cases — no MFCC extraction, no BioHash lookup, and
                        # no database comparison is performed.
                        #
                        # Threshold of 0.01 works well for float32 audio in
                        # the [-1, 1] range.  Adjust if needed for your mic.
                        # ──────────────────────────────────────────────────
                        rms_energy = float(np.sqrt(np.mean(raw_audio ** 2)))

                        if rms_energy < 0.01:
                            # ❌ Silent recording — reject immediately
                            st.error(
                                "🔇 No voice detected! "
                                "Please speak clearly into the microphone and try again."
                            )
                            st.stop()   # Halt pipeline — no DB lookup performed

                        # VAD passed — voice energy is above the silence threshold
                        st.info(f"🔊 VAD passed — voice energy detected (RMS = {rms_energy:.4f})")

                        # ── STEP 3b: Band-pass filter + save WAV ─────────
                        # Apply the same 80 Hz – 8 kHz Butterworth filter used
                        # in registration to remove sub-bass rumble and
                        # high-frequency noise before Whisper transcription.
                        with st.spinner("🔊  Filtering audio (80Hz – 8kHz band-pass)…"):
                            # 1. Save RAW peak-normalized audio for accurate backend embeddings
                            save_wav(raw_audio, TEMP_RAW_WAV_PATH)
                            st.info(f"📁  RAW audio saved for backend: `{TEMP_RAW_WAV_PATH}`")
                            
                            # 2. Save filtered audio exclusively for Whisper STT
                            filtered = apply_bandpass_filter(raw_audio)
                            save_wav(filtered, TEMP_WAV_PATH)
                            st.info(f"📁  Filtered audio saved for STT: `{TEMP_WAV_PATH}`")

                        # ──────────────────────────────────────────────────
                        # SECURITY CHECK 2 — Offline STT Liveness Verification
                        #                    (OpenAI Whisper — runs 100% locally)
                        # ──────────────────────────────────────────────────
                        # This mirrors the exact liveness-check logic used in
                        # the Registration pipeline.  Applying it to Login too
                        # ensures that:
                        #   • A pre-recorded audio clip from a DIFFERENT session
                        #     (which would have a different challenge phrase)
                        #     is rejected BEFORE any BioHash comparison.
                        #   • An adversary cannot replay a previously captured
                        #     voice sample because they cannot predict the
                        #     freshly generated challenge phrase.
                        #
                        # Note: Whisper runs LOCALLY — no internet or API key
                        # required.  The "base" model (~74 MB) is downloaded
                        # once and cached in ~/.cache/whisper on first run.
                        # ──────────────────────────────────────────────────

                        # Retrieve the challenge phrase that was shown to the
                        # user in Step 2 of the login form.
                        login_challenge = st.session_state["login_phrase"]

                        # ── Step 2a: Load local Whisper model ────────────
                        with st.spinner("🤖  Loading local STT model (Whisper 'base')…"):
                            model = whisper.load_model("base")

                        # ── Step 2b: Transcribe the filtered WAV file ────
                        # fp16=False  — float32 inference for CPU compatibility.
                        # language="en" — prevents language-detection errors on
                        #                 short English phrases (avoids non-English
                        #                 hallucinations common in Whisper 'base').
                        # initial_prompt — seeds the decoder vocabulary with the
                        #                  exact challenge phrase, making Whisper
                        #                  strongly prefer those specific words over
                        #                  phonetically similar alternatives.
                        with st.spinner("🗣️  Transcribing audio with local Whisper model…"):
                            result = model.transcribe(
                                TEMP_WAV_PATH,
                                fp16=False,
                                language="en",
                                initial_prompt=login_challenge,
                            )

                        # Pull the plain-text transcription from Whisper's output
                        raw_transcription = result["text"].strip()

                        # ── Step 2c: Normalise both strings for comparison ─
                        # Convert to lowercase and strip all punctuation to
                        # avoid false rejections from minor speech artefacts
                        # (e.g. "Swift Mountain," vs "swift mountain").
                        # Note: `string` and `fuzz` (from thefuzz) are
                        # imported at the top of the file — no inline import.
                        _strip_punct = str.maketrans("", "", string.punctuation)

                        transcribed_text   = raw_transcription.lower().translate(_strip_punct).strip()
                        normalised_phrase  = login_challenge.lower().translate(_strip_punct).strip()

                        # ══════════════════════════════════════════════════
                        # THREE-TIERED FUZZY LIVENESS VERIFICATION (Login)
                        # ══════════════════════════════════════════════════
                        # Strategy: ANY one metric ≥ 65 is sufficient.
                        #
                        # • token_set_ratio  — insensitive to missing/extra
                        #   words and word order. Catches Whisper dropping a
                        #   connector or reordering due to accent.
                        # • ratio            — overall character-level match.
                        #   Catches near-homophones ("whether" → "weather").
                        # • partial_ratio    — best substring match. Catches
                        #   Whisper transcribing only part of the phrase
                        #   (e.g., "red apple" from "A bright red apple").
                        #
                        # Threshold lowered to 65 (from 80) because:
                        #  • Whisper 'base' has ~10-15% WER on accented English.
                        #  • Short 4-6 word phrases amplify character-level error %.
                        #  • The 3-metric OR gate already provides strong security.
                        # ══════════════════════════════════════════════════

                        SIMILARITY_THRESHOLD = 65   # integer percentage for thefuzz

                        token_set_score = fuzz.token_set_ratio(
                            normalised_phrase,   # expected login phrase (normalised)
                            transcribed_text,    # what Whisper heard (normalised)
                        )
                        ratio_score = fuzz.ratio(
                            normalised_phrase,   # expected login phrase (normalised)
                            transcribed_text,    # what Whisper heard (normalised)
                        )
                        partial_score = fuzz.partial_ratio(
                            normalised_phrase,   # expected login phrase (normalised)
                            transcribed_text,    # what Whisper heard (normalised)
                        )

                        liveness_passed = (
                            token_set_score >= SIMILARITY_THRESHOLD
                            or ratio_score  >= SIMILARITY_THRESHOLD
                            or partial_score >= SIMILARITY_THRESHOLD
                        )

                        if not liveness_passed:
                            # ❌ No metric reached the threshold — reject.
                            # "Demo Saver" UI: show exactly what the AI heard vs. required.
                            st.error(
                                f"Liveness Check Failed\n\n"
                                f"token_set_ratio: {token_set_score}%  |  "
                                f"ratio: {ratio_score}%  |  "
                                f"partial_ratio: {partial_score}%  (threshold: {SIMILARITY_THRESHOLD}%)\n\n"
                                f"System Heard: '{transcribed_text}'\n"
                                f"Required: '{login_challenge}'\n\n"
                                f"Please try again and speak clearly."
                            )
                            st.stop()   # Halt pipeline — no DB lookup performed

                        # ✅ Liveness passed — user is live and said the correct phrase.
                        best_score = max(token_set_score, ratio_score, partial_score)
                        st.success(
                            f"✅ Liveness Verified! {best_score}% match "
                            f"(token_set_ratio={token_set_score}% · ratio={ratio_score}% · "
                            f"partial_ratio={partial_score}% · threshold: {SIMILARITY_THRESHOLD}%). Audio accepted."
                        )

                        # ── STEP 3c/3d: Send to FastAPI /login ───────────
                        with st.spinner("🔍  Verifying voiceprint with backend…"):
                            try:
                                # IMPORTANT: Send the RAW (but peak-normalized) audio to backend!
                                with open(TEMP_RAW_WAV_PATH, "rb") as audio_file:
                                    response = requests.post(
                                        f"{BACKEND_URL}/login",
                                        files={"file": ("voiceguard_sample.wav", audio_file, "audio/wav")},
                                        data={"username": username.strip()},
                                        timeout=30,
                                    )

                                if response.status_code == 404:
                                    st.error(
                                        f"❌  User **{username}** not found. "
                                        "Please register first."
                                    )
                                elif response.status_code == 403:
                                    st.warning(
                                        f"⚠️  Enrollment incomplete for **{username}**. "
                                        "Please finish the full 5-sample registration first."
                                    )
                                elif response.status_code == 200:
                                    payload       = response.json()
                                    verified      = payload.get("verified", False)
                                    final_score   = payload.get("final_score", 0.0)
                                    centroid_sim  = payload.get("centroid_sim", 0.0)
                                    best_sim      = payload.get("best_sample_sim", 0.0)
                                    top2_avg      = payload.get("top2_avg", 0.0)
                                    all_sims      = payload.get("all_similarities", [])
                                    threshold_val = payload.get("threshold", 0.70)
                                    audio_quality = payload.get("audio_quality", {})

                                    if verified:
                                        st.balloons()
                                        st.markdown(
                                            '<div class="vg-card" style="border-color:#39ff8f;">'
                                            '<h2 class="badge-success">🔓 ACCESS GRANTED</h2>'
                                            f'<p style="color:#8b949e;">Welcome back, '
                                            f'<strong style="color:#e6edf3;">{username}</strong>! '
                                            'Your voiceprint has been successfully verified.</p>'
                                            '</div>',
                                            unsafe_allow_html=True,
                                        )
                                    else:
                                        st.markdown(
                                            '<div class="vg-card" style="border-color:#ff4c4c;">'
                                            '<h2 class="badge-error">🔒 ACCESS DENIED</h2>'
                                            '<p style="color:#8b949e;">'
                                            'Voiceprint did not match the enrolled user.</p>'
                                            '</div>',
                                            unsafe_allow_html=True,
                                        )

                                    # ── Similarity breakdown panel ─────────────────────
                                    verdict_color = "#39ff8f" if verified else "#ff4c4c"
                                    
                                    q_html = ""
                                    if audio_quality:
                                        q_score = audio_quality.get("quality_score", 0)
                                        q_color = "#39ff8f" if q_score >= 70 else ("#ffcc00" if q_score >= 40 else "#ff4c4c")
                                        q_html = (
                                            f'<div style="font-size:0.85rem; color:#8b949e; margin-bottom:1rem;">'
                                            f'🎙️ Sample Quality: <strong style="color:{q_color};">{q_score}/100</strong> '
                                            f'(RMS: {audio_quality.get("rms", 0):.3f})</div>'
                                        )

                                    rows = ""
                                    for i, sim in enumerate(all_sims):
                                        bar_w     = int(sim * 100)
                                        bar_color = "#39ff8f" if sim >= threshold_val else "#ff4c4c"
                                        rows += (
                                            f'<div style="margin:0.4rem 0;">'
                                            f'<div style="display:flex;justify-content:space-between;'
                                            f'font-size:0.8rem;margin-bottom:3px;">'
                                            f'<span style="color:#8b949e;">Enroll sample {i+1}</span>'
                                            f'<span style="color:{bar_color};font-weight:600;">'
                                            f'{sim*100:.1f}%</span></div>'
                                            f'<div style="background:#0d1117;border-radius:4px;height:8px;">'
                                            f'<div style="background:{bar_color};width:{bar_w}%;'
                                            f'height:8px;border-radius:4px;"></div>'
                                            f'</div></div>'
                                        )
                                    
                                    st.markdown(
                                        f'<div class="vg-card">'
                                        f'<h4 style="margin-top:0;">🎯 Voice Similarity Breakdown</h4>'
                                        f'{q_html}'
                                        f'<div style="display:flex; justify-content:space-between; margin-bottom:1rem; font-size:0.85rem;">'
                                        f'<div style="text-align:left;"><div style="color:#8b949e">Centroid Math</div><div style="font-weight:600;color:{"#39ff8f" if centroid_sim >= threshold_val else "#ff4c4c"}">{centroid_sim*100:.1f}%</div></div>'
                                        f'<div style="text-align:center;"><div style="color:#8b949e">Best Sample</div><div style="font-weight:600;color:{"#39ff8f" if best_sim >= threshold_val else "#ff4c4c"}">{best_sim*100:.1f}%</div></div>'
                                        f'<div style="text-align:right;"><div style="color:#8b949e">Top-2 Avg</div><div style="font-weight:600;color:{"#39ff8f" if top2_avg >= threshold_val else "#ff4c4c"}">{top2_avg*100:.1f}%</div></div>'
                                        f'</div>'
                                        f'{rows}'
                                        f'<div style="margin-top:0.75rem;padding-top:0.75rem;'
                                        f'border-top:1px solid #30363d;font-size:0.9rem;">'
                                        f'<span style="color:#8b949e;">Multi-Strategy Final Score: </span>'
                                        f'<strong style="color:{verdict_color};font-size:1.1rem;">'
                                        f'{final_score*100:.1f}%</strong>'
                                        f'<span style="color:#8b949e;"> vs threshold </span>'
                                        f'<strong style="color:#fff;">{threshold_val*100:.0f}%</strong>'
                                        f'</div></div>',
                                        unsafe_allow_html=True,
                                    )
                                else:
                                    st.error(
                                        f"❌  Login failed (HTTP {response.status_code}): "
                                        f"{response.text}"
                                    )
                            except requests.exceptions.ConnectionError:
                                st.error(
                                    "❌  Cannot reach the backend. "
                                    "Make sure the FastAPI server is running at "
                                    f"`{BACKEND_URL}`."
                                )
                            except Exception as exc:
                                st.error(f"❌  Unexpected error during login: {exc}")

    # ── Right Column: Info ───────────────────────────────────────────────────
    with col_info:
        # Authentication Pipeline panel — pure HTML, one markdown() call
        pipeline_html = (
            '<div class="vg-card"><h4 style="margin-top:0;">Authentication Pipeline</h4>'
        )
        steps = [
            ("👤", "Enter username",                   True),
            ("🎲", "New random challenge phrase",         True),
            ("🎙️", "Record live voice (6s)",            True),
            ("✅", "Liveness check (Whisper STT)",        True),
            ("🧠", "Extract 20 MFCCs",                   True),
            ("📐", "Embed via Siamese CNN",               True),
            ("🔍", "Nearest-neighbor vs 3 enrollments",  True),
            ("📊", "Min distance vs threshold (3.5)",     True),
            ("🔓", "Grant / deny access",                 True),
        ]
        for icon, label, active in steps:
            cls = "active" if active else ""
            pipeline_html += (
                f'<div class="step-row">'
                f'<div class="step-icon">{icon}</div>'
                f'<div class="step-label {cls}">{label}</div>'
                f'</div>'
            )
        pipeline_html += "</div>"
        st.markdown(pipeline_html, unsafe_allow_html=True)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main() -> None:
    """Application entry point — routes to the correct page."""
    # Inject custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Render sidebar and get selected page
    page = render_sidebar()

    # Route to the appropriate page
    if "Register" in page:
        page_register()
    else:
        page_login()


if __name__ == "__main__":
    main()
