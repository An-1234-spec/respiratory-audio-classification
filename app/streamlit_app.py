"""
app/streamlit_app.py
Streamlined, Modern UI for Respiratory Sound Classification
Upload an audio file and get an instant prediction.
"""

import os
import sys
import tempfile
import numpy as np
import torch
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

# ── Project paths ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from predict import load_model, predict
from preprocessing import normalize_audio
from feature_extraction import SR_TARGET

PROJECT_ROOT    = os.path.join(os.path.dirname(__file__), "..")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.pth")
SEGMENTS_DIR    = os.path.join(PROJECT_ROOT, "dataset", "processed", "segments")

CLASS_NAMES  = ["Normal", "Crackle", "Wheeze", "Both"]
CLASS_COLORS = {
    "Normal" : "#10b981",  # emerald
    "Crackle": "#ef4444",  # coral red
    "Wheeze" : "#0284c7",  # sky blue
    "Both"   : "#f59e0b",  # amber
}
CLASS_ICONS = {
    "Normal" : "🫁",
    "Crackle": "⚡",
    "Wheeze" : "🌬️",
    "Both"   : "🔄",
}
CLASS_DESC = {
    "Normal" : "Clean, unobstructed respiratory cycle with no adventitious sounds.",
    "Crackle": "Discontinuous popping or bubbling acoustic signatures detected.",
    "Wheeze" : "Continuous, high-pitched musical or whistling acoustic pattern.",
    "Both"   : "Simultaneous combination of continuous wheezing and crackling.",
}


# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Respiratory Sound Classifier",
    page_icon="🫁",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Sleek Modern Styling ──────────────────────────────────────────────────────
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 780px;
    }

    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #0284c7 0%, #10b981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .hero-sub {
        font-size: 1.05rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 1.8rem;
    }

    .result-container {
        border-radius: 16px;
        padding: 24px;
        margin-top: 1.2rem;
        margin-bottom: 1.2rem;
        background: #ffffff;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.06), 0 8px 10px -6px rgba(0, 0, 0, 0.03);
        border: 1px solid #e2e8f0;
    }
    .result-badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.6rem;
    }
    .result-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0;
        line-height: 1.2;
    }
    .result-desc {
        color: #475569;
        font-size: 0.98rem;
        margin-top: 0.5rem;
        line-height: 1.5;
    }
    .confidence-tag {
        font-size: 1.2rem;
        font-weight: 700;
        color: #0f172a;
    }

    .prob-bar-bg {
        background-color: #f1f5f9;
        border-radius: 8px;
        height: 10px;
        overflow: hidden;
        margin-top: 6px;
        margin-bottom: 14px;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 8px;
    }

    .sample-section {
        background: #f8fafc;
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 1.2rem;
        border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)


# ── Load Model (Cached) ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_model():
    try:
        model, device = load_model(CHECKPOINT_PATH)
        return model, device, None
    except Exception as e:
        return None, None, str(e)


# ── Visualization Helpers ─────────────────────────────────────────────────────
def plot_waveform(y: np.ndarray, sr: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 2))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#f8fafc')
    times = np.linspace(0, len(y) / sr, len(y))
    ax.plot(times, y, color="#0284c7", linewidth=0.7, alpha=0.9)
    ax.axhline(0, color="#cbd5e1", linewidth=0.5, linestyle="--")
    ax.set_xlim(0, len(y) / sr)
    ax.set_ylabel("Amplitude", fontsize=9, color="#64748b")
    ax.set_xlabel("Time (s)", fontsize=9, color="#64748b")
    ax.tick_params(colors="#64748b", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#e2e8f0")
    fig.tight_layout()
    return fig


def plot_spectrogram(y: np.ndarray, sr: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 2.5))
    fig.patch.set_facecolor('#ffffff')
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, n_fft=1024, hop_length=256)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    img = librosa.display.specshow(mel_db, sr=sr, hop_length=256,
                                   x_axis="time", y_axis="mel", ax=ax, cmap="magma")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_xlabel("Time (s)", fontsize=9, color="#64748b")
    ax.set_ylabel("Mel Freq", fontsize=9, color="#64748b")
    ax.tick_params(colors="#64748b", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#e2e8f0")
    fig.tight_layout()
    return fig


# ── App Header ────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🫁 Respiratory Sound Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Upload an audio recording to instantly classify respiratory sound patterns.</div>', unsafe_allow_html=True)

# ── Load Model ────────────────────────────────────────────────────────────────
model, device, error_msg = get_model()
if error_msg:
    st.error(f"Model checkpoint not found. Please train first: {error_msg}")
    st.stop()


# ── Session State for Samples & Uploads ────────────────────────────────────────
if "active_sample" not in st.session_state:
    st.session_state.active_sample = None
if "active_label" not in st.session_state:
    st.session_state.active_label = None


def find_sample(cls_name):
    cls_folder = os.path.join(SEGMENTS_DIR, cls_name)
    if os.path.isdir(cls_folder):
        files = [f for f in os.listdir(cls_folder) if f.endswith(".wav")]
        if files:
            return os.path.join(cls_folder, files[0])
    return None


# ── Quick Test Pills ──────────────────────────────────────────────────────────
st.markdown('<div class="sample-section"><small style="color:#64748b; font-weight:700; letter-spacing:0.04em;">⚡ TEST WITH PRELOADED SAMPLES</small>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
with c1:
    if st.button("🫁 Normal", use_container_width=True):
        st.session_state.active_sample = find_sample("Normal")
        st.session_state.active_label = "Sample: Normal Breathing"
with c2:
    if st.button("⚡ Crackle", use_container_width=True):
        st.session_state.active_sample = find_sample("Crackle")
        st.session_state.active_label = "Sample: Crackle"
with c3:
    if st.button("🌬️ Wheeze", use_container_width=True):
        st.session_state.active_sample = find_sample("Wheeze")
        st.session_state.active_label = "Sample: Wheeze"
with c4:
    if st.button("🔄 Both", use_container_width=True):
        st.session_state.active_sample = find_sample("Both")
        st.session_state.active_label = "Sample: Combined (Both)"
st.markdown('</div>', unsafe_allow_html=True)


# ── Upload Audio File ─────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload Audio", type=["wav", "mp3", "ogg"], label_visibility="collapsed")
if uploaded is not None:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(uploaded.read())
        st.session_state.active_sample = tmp.name
        st.session_state.active_label = uploaded.name


# ── Run Prediction & Display Result ───────────────────────────────────────────
if st.session_state.active_sample and os.path.exists(st.session_state.active_sample):
    audio_path = st.session_state.active_sample

    st.markdown("---")
    st.caption(f"🎧 Selected: **{st.session_state.active_label}**")
    st.audio(audio_path, format="audio/wav")

    # Run Prediction Automatically
    with st.spinner("Analyzing audio..."):
        result = predict(audio_path, model=model, device=device, apply_noise_reduction=True)

    pred_cls   = result["predicted_class"]
    confidence = result["confidence"]
    probs      = result["probabilities"]
    theme_col  = CLASS_COLORS.get(pred_cls, "#0284c7")
    icon       = CLASS_ICONS.get(pred_cls, "🫁")

    # Result Card
    st.markdown(f"""
    <div class="result-container" style="border-left: 6px solid {theme_col};">
        <div class="result-badge" style="background-color: {theme_col}20; color: {theme_col};">
            Detected Pattern
        </div>
        <div style="display:flex; justify-content:space-between; align-items:baseline;">
            <div class="result-title">{icon} {pred_cls}</div>
            <div class="confidence-tag">{confidence*100:.1f}% Confidence</div>
        </div>
        <div class="result-desc">{CLASS_DESC[pred_cls]}</div>
    </div>
    """, unsafe_allow_html=True)

    # Class Probability Breakdown
    st.markdown("##### Probabilities")
    for cls_name in CLASS_NAMES:
        prob = probs[cls_name]
        col_bar = CLASS_COLORS[cls_name]
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; font-size: 0.9rem; font-weight: 600; color: #334155;">
            <span>{CLASS_ICONS[cls_name]} {cls_name}</span>
            <span>{prob * 100:.1f}%</span>
        </div>
        <div class="prob-bar-bg">
            <div class="prob-bar-fill" style="width: {prob * 100}%; background-color: {col_bar};"></div>
        </div>
        """, unsafe_allow_html=True)

    # Audio Waveform & Spectrogram
    with st.expander("📊 View Audio Waveform & Spectrogram", expanded=False):
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        if sr != SR_TARGET:
            y = librosa.resample(y, orig_sr=sr, target_sr=SR_TARGET)
        y = normalize_audio(y)

        st.markdown("**Waveform**")
        st.pyplot(plot_waveform(y, SR_TARGET))

        st.markdown("**Log-Mel Spectrogram**")
        st.pyplot(plot_spectrogram(y, SR_TARGET))
