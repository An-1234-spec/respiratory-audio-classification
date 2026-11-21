"""
src/predict.py
Stage 7 — Single-File Inference

Loads any WAV file, preprocesses it, extracts features,
runs the trained CNN-LSTM model, and returns class probabilities.

Used by both the CLI and the Streamlit frontend (app/streamlit_app.py).

Usage (CLI):
  python src/predict.py path/to/audio.wav
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import librosa

sys.path.insert(0, os.path.dirname(__file__))
from model import build_model, CLASS_NAMES
from feature_extraction import (
    extract_mel_spectrogram,
    SR_TARGET,
    FIXED_FRAMES,
)
from preprocessing import normalize_audio, reduce_noise_spectral

PROJECT_ROOT    = r"c:\Users\User\Desktop\NLP PROJECT"
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.pth")

CLASS_COLORS = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12"]


def load_model(checkpoint_path: str = CHECKPOINT_PATH,
               device: str | None = None):
    """Load trained model from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run src/train.py first to train the model."
        )
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, device = build_model(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, device


def preprocess_audio(audio_path: str,
                     sr_target: int = SR_TARGET,
                     apply_noise_reduction: bool = True) -> np.ndarray:
    """
    Load, preprocess and extract Mel Spectrogram from an audio file.
    Returns feature tensor of shape (1, 1, 64, 128).
    """
    # Load & resample
    y, sr_orig = librosa.load(audio_path, sr=None, mono=True)
    if sr_orig != sr_target:
        y = librosa.resample(y, orig_sr=sr_orig, target_sr=sr_target)

    # Normalize
    y = normalize_audio(y)

    # Light noise reduction
    if apply_noise_reduction:
        y = reduce_noise_spectral(y, sr_target)

    # Extract Mel Spectrogram
    mel = extract_mel_spectrogram(y, sr_target)   # (64, 128)
    mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    # → (1, 1, 64, 128)
    return mel_tensor


@torch.no_grad()
def predict(audio_path: str,
            model=None,
            device: str = "cpu",
            apply_noise_reduction: bool = True) -> dict:
    """
    Full inference pipeline for a single audio file.

    Args:
        audio_path           : Path to input WAV file
        model                : Pre-loaded model (loads from checkpoint if None)
        device               : 'cuda' or 'cpu'
        apply_noise_reduction: Whether to apply spectral subtraction

    Returns:
        dict with keys:
            predicted_class : str  — most likely class name
            predicted_idx   : int  — class index
            probabilities   : dict — {class_name: probability} for all 4 classes
            confidence      : float — max probability
    """
    if model is None:
        model, device = load_model()

    # Preprocess
    x = preprocess_audio(audio_path, apply_noise_reduction=apply_noise_reduction)
    x = x.to(device)

    # Inference
    logits = model(x)                                # (1, 4)
    probs  = F.softmax(logits, dim=-1).cpu().numpy()[0]  # (4,)

    pred_idx  = int(np.argmax(probs))
    pred_name = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    return {
        "predicted_class": pred_name,
        "predicted_idx"  : pred_idx,
        "confidence"     : confidence,
        "probabilities"  : {
            name: float(p)
            for name, p in zip(CLASS_NAMES, probs)
        },
    }


def predict_from_array(y: np.ndarray,
                        sr: int,
                        model=None,
                        device: str = "cpu") -> dict:
    """
    Predict from a raw numpy audio array (e.g., from microphone recording).
    Useful for the Streamlit live recording feature.

    Args:
        y     : Audio samples as float32 array
        sr    : Sample rate of y
        model : Pre-loaded model
        device: Device string

    Returns:
        Same dict format as predict()
    """
    if model is None:
        model, device = load_model()

    # Resample if needed
    if sr != SR_TARGET:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR_TARGET)

    y = normalize_audio(y.astype(np.float32))

    mel = extract_mel_spectrogram(y, SR_TARGET)
    x   = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    logits = model(x)
    probs  = F.softmax(logits, dim=-1).cpu().numpy()[0]
    pred_idx  = int(np.argmax(probs))

    return {
        "predicted_class": CLASS_NAMES[pred_idx],
        "predicted_idx"  : pred_idx,
        "confidence"     : float(probs[pred_idx]),
        "probabilities"  : {n: float(p) for n, p in zip(CLASS_NAMES, probs)},
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <path_to_audio.wav>")
        sys.exit(1)

    audio_path = sys.argv[1]
    print(f"\nAnalyzing: {audio_path}")

    model, device = load_model()
    result = predict(audio_path, model=model, device=device)

    print("\n" + "=" * 45)
    print("  RESPIRATORY SOUND CLASSIFICATION RESULT")
    print("=" * 45)
    print(f"  Predicted Class : {result['predicted_class']}")
    print(f"  Confidence      : {result['confidence']*100:.1f}%")
    print("\n  Class Probabilities:")
    for cls_name, prob in result["probabilities"].items():
        bar = "█" * int(prob * 30)
        print(f"  {cls_name:<8} : {prob*100:5.1f}%  {bar}")
    print("=" * 45)
    
