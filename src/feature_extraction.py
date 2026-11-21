"""
src/feature_extraction.py
Stage 3 — Feature Extraction Pipeline

Extracts the following features from each preprocessed audio segment:
  - Mel Spectrogram (primary input to CNN)
  - MFCC (40 coefficients)
  - Chroma Features
  - Zero Crossing Rate
  - Spectral Centroid
  - Spectral Rolloff

Features are saved as compressed .npz files for fast loading during training.
Metadata CSV is also saved for DataLoader reference.

Usage:
  python src/feature_extraction.py
"""

import os
import logging
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
SEED       = 42
SR_TARGET  = 22050
N_FFT      = 1024
HOP_LENGTH = 256
N_MELS     = 64         # Mel spectrogram bins
N_MFCC     = 40         # Number of MFCC coefficients
N_CHROMA   = 12         # Chroma bins
FIXED_FRAMES = 128      # Fixed time axis length (pad/trim all features to this)
                        # At 22050 Hz with hop=256: ~1.49 seconds per 128 frames
                        # Most cycles are 1-5s; we'll pad shorter and trim longer

CLASS_NAMES = ["Normal", "Crackle", "Wheeze", "Both"]

PROJECT_ROOT    = r"c:\Users\User\Desktop\NLP PROJECT"
PROCESSED_DIR   = os.path.join(PROJECT_ROOT, "dataset", "processed", "segments")
FEATURES_DIR    = os.path.join(PROJECT_ROOT, "dataset", "processed", "features")
REPORTS_DIR     = os.path.join(PROJECT_ROOT, "results", "reports")

np.random.seed(SEED)


# ── Feature Extraction Helpers ───────────────────────────────────────────────

def fix_length(arr: np.ndarray, target_frames: int = FIXED_FRAMES) -> np.ndarray:
    """
    Pad or trim a 2D feature array (shape: n_features × time_frames)
    to have exactly `target_frames` on the time axis.
    Uses reflection padding to avoid introducing silence artifacts.
    """
    n_feat, n_frames = arr.shape
    if n_frames < target_frames:
        pad_width = target_frames - n_frames
        arr = np.pad(arr, ((0, 0), (0, pad_width)), mode="reflect")
    else:
        arr = arr[:, :target_frames]
    return arr


def extract_mel_spectrogram(y: np.ndarray, sr: int = SR_TARGET) -> np.ndarray:
    """
    Extract log-scaled Mel Spectrogram.
    Shape: (N_MELS, FIXED_FRAMES) → (64, 128)
    """
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=20, fmax=sr // 2
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # Normalize to [0, 1]
    mel_min, mel_max = mel_db.min(), mel_db.max()
    if mel_max > mel_min:
        mel_db = (mel_db - mel_min) / (mel_max - mel_min)
    return fix_length(mel_db.astype(np.float32))


def extract_mfcc(y: np.ndarray, sr: int = SR_TARGET) -> np.ndarray:
    """
    Extract MFCC features (40 coefficients).
    Shape: (N_MFCC, FIXED_FRAMES) → (40, 128)
    Includes delta and delta-delta for temporal dynamics.
    """
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC,
        n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    # Delta features capture rate of change
    mfcc_delta  = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    # Stack: (N_MFCC×3, time) → (120, time)
    mfcc_full = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
    # Standardize per coefficient
    mean = mfcc_full.mean(axis=1, keepdims=True)
    std  = mfcc_full.std(axis=1, keepdims=True) + 1e-8
    mfcc_norm = (mfcc_full - mean) / std
    return fix_length(mfcc_norm.astype(np.float32))


def extract_chroma(y: np.ndarray, sr: int = SR_TARGET) -> np.ndarray:
    """
    Extract Chroma features (12 pitch classes).
    Shape: (N_CHROMA, FIXED_FRAMES) → (12, 128)
    """
    chroma = librosa.feature.chroma_stft(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_chroma=N_CHROMA
    )
    return fix_length(chroma.astype(np.float32))


def extract_spectral_features(y: np.ndarray, sr: int = SR_TARGET) -> np.ndarray:
    """
    Extract scalar spectral features over time:
      - Spectral Centroid  (1 × time)
      - Spectral Rolloff   (1 × time)
      - Zero Crossing Rate (1 × time)
      - RMS Energy         (1 × time)
    Returns stacked array: (4, FIXED_FRAMES)
    """
    centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    rolloff  = librosa.feature.spectral_rolloff(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    zcr      = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)
    rms      = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)

    # Normalize each to [0,1]
    def norm(x):
        xmin, xmax = x.min(), x.max()
        return (x - xmin) / (xmax - xmin + 1e-8)

    stacked = np.vstack([norm(centroid), norm(rolloff), norm(zcr), norm(rms)])
    return fix_length(stacked.astype(np.float32))


def extract_all_features(y: np.ndarray, sr: int = SR_TARGET) -> dict[str, np.ndarray]:
    """
    Extract all features from one audio segment.

    Returns a dict with keys:
        mel    → (64,  128) float32   ← PRIMARY INPUT to CNN
        mfcc   → (120, 128) float32   ← Supplementary
        chroma → (12,  128) float32   ← Supplementary
        spec   → (4,   128) float32   ← Supplementary
    """
    return {
        "mel"    : extract_mel_spectrogram(y, sr),
        "mfcc"   : extract_mfcc(y, sr),
        "chroma" : extract_chroma(y, sr),
        "spec"   : extract_spectral_features(y, sr),
    }


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def run_feature_extraction(metadata_csv: str | None = None) -> pd.DataFrame:
    """
    Extract features for all processed audio segments.

    If metadata_csv is None, scans PROCESSED_DIR directly.
    Features are saved as .npz files in FEATURES_DIR/<ClassName>/
    Returns a DataFrame with paths and labels for all extracted features.
    """
    os.makedirs(FEATURES_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR,  exist_ok=True)

    # ── 1. Find all segments ──────────────────────────────────────────────────
    if metadata_csv and os.path.exists(metadata_csv):
        df = pd.read_csv(metadata_csv)
        wav_paths  = df["processed_path"].tolist()
        labels     = df["label"].tolist()
        label_names = df["label_name"].tolist()
        patient_ids = df["patient_id"].tolist()
        log.info(f"Loaded {len(df)} segments from metadata CSV.")
    else:
        # Scan directory structure
        wav_paths, labels, label_names, patient_ids = [], [], [], []
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            cls_dir = os.path.join(PROCESSED_DIR, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if not fname.endswith(".wav"):
                    continue
                wav_paths.append(os.path.join(cls_dir, fname))
                labels.append(cls_idx)
                label_names.append(cls_name)
                # Extract patient ID from filename: patientID_...
                pid = int(fname.split("_")[0])
                patient_ids.append(pid)
        log.info(f"Scanned {len(wav_paths)} segments from processed directory.")

    if len(wav_paths) == 0:
        log.error("No audio segments found! Run preprocessing.py first.")
        return pd.DataFrame()

    # ── 2. Extract features ───────────────────────────────────────────────────
    records  = []
    errors   = 0

    for i, (wav_path, label, label_name, pid) in enumerate(
            tqdm(zip(wav_paths, labels, label_names, patient_ids),
                 total=len(wav_paths), desc="Extracting features", unit="file")):

        try:
            # Load segment (already preprocessed & normalized)
            y, sr = librosa.load(wav_path, sr=None, mono=True)
            if sr != SR_TARGET:
                y = librosa.resample(y, orig_sr=sr, target_sr=SR_TARGET)

            # Extract all features
            feats = extract_all_features(y, SR_TARGET)

            # Save .npz
            base_name = os.path.splitext(os.path.basename(wav_path))[0]
            cls_feat_dir = os.path.join(FEATURES_DIR, label_name)
            os.makedirs(cls_feat_dir, exist_ok=True)
            npz_path = os.path.join(cls_feat_dir, base_name + ".npz")

            np.savez_compressed(npz_path, **feats, label=label)

            records.append({
                "patient_id"   : pid,
                "wav_path"     : wav_path,
                "feature_path" : npz_path,
                "label"        : label,
                "label_name"   : label_name,
                "mel_shape"    : str(feats["mel"].shape),
                "mfcc_shape"   : str(feats["mfcc"].shape),
            })

        except Exception as e:
            log.warning(f"[{i}] Error on {wav_path}: {e}")
            errors += 1
            continue

    # ── 3. Summary ────────────────────────────────────────────────────────────
    result_df = pd.DataFrame(records)

    log.info("=" * 55)
    log.info("FEATURE EXTRACTION COMPLETE")
    log.info(f"  Total .npz files saved : {len(result_df):,}")
    log.info(f"  Errors                 : {errors}")
    log.info("  Feature shapes (per segment):")
    log.info(f"    mel    : (64,  {FIXED_FRAMES}) → CNN primary input")
    log.info(f"    mfcc   : (120, {FIXED_FRAMES}) → with deltas")
    log.info(f"    chroma : (12,  {FIXED_FRAMES})")
    log.info(f"    spec   : (4,   {FIXED_FRAMES})")
    log.info("")
    log.info("  Class breakdown:")
    for cls_name in CLASS_NAMES:
        n = (result_df["label_name"] == cls_name).sum()
        log.info(f"    {cls_name:<8} : {n:>5}")
    log.info("=" * 55)

    # ── 4. Save feature metadata CSV ─────────────────────────────────────────
    meta_path = os.path.join(REPORTS_DIR, "feature_metadata.csv")
    result_df.to_csv(meta_path, index=False)
    log.info(f"Feature metadata saved → {meta_path}")

    return result_df


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    meta_csv = os.path.join(REPORTS_DIR, "processed_metadata.csv")
    df = run_feature_extraction(metadata_csv=meta_csv)
    print(f"\nDone. {len(df):,} feature files saved to: {FEATURES_DIR}")
