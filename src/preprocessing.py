"""
src/preprocessing.py
Stage 3 — Audio Preprocessing Pipeline

Responsibilities:
  - Load WAV files and their ICBHI annotations
  - Resample to a uniform 22,050 Hz
  - Convert to mono
  - Slice recordings into individual respiratory cycle segments
  - Normalize amplitude
  - Apply lightweight noise reduction (spectral subtraction)
  - Save processed segments to dataset/processed/

Usage:
  python src/preprocessing.py
"""

import os
import glob
import logging
import numpy as np
import pandas as pd
import soundfile as sf
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
SEED        = 42
SR_TARGET   = 22050          # uniform sampling rate (Hz)
MIN_DURATION = 0.5           # ignore cycles shorter than this (seconds)
MAX_DURATION = 10.0          # cap cycles longer than this (seconds)
SILENCE_TOP_DB = 40          # threshold for silence trimming

DATASET_ROOT = r"C:\Users\User\Desktop\respiratory  -audio- datasets"
AUDIO_DIR    = os.path.join(DATASET_ROOT,
                            "Respiratory_Sound_Database",
                            "Respiratory_Sound_Database",
                            "audio_and_txt_files")
DIAGNOSIS_CSV = os.path.join(DATASET_ROOT,
                             "Respiratory_Sound_Database",
                             "Respiratory_Sound_Database",
                             "patient_diagnosis.csv")

PROJECT_ROOT   = r"c:\Users\User\Desktop\NLP PROJECT"
PROCESSED_DIR  = os.path.join(PROJECT_ROOT, "dataset", "processed", "segments")
REPORTS_DIR    = os.path.join(PROJECT_ROOT, "results", "reports")

CLASS_MAP   = {(0, 0): 0, (1, 0): 1, (0, 1): 2, (1, 1): 3}
CLASS_NAMES = ["Normal", "Crackle", "Wheeze", "Both"]

np.random.seed(SEED)


# ── Helper Functions ─────────────────────────────────────────────────────────

def load_audio(wav_path: str, sr_target: int = SR_TARGET) -> tuple[np.ndarray, int]:
    """
    Load a WAV file, convert to mono, and resample to sr_target.
    Returns (audio_array, sample_rate).
    """
    y, sr_orig = librosa.load(wav_path, sr=None, mono=True)
    if sr_orig != sr_target:
        y = librosa.resample(y, orig_sr=sr_orig, target_sr=sr_target)
    return y, sr_target


def normalize_audio(y: np.ndarray) -> np.ndarray:
    """Peak-normalize audio to [-1, 1]."""
    peak = np.abs(y).max()
    if peak > 0:
        y = y / peak
    return y


def reduce_noise_spectral(y: np.ndarray, sr: int,
                           n_fft: int = 1024, noise_factor: float = 0.02) -> np.ndarray:
    """
    Lightweight spectral subtraction noise reduction.
    Estimates noise floor from the first 0.1s and subtracts it.
    Avoids over-filtering which would destroy respiratory sound patterns.
    """
    # Noise estimate from initial silence
    noise_samples = int(0.1 * sr)
    if len(y) < noise_samples * 2:
        return y  # too short to estimate noise, skip

    noise_profile = y[:noise_samples]
    stft_full  = librosa.stft(y, n_fft=n_fft)
    stft_noise = librosa.stft(noise_profile, n_fft=n_fft)

    noise_mag  = np.mean(np.abs(stft_noise), axis=1, keepdims=True)
    signal_mag = np.abs(stft_full)
    phase      = np.angle(stft_full)

    # Spectral subtraction with flooring to prevent musical noise
    clean_mag = np.maximum(signal_mag - noise_factor * noise_mag, 0.01 * signal_mag)
    stft_clean = clean_mag * np.exp(1j * phase)
    y_clean    = librosa.istft(stft_clean, length=len(y))
    return y_clean.astype(np.float32)


def parse_annotation(txt_path: str) -> list[dict]:
    """
    Parse one ICBHI annotation .txt file.
    Returns a list of cycle dicts with start, end, crackle, wheeze keys.
    """
    cycles = []
    fname  = os.path.basename(txt_path).replace(".txt", "")
    parts  = fname.split("_")
    patient_id = int(parts[0])
    location   = parts[2] if len(parts) > 2 else "Unknown"
    equipment  = parts[4] if len(parts) > 4 else "Unknown"

    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cols = line.split("\t")
            if len(cols) < 4:
                continue
            start, end   = float(cols[0]), float(cols[1])
            crackle, wheeze = int(cols[2]), int(cols[3])
            duration = end - start

            if duration < MIN_DURATION:
                continue

            label_idx = CLASS_MAP.get((crackle, wheeze), 0)
            cycles.append({
                "filename"   : fname,
                "patient_id" : patient_id,
                "location"   : location,
                "equipment"  : equipment,
                "start"      : start,
                "end"        : min(end, start + MAX_DURATION),
                "duration"   : min(duration, MAX_DURATION),
                "crackle"    : crackle,
                "wheeze"     : wheeze,
                "label"      : label_idx,
                "label_name" : CLASS_NAMES[label_idx],
                "wav_path"   : os.path.join(AUDIO_DIR, fname + ".wav"),
            })
    return cycles


def slice_and_save_cycle(row: dict, y_full: np.ndarray,
                          sr: int, out_dir: str) -> str | None:
    """
    Extract one respiratory cycle from a full recording and save it.
    Returns the saved file path, or None if skipped.
    """
    start_samp = int(row["start"] * sr)
    end_samp   = int(row["end"]   * sr)
    cycle      = y_full[start_samp:end_samp]

    if len(cycle) < int(MIN_DURATION * sr):
        return None

    # Normalize
    cycle = normalize_audio(cycle)

    # Build output filename: patientID_filename_startMs_label.wav
    start_ms  = int(row["start"] * 1000)
    out_name  = f"{row['patient_id']}_{row['filename']}_{start_ms}_{row['label']}.wav"
    class_dir = os.path.join(out_dir, row["label_name"])
    os.makedirs(class_dir, exist_ok=True)
    out_path  = os.path.join(class_dir, out_name)

    sf.write(out_path, cycle, sr)
    return out_path


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def run_preprocessing(apply_noise_reduction: bool = True,
                       force_reprocess: bool = False) -> pd.DataFrame:
    """
    Full preprocessing pipeline.

    Args:
        apply_noise_reduction: Whether to apply spectral subtraction.
        force_reprocess: If True, overwrite already-processed files.

    Returns:
        DataFrame with metadata for all processed segments.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR,   exist_ok=True)

    # ── 1. Collect all annotation files ───────────────────────────────────────
    txt_files = sorted(glob.glob(os.path.join(AUDIO_DIR, "*.txt")))
    log.info(f"Found {len(txt_files)} annotation files.")

    # ── 2. Parse all annotations ──────────────────────────────────────────────
    all_cycles = []
    for txt_path in txt_files:
        all_cycles.extend(parse_annotation(txt_path))

    log.info(f"Parsed {len(all_cycles)} respiratory cycles "
             f"(after filtering < {MIN_DURATION}s cycles).")

    # ── 3. Group cycles by WAV file (avoid loading the same file multiple times)
    cycles_df  = pd.DataFrame(all_cycles)
    wav_groups = cycles_df.groupby("wav_path")
    log.info(f"Processing {len(wav_groups)} unique WAV files...")

    saved_records = []
    skipped       = 0
    errors        = 0

    for wav_path, group in tqdm(wav_groups, desc="Processing WAV files", unit="file"):
        try:
            # Load once per file
            y, sr = load_audio(wav_path)

            # Optional noise reduction (applied to the full recording first)
            if apply_noise_reduction:
                y = reduce_noise_spectral(y, sr)

            # Slice each cycle from this recording
            for _, row in group.iterrows():
                out_path = slice_and_save_cycle(row, y, sr, PROCESSED_DIR)
                if out_path is None:
                    skipped += 1
                    continue

                record = row.to_dict()
                record["processed_path"] = out_path
                saved_records.append(record)

        except Exception as e:
            log.warning(f"Error processing {wav_path}: {e}")
            errors += 1
            continue

    # ── 4. Build processed metadata DataFrame ─────────────────────────────────
    processed_df = pd.DataFrame(saved_records)

    log.info("=" * 55)
    log.info("PREPROCESSING COMPLETE")
    log.info(f"  Total cycles saved  : {len(processed_df):,}")
    log.info(f"  Skipped (too short) : {skipped}")
    log.info(f"  Errors              : {errors}")
    log.info("")
    log.info("  Class breakdown:")
    for cls_name in CLASS_NAMES:
        n = (processed_df["label_name"] == cls_name).sum()
        log.info(f"    {cls_name:<8} : {n:>5}")
    log.info("=" * 55)

    # ── 5. Save metadata CSV ──────────────────────────────────────────────────
    meta_path = os.path.join(REPORTS_DIR, "processed_metadata.csv")
    processed_df.to_csv(meta_path, index=False)
    log.info(f"Metadata saved → {meta_path}")

    return processed_df


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = run_preprocessing(apply_noise_reduction=True, force_reprocess=False)
    print(f"\nDone. {len(df):,} segments saved to: {PROCESSED_DIR}")
