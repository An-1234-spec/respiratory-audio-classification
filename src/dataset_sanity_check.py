"""
Stage 2 Sanity Check Script
Run this to verify: imports, dataset path, annotation parsing, cycle loading.
Usage: python src/dataset_sanity_check.py
"""

import os, sys, glob
import numpy as np
import pandas as pd
import librosa

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_ROOT = r"C:\Users\User\Desktop\respiratory  -audio- datasets"
AUDIO_DIR    = os.path.join(DATASET_ROOT,
                            "Respiratory_Sound_Database",
                            "Respiratory_Sound_Database",
                            "audio_and_txt_files")
DIAGNOSIS_CSV = os.path.join(DATASET_ROOT,
                             "Respiratory_Sound_Database",
                             "Respiratory_Sound_Database",
                             "patient_diagnosis.csv")

CLASS_MAP   = {(0,0):0, (1,0):1, (0,1):2, (1,1):3}
CLASS_NAMES = ["Normal", "Crackle", "Wheeze", "Both"]
SR_TARGET   = 22050

def check(msg, condition, fatal=True):
    icon = "✅" if condition else "❌"
    print(f"  {icon}  {msg}")
    if not condition and fatal:
        sys.exit(1)

print("\n" + "="*55)
print("  STAGE 2 — DATASET SANITY CHECK")
print("="*55)

# 1. Paths exist
check("Dataset root exists",  os.path.isdir(DATASET_ROOT))
check("Audio directory exists",os.path.isdir(AUDIO_DIR))
check("Diagnosis CSV exists",  os.path.isfile(DIAGNOSIS_CSV))

# 2. File counts
wavs = glob.glob(os.path.join(AUDIO_DIR, "*.wav"))
txts = glob.glob(os.path.join(AUDIO_DIR, "*.txt"))
check(f"920 WAV files found  (found {len(wavs)})",  len(wavs) == 920)
check(f"920 TXT files found  (found {len(txts)})",  len(txts) == 920)

# 3. Parse annotations
records = []
for txt_path in sorted(txts):
    fname  = os.path.basename(txt_path).replace(".txt","")
    parts  = fname.split("_")
    pid    = int(parts[0])
    equip  = parts[4] if len(parts)>4 else "Unknown"
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            cols = line.split("\t")
            if len(cols) < 4: continue
            start, end, c, w = float(cols[0]), float(cols[1]), int(cols[2]), int(cols[3])
            records.append({
                "patient_id": pid, "filename": fname, "equipment": equip,
                "start": start, "end": end, "duration": round(end-start,4),
                "crackle": c, "wheeze": w,
                "label": CLASS_MAP[(c,w)],
                "label_name": CLASS_NAMES[CLASS_MAP[(c,w)]],
                "wav_path": os.path.join(AUDIO_DIR, fname+".wav"),
            })

df = pd.DataFrame(records)
check(f"6898 cycles parsed    (found {len(df)})", len(df)==6898)
check(f"126 unique patients   (found {df['patient_id'].nunique()})", df['patient_id'].nunique()==126)

# 4. Class distribution
counts = df["label_name"].value_counts().reindex(CLASS_NAMES)
print(f"\n  Class distribution:")
for name, v in counts.items():
    print(f"    {name:<8}: {v:>5}  ({v/len(df)*100:.1f}%)")

# 5. Load one audio cycle from each class
print(f"\n  Audio load test (one cycle per class):")
for cls_idx, cls_name in enumerate(CLASS_NAMES):
    subset = df[(df["label"]==cls_idx) & (df["duration"]>1.0) & (df["duration"]<5.0)]
    row    = subset.sample(1, random_state=42).iloc[0]
    try:
        y, sr = librosa.load(row["wav_path"], sr=None, mono=True)
        y     = librosa.resample(y, orig_sr=sr, target_sr=SR_TARGET) if sr!=SR_TARGET else y
        s_idx = int(row["start"]*SR_TARGET)
        e_idx = int(row["end"]  *SR_TARGET)
        cycle = y[s_idx:e_idx]
        cycle = cycle / np.abs(cycle).max() if np.abs(cycle).max() > 0 else cycle
        mel   = librosa.feature.melspectrogram(y=cycle, sr=SR_TARGET, n_mels=64)
        mfcc  = librosa.feature.mfcc(y=cycle, sr=SR_TARGET, n_mfcc=40)
        check(f"{cls_name:<8}: loaded {len(cycle)} samples | mel{mel.shape} | mfcc{mfcc.shape}", True, fatal=False)
    except Exception as e:
        check(f"{cls_name:<8}: FAILED — {e}", False, fatal=False)

print("\n" + "="*55)
print("  SANITY CHECK COMPLETE — Ready for Stage 3")
print("="*55 + "\n")
