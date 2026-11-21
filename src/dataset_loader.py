"""
src/dataset_loader.py
Stage 4 — PyTorch Dataset & DataLoader

Provides:
  - RespiratorySoundDataset: loads .npz feature files, returns tensors
  - get_dataloaders(): builds train/val/test DataLoaders with
    class-balanced sampling for the train set
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

CLASS_NAMES = ["Normal", "Crackle", "Wheeze", "Both"]
NUM_CLASSES = len(CLASS_NAMES)

PROJECT_ROOT  = r"c:\Users\User\Desktop\NLP PROJECT"
REPORTS_DIR   = os.path.join(PROJECT_ROOT, "results", "reports")
FEATURES_DIR  = os.path.join(PROJECT_ROOT, "dataset", "processed", "features")


# ── Dataset ───────────────────────────────────────────────────────────────────

class RespiratorySoundDataset(Dataset):
    """
    PyTorch Dataset for respiratory sound classification.

    Loads pre-extracted .npz feature files produced by feature_extraction.py.
    Returns the Mel Spectrogram as the primary model input (shape: 1 × 64 × 128).
    Label is a long integer (0-3).

    Args:
        feature_paths : list of paths to .npz files
        labels        : list of integer class labels (0-3)
        augment       : if True, apply training-time audio augmentation
        feature_key   : which feature to use as main input ('mel', 'mfcc', etc.)
    """

    def __init__(self,
                 feature_paths: list[str],
                 labels: list[int],
                 augment: bool = False,
                 feature_key: str = "mel",
                 cache_in_memory: bool = True):
        self.feature_paths    = feature_paths
        self.labels           = labels
        self.augment          = augment
        self.feature_key      = feature_key
        self.cache            = [None] * len(feature_paths) if cache_in_memory else None

    def __len__(self) -> int:
        return len(self.feature_paths)

    def _augment(self, x: np.ndarray) -> np.ndarray:
        """
        Apply training-time spectrogram augmentation.
        SpecAugment-style: frequency masking + time masking.
        Also adds Gaussian noise for robustness.
        """
        x = x.copy()
        _, n_freq, n_time = x.shape

        # 1. Frequency masking (mask up to 8 frequency bins)
        f_mask = np.random.randint(0, min(8, n_freq))
        f_start = np.random.randint(0, n_freq - f_mask + 1)
        x[:, f_start:f_start + f_mask, :] = 0.0

        # 2. Time masking (mask up to 16 time frames)
        t_mask = np.random.randint(0, min(16, n_time))
        t_start = np.random.randint(0, n_time - t_mask + 1)
        x[:, :, t_start:t_start + t_mask] = 0.0

        # 3. Additive Gaussian noise
        noise_level = np.random.uniform(0.001, 0.01)
        x = x + noise_level * np.random.randn(*x.shape).astype(np.float32)

        return x

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cache is not None and self.cache[idx] is not None:
            feat = self.cache[idx]
        else:
            npz = np.load(self.feature_paths[idx])
            feat = npz[self.feature_key].astype(np.float32)   # (n_freq, n_time)
            feat = feat[np.newaxis, :, :]                      # → (1, n_freq, n_time)
            if self.cache is not None:
                self.cache[idx] = feat

        if self.augment:
            feat = self._augment(feat)
        else:
            feat = feat.copy()

        x = torch.tensor(feat, dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


# ── DataLoader Factory ────────────────────────────────────────────────────────

def get_dataloaders(batch_size: int = 32,
                    feature_key: str = "mel",
                    num_workers: int = 0,
                    seed: int = 42) -> tuple[DataLoader, DataLoader, DataLoader, list[float]]:
    """
    Build train / val / test DataLoaders using patient-level splits.

    Uses WeightedRandomSampler on train set to handle class imbalance.

    Returns:
        train_loader, val_loader, test_loader, class_weights (list)
    """
    meta_path = os.path.join(REPORTS_DIR, "feature_metadata.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Feature metadata not found at {meta_path}. "
            "Run feature_extraction.py first."
        )

    df = pd.read_csv(meta_path)
    if len(df) == 0:
        raise ValueError("Feature metadata CSV is empty.")

    # ── Patient-level split (matches Stage 2 logic) ──────────────────────────
    all_patients = df["patient_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(all_patients)

    n_total = len(all_patients)
    n_train = int(0.80 * n_total)
    n_val   = int(0.10 * n_total)

    train_patients = all_patients[:n_train]
    val_patients   = all_patients[n_train:n_train + n_val]
    test_patients  = all_patients[n_train + n_val:]

    train_df = df[df["patient_id"].isin(train_patients)]
    val_df   = df[df["patient_id"].isin(val_patients)]
    test_df  = df[df["patient_id"].isin(test_patients)]

    # Verify no leakage
    assert len(set(train_patients) & set(val_patients)) == 0
    assert len(set(train_patients) & set(test_patients)) == 0

    print(f"Split  → Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    def _make_dataset(split_df, augment):
        return RespiratorySoundDataset(
            feature_paths=split_df["feature_path"].tolist(),
            labels=split_df["label"].tolist(),
            augment=augment,
            feature_key=feature_key,
        )

    train_ds = _make_dataset(train_df, augment=True)
    val_ds   = _make_dataset(val_df,   augment=False)
    test_ds  = _make_dataset(test_df,  augment=False)

    # ── Class weights for balanced loss ───────────────────────────────────────
    train_labels = train_df["label"].values
    class_counts = np.bincount(train_labels, minlength=NUM_CLASSES).astype(float)
    class_counts = np.where(class_counts == 0, 1.0, class_counts)
    total_samples = float(len(train_labels))
    class_weights = (total_samples / (NUM_CLASSES * class_counts)).tolist()

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_weights


if __name__ == "__main__":
    tl, vl, tst, cw = get_dataloaders(batch_size=32)
    x, y = next(iter(tl))
    print(f"Batch shape : {x.shape}")   # (32, 1, 64, 128)
    print(f"Labels      : {y}")
    print(f"Class weights: {[f'{w:.4f}' for w in cw]}")
