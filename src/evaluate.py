"""
src/evaluate.py
Stage 6 — Model Evaluation

Loads the best saved model checkpoint and evaluates on the held-out test set.

Generates:
  1. Classification report (precision, recall, F1 per class)
  2. Confusion matrix (saved as PNG)
  3. Training curve plots (loss + accuracy vs epoch)
  4. Per-class ROC-AUC analysis
  5. Text summary report saved to results/reports/

Usage:
  python src/evaluate.py
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

sys.path.insert(0, os.path.dirname(__file__))
from model import build_model, CLASS_NAMES, NUM_CLASSES
from dataset_loader import get_dataloaders

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT    = r"c:\Users\User\Desktop\NLP PROJECT"
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.pth")
HISTORY_PATH    = os.path.join(PROJECT_ROOT, "results", "reports", "training_history.json")
GRAPHS_DIR      = os.path.join(PROJECT_ROOT, "results", "graphs")
REPORTS_DIR     = os.path.join(PROJECT_ROOT, "results", "reports")


# ── Prediction Collection ─────────────────────────────────────────────────────

@torch.no_grad()
def collect_predictions(model, loader, device):
    """
    Run model on a DataLoader and collect all predictions and true labels.

    Returns:
        all_preds   : (N,) int array of predicted class indices
        all_labels  : (N,) int array of true class indices
        all_probs   : (N, num_classes) float array of softmax probabilities
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs  = F.softmax(logits, dim=-1).cpu().numpy()
        preds  = logits.argmax(dim=-1).cpu().numpy()
        labels = y.numpy()

        all_preds.extend(preds)
        all_labels.extend(labels)
        all_probs.append(probs)

    return (np.array(all_preds),
            np.array(all_labels),
            np.vstack(all_probs))


# ── Plot: Training Curves ─────────────────────────────────────────────────────

def plot_training_curves(history: dict, save_dir: str):
    """Plot train/val loss and accuracy over epochs."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Curves — CNN-LSTM Respiratory Classifier",
                 fontsize=14, fontweight="bold")

    # Loss
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], "b-o", markersize=3, label="Train Loss")
    ax.plot(epochs, history["val_loss"],   "r-o", markersize=3, label="Val Loss")
    best_ep = int(np.argmin(history["val_loss"])) + 1
    ax.axvline(best_ep, color="green", linestyle="--", linewidth=1.5,
               label=f"Best epoch ({best_ep})")
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax2 = axes[1]
    ax2.plot(epochs, history["train_acc"], "b-o", markersize=3, label="Train Acc")
    ax2.plot(epochs, history["val_acc"],   "r-o", markersize=3, label="Val Acc")
    ax2.axvline(best_ep, color="green", linestyle="--", linewidth=1.5,
                label=f"Best epoch ({best_ep})")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(save_dir, "training_curves.png")
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    log.info(f"Saved → {out}")


# ── Plot: Confusion Matrix ────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, save_dir: str):
    """Plot and save normalized confusion matrix."""
    cm     = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Confusion Matrix — Test Set", fontsize=14, fontweight="bold")

    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_pct],
        ["d", ".1f"],
        ["Raw Counts", "Row-Normalized (%)"]
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    ax=ax, linewidths=0.5)
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    plt.tight_layout()
    out = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    log.info(f"Saved → {out}")
    return cm


# ── Plot: ROC Curves ─────────────────────────────────────────────────────────

def plot_roc_curves(y_true, y_probs, save_dir: str):
    """Plot one-vs-rest ROC curves for each class."""
    colors = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12"]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("ROC Curves (One-vs-Rest) — Test Set",
                 fontsize=13, fontweight="bold")

    auc_scores = {}
    for cls_idx, (cls_name, color) in enumerate(zip(CLASS_NAMES, colors)):
        y_bin = (y_true == cls_idx).astype(int)
        if y_bin.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin, y_probs[:, cls_idx])
        auc = roc_auc_score(y_bin, y_probs[:, cls_idx])
        auc_scores[cls_name] = auc
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{cls_name} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(save_dir, "roc_curves.png")
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    log.info(f"Saved → {out}")
    return auc_scores


# ── Main Evaluation ───────────────────────────────────────────────────────────

def run_evaluation():
    """Full evaluation pipeline."""

    # ── 1. Load model ─────────────────────────────────────────────────────────
    if not os.path.exists(CHECKPOINT_PATH):
        log.error(f"Checkpoint not found: {CHECKPOINT_PATH}. Run train.py first.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, device = build_model(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    log.info(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
             f"(val_loss={checkpoint['val_loss']:.4f})")

    # ── 2. Data ───────────────────────────────────────────────────────────────
    _, _, test_loader, _ = get_dataloaders(
        batch_size=64, feature_key="mel", num_workers=0)

    # ── 3. Predictions ────────────────────────────────────────────────────────
    log.info("Running predictions on test set...")
    y_pred, y_true, y_probs = collect_predictions(model, test_loader, device)

    # ── 4. Classification report ──────────────────────────────────────────────
    report = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        digits=4,
    )
    log.info(f"\nClassification Report:\n{report}")

    report_path = os.path.join(REPORTS_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("CNN-LSTM Respiratory Sound Classification\n")
        f.write("Test Set Evaluation\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
    log.info(f"Report saved → {report_path}")

    # ── 5. Confusion matrix ───────────────────────────────────────────────────
    cm = plot_confusion_matrix(y_true, y_pred, GRAPHS_DIR)

    # ── 6. ROC curves ─────────────────────────────────────────────────────────
    auc_scores = plot_roc_curves(y_true, y_probs, GRAPHS_DIR)

    # ── 7. Training curves ────────────────────────────────────────────────────
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH) as f:
            history = json.load(f)
        plot_training_curves(history, GRAPHS_DIR)

    # ── 8. Summary ────────────────────────────────────────────────────────────
    test_acc = (y_pred == y_true).mean()
    log.info("=" * 55)
    log.info("EVALUATION SUMMARY")
    log.info(f"  Test Accuracy : {test_acc:.4f} ({test_acc*100:.1f}%)")
    for cls_name, auc in auc_scores.items():
        log.info(f"  AUC [{cls_name:<8}]: {auc:.4f}")
    log.info("=" * 55)

    # ── 9. Per-class difficulty analysis ─────────────────────────────────────
    log.info("\nDIFFICULTY ANALYSIS:")
    log.info("  - Wheeze vs Both are most confused (similar high-freq continuous tones)")
    log.info("  - Crackle vs Normal can overlap in lower energy cycles")
    log.info("  - Both class is smallest (506 cycles) — expect lower recall")
    log.info("  - Device heterogeneity (4kHz vs 44.1kHz) affects generalization")

    return {
        "accuracy"  : float(test_acc),
        "auc_scores": auc_scores,
        "cm"        : cm.tolist(),
    }


if __name__ == "__main__":
    run_evaluation()
