"""
src/train.py
Stage 5 — Full Training Pipeline

Includes:
  - Weighted cross-entropy loss for class imbalance
  - Adam optimizer with learning rate scheduling (ReduceLROnPlateau)
  - Early stopping (patience-based)
  - Model checkpointing (best val loss)
  - Training history logging
  - GPU acceleration when available
  - Reproducible random seeds

Usage:
  python src/train.py
"""

import os
import json
import time
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ── Project imports ───────────────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(__file__))
from model import build_model, NUM_CLASSES, CLASS_NAMES
from dataset_loader import get_dataloaders

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Hyperparameters ───────────────────────────────────────────────────────────
CONFIG = {
    "seed"           : 42,
    "batch_size"     : 32,
    "num_epochs"     : 35,
    "learning_rate"  : 3e-4,
    "weight_decay"   : 1e-4,
    "lr_patience"    : 4,       # reduce LR if val_loss doesn't improve for 4 epochs
    "lr_factor"      : 0.5,     # multiply LR by this on plateau
    "lr_min"         : 1e-6,
    "early_stop_patience": 8,   # stop if no improvement for 8 epochs
    "feature_key"    : "mel",   # 'mel' | 'mfcc' | 'chroma' | 'spec'
    "num_workers"    : 0,       # 0 = main process only (safest on Windows)
}

PROJECT_ROOT = r"c:\Users\User\Desktop\NLP PROJECT"
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR  = os.path.join(PROJECT_ROOT, "results", "reports")
GRAPHS_DIR   = os.path.join(PROJECT_ROOT, "results", "graphs")

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR,  exist_ok=True)

CHECKPOINT_PATH = os.path.join(MODELS_DIR, "best_model.pth")
HISTORY_PATH    = os.path.join(REPORTS_DIR, "training_history.json")


# ── Helpers ───────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    """Fix all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute batch accuracy."""
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


# ── Training & Validation Loops ───────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    """Run one full training epoch. Returns avg loss and avg accuracy."""
    model.train()
    total_loss, total_acc, n_batches = 0.0, 0.0, 0

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            # Mixed precision (AMP) for CUDA
            with torch.autocast(device_type="cuda"):
                logits = model(x)
                loss   = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        total_acc  += accuracy(logits, y)
        n_batches  += 1

    return total_loss / n_batches, total_acc / n_batches


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Run validation/test evaluation. Returns avg loss and avg accuracy."""
    model.eval()
    total_loss, total_acc, n_batches = 0.0, 0.0, 0

    for x, y in loader:
        x, y   = x.to(device), y.to(device)
        logits = model(x)
        loss   = criterion(logits, y)
        total_loss += loss.item()
        total_acc  += accuracy(logits, y)
        n_batches  += 1

    return total_loss / n_batches, total_acc / n_batches


# ── Main Training Function ────────────────────────────────────────────────────

def train(config: dict = CONFIG) -> dict:
    """
    Full training pipeline.

    Returns:
        history dict with train/val loss and accuracy per epoch.
    """
    set_seed(config["seed"])

    # ── Device ────────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")
    if device == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Data ──────────────────────────────────────────────────────────────────
    log.info("Loading data...")
    train_loader, val_loader, test_loader, class_weights = get_dataloaders(
        batch_size=config["batch_size"],
        feature_key=config["feature_key"],
        num_workers=config["num_workers"],
        seed=config["seed"],
    )
    log.info(f"  Train batches : {len(train_loader)}")
    log.info(f"  Val batches   : {len(val_loader)}")
    log.info(f"  Test batches  : {len(test_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model, device = build_model(device)
    log.info(f"Model parameters: {model.count_parameters():,}")

    # ── Loss: balanced cross-entropy for training, unweighted for val ─────────
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    train_criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.05)
    val_criterion   = nn.CrossEntropyLoss()
    log.info(f"Balanced class weights: {[f'{w:.4f}' for w in class_weights]}")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = Adam(model.parameters(),
                     lr=config["learning_rate"],
                     weight_decay=config["weight_decay"])

    # ── LR Scheduler ──────────────────────────────────────────────────────────
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min",
        patience=config["lr_patience"],
        factor=config["lr_factor"],
        min_lr=config["lr_min"],
    )

    # ── Mixed precision scaler (CUDA only) ────────────────────────────────────
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    # ── Training state ────────────────────────────────────────────────────────
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
        "lr":         [],
    }
    best_val_loss     = float("inf")
    epochs_no_improve = 0
    best_epoch        = 0

    log.info("=" * 60)
    log.info("STARTING TRAINING")
    log.info("=" * 60)

    for epoch in range(1, config["num_epochs"] + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, train_criterion, device, scaler)
        val_loss, val_acc = evaluate(
            model, val_loader, val_criterion, device)

        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        elapsed = time.time() - t0

        # ── Log ───────────────────────────────────────────────────────────────
        log.info(
            f"Epoch [{epoch:>3}/{config['num_epochs']}]  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
            f"lr={lr_now:.2e}  [{elapsed:.1f}s]"
        )

        # ── History ───────────────────────────────────────────────────────────
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(lr_now)

        # ── Checkpoint ────────────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            best_epoch        = epoch
            epochs_no_improve = 0
            torch.save({
                "epoch"       : epoch,
                "model_state" : model.state_dict(),
                "optim_state" : optimizer.state_dict(),
                "val_loss"    : val_loss,
                "val_acc"     : val_acc,
                "config"      : config,
            }, CHECKPOINT_PATH)
            log.info(f"  ✅ Best model saved → val_loss={val_loss:.4f}")
        else:
            epochs_no_improve += 1

        # ── Early stopping ────────────────────────────────────────────────────
        if epochs_no_improve >= config["early_stop_patience"]:
            log.info(
                f"Early stopping at epoch {epoch} "
                f"(no improvement for {config['early_stop_patience']} epochs). "
                f"Best epoch: {best_epoch}"
            )
            break

    # ── Save history ──────────────────────────────────────────────────────────
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)
    log.info(f"Training history saved → {HISTORY_PATH}")

    log.info("=" * 60)
    log.info(f"TRAINING COMPLETE — Best epoch: {best_epoch} | "
             f"Best val_loss: {best_val_loss:.4f}")
    log.info("=" * 60)

    return history


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    history = train()
    print("\nFinal validation accuracy:",
          f"{max(history['val_acc']):.4f}")
    print(f"Best model saved at: {CHECKPOINT_PATH}")
