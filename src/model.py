"""
src/model.py
Stage 4 — CNN-LSTM Hybrid Architecture

Architecture:
    Input: (B, 1, 64, 128)  — Mel Spectrogram (batch, channel, freq, time)
        ↓
    CNN Block 1: Conv2d → BN → ReLU → MaxPool
    CNN Block 2: Conv2d → BN → ReLU → MaxPool
    CNN Block 3: Conv2d → BN → ReLU → MaxPool  (optional depth)
        ↓
    Reshape: (B, CNN_channels, freq', time') → (B, time', features)
        ↓
    LSTM: 2-layer bidirectional LSTM for temporal learning
        ↓
    Attention: Self-attention over LSTM outputs
        ↓
    FC → Dropout → Softmax(4 classes)

Design goals:
  - Fits on a laptop GPU (< 2M parameters)
  - Runs on CPU if no GPU is available
  - Fully reproducible with fixed seeds
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_CLASSES = 4
CLASS_NAMES = ["Normal", "Crackle", "Wheeze", "Both"]


# ── Building Blocks ───────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """
    CNN block: Conv2d → BatchNorm → ReLU → MaxPool.
    Uses same-padding for consistent feature map sizes before pooling.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, pool_size: int = 2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size,
                      padding=kernel_size // 2,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_size, stride=pool_size),
            nn.Dropout2d(p=0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TemporalAttention(nn.Module):
    """
    Additive (Bahdanau-style) self-attention over LSTM time steps.
    Weights each time step by learned relevance before aggregation.
    This helps the model focus on the most informative breathing phases.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        # lstm_out: (B, T, H)
        scores = self.attention(lstm_out)          # (B, T, 1)
        weights = F.softmax(scores, dim=1)         # (B, T, 1)
        context = (weights * lstm_out).sum(dim=1)  # (B, H)
        return context


# ── Main CNN-LSTM Model ───────────────────────────────────────────────────────

class CNNLSTMClassifier(nn.Module):
    """
    Hybrid CNN-LSTM model for respiratory sound classification.

    Args:
        num_classes  : Number of output classes (default: 4)
        cnn_channels : Number of CNN filter channels at each stage
        lstm_hidden  : Hidden units per LSTM direction
        lstm_layers  : Number of stacked LSTM layers
        lstm_dropout : Dropout between LSTM layers
        fc_dropout   : Dropout before final classifier
        bidirectional: Whether to use bidirectional LSTM

    Input:
        x: (B, 1, 64, 128)   Mel spectrogram — single-channel 2D feature map

    Output:
        logits: (B, num_classes)   Raw scores (apply softmax for probabilities)
    """

    def __init__(self,
                 num_classes:   int   = NUM_CLASSES,
                 cnn_channels:  tuple = (32, 64, 128),
                 lstm_hidden:   int   = 128,
                 lstm_layers:   int   = 2,
                 lstm_dropout:  float = 0.3,
                 fc_dropout:    float = 0.4,
                 bidirectional: bool  = True):
        super().__init__()

        self.bidirectional = bidirectional
        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)

        # ── CNN Encoder ───────────────────────────────────────────────────────
        # Input: (B, 1, 64, 128)
        # After block1: (B, 32,  32, 64)
        # After block2: (B, 64,  16, 32)
        # After block3: (B, 128,  8, 16)
        self.cnn = nn.Sequential(
            ConvBlock(1,              cnn_channels[0]),   # (B, 32, 32, 64)
            ConvBlock(cnn_channels[0], cnn_channels[1]),  # (B, 64, 16, 32)
            ConvBlock(cnn_channels[1], cnn_channels[2]),  # (B,128,  8, 16)
        )

        # ── CNN → LSTM reshape ────────────────────────────────────────────────
        # After CNN: (B, 128, 8, 16)
        # Reshape to (B, time=16, features=128*8=1024)
        # We treat the last CNN dimension as time steps for LSTM
        self.cnn_out_freq = 8    # freq dimension after 3× pooling of 64 → 8
        self.cnn_out_time = 16   # time dimension after 3× pooling of 128 → 16
        self.lstm_input_dim = cnn_channels[2] * self.cnn_out_freq   # 128*8 = 1024

        # ── LSTM Temporal Encoder ─────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
        )

        # ── Attention over time steps ─────────────────────────────────────────
        self.attention = TemporalAttention(lstm_out_dim)

        # ── Classifier Head ───────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(fc_dropout),
            nn.Linear(128, num_classes),
        )

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        """Initialize CNN weights with Kaiming and LSTM with orthogonal."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        x: (B, 1, 64, 128)
        Returns logits: (B, num_classes)
        """
        B = x.size(0)

        # CNN feature extraction
        cnn_out = self.cnn(x)                         # (B, 128, 8, 16)

        # Reshape for LSTM: treat time axis as sequence
        # (B, C, H, W) → (B, W, C*H) = (B, 16, 1024)
        cnn_out = cnn_out.permute(0, 3, 1, 2)         # (B, 16, 128, 8)
        cnn_out = cnn_out.reshape(B, self.cnn_out_time,
                                  self.lstm_input_dim) # (B, 16, 1024)

        # LSTM temporal learning
        lstm_out, _ = self.lstm(cnn_out)               # (B, 16, lstm_out_dim)

        # Attention pooling
        context = self.attention(lstm_out)              # (B, lstm_out_dim)

        # Classification
        logits = self.classifier(context)               # (B, num_classes)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities (B, num_classes)."""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Model Factory ─────────────────────────────────────────────────────────────

def build_model(device: str | None = None) -> tuple[CNNLSTMClassifier, str]:
    """
    Instantiate the model and move it to the best available device.

    Returns:
        model : CNNLSTMClassifier instance
        device: 'cuda' or 'cpu'
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CNNLSTMClassifier(
        num_classes=NUM_CLASSES,
        cnn_channels=(32, 64, 128),
        lstm_hidden=128,
        lstm_layers=2,
        lstm_dropout=0.3,
        fc_dropout=0.4,
        bidirectional=True,
    ).to(device)

    return model, device


# ── Summary ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)
    model, device = build_model()
    print(f"\nDevice : {device}")
    print(f"Params : {model.count_parameters():,}")

    # Dummy forward pass
    dummy = torch.randn(4, 1, 64, 128).to(device)
    out   = model(dummy)
    proba = model.predict_proba(dummy)

    print(f"\nInput  : {dummy.shape}")
    print(f"Logits : {out.shape}")
    print(f"Proba  : {proba.shape}")
    print(f"\nSample probabilities:\n{proba[0].cpu().numpy()}")
    print(f"Predicted class: {CLASS_NAMES[proba[0].argmax().item()]}")
    print("\nModel architecture:")
    print(model)
