import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src import config


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.se(x).reshape(x.size(0), -1, 1, 1)
        return x * w


class MultiScaleDilatedCNN(nn.Module):
    def __init__(self, in_channels=1, base_filters=64):
        super().__init__()
        bf = base_filters

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, bf, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(bf),
            nn.GELU(),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, bf, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(bf),
            nn.GELU(),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, bf, kernel_size=3, padding=6, dilation=6),
            nn.BatchNorm2d(bf),
            nn.GELU(),
        )

        total_ch = bf * 3
        self.se = SEBlock(total_ch, reduction=16)
        self.residual = nn.Conv2d(in_channels, total_ch, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        out = torch.cat([b1, b2, b3], dim=1)
        out = self.se(out)
        out = out + self.residual(x)
        out = self.pool(out)
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        return self.pool(self.conv(x) + self.skip(x))


class HybridTemporalBlock(nn.Module):
    def __init__(self, input_size, lstm_hidden=128, gru_hidden=128, dropout=0.2):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )

        self.lstm = nn.LSTM(
            256, lstm_hidden, num_layers=2,
            batch_first=True, dropout=dropout, bidirectional=True,
        )

        self.gru = nn.GRU(
            256, gru_hidden, num_layers=2,
            batch_first=True, dropout=dropout,
        )

        fusion_input = lstm_hidden * 2 + gru_hidden
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.proj(x)

        lstm_out, _ = self.lstm(x)
        gru_out, _ = self.gru(x)

        combined = torch.cat([lstm_out, gru_out], dim=-1)

        fused_seq = self.fusion(combined)

        pooled = fused_seq[:, -1, :]

        return fused_seq, pooled


class MultiHeadTemporalAttention(nn.Module):
    def __init__(self, d_model=256, num_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, return_weights=False):
        B, S, D = x.shape

        Q = self.W_q(x).reshape(B, S, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).reshape(B, S, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).reshape(B, S, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).reshape(B, S, D)

        out = self.W_o(context)
        out = self.norm(x + out)

        attn_pool = attn.mean(dim=1).mean(dim=1)
        attn_pool = F.softmax(attn_pool, dim=-1).unsqueeze(-1)
        pooled = (out * attn_pool).sum(dim=1)

        if return_weights:
            weights = attn.mean(dim=1).mean(dim=1)
            return pooled, weights
        return pooled


class SentinelV2(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_classes: int = None,
        sequence_length: int = None,
        cnn_base_filters: int = None,
        lstm_hidden: int = None,
        gru_hidden: int = None,
        num_heads: int = None,
        dropout_cnn: float = None,
        dropout_rnn: float = None,
        dropout_head: float = None,
    ):
        super().__init__()

        if num_classes is None:
            num_classes = config.NUM_CLASSES
        if sequence_length is None:
            sequence_length = config.SEQUENCE_LENGTH
        if cnn_base_filters is None:
            cnn_base_filters = config.V2_CNN_BASE_FILTERS
        if lstm_hidden is None:
            lstm_hidden = config.V2_LSTM_HIDDEN
        if gru_hidden is None:
            gru_hidden = config.V2_GRU_HIDDEN
        if num_heads is None:
            num_heads = config.V2_NUM_ATTENTION_HEADS
        if dropout_cnn is None:
            dropout_cnn = config.V2_DROPOUT_CNN
        if dropout_rnn is None:
            dropout_rnn = config.V2_DROPOUT_RNN
        if dropout_head is None:
            dropout_head = config.V2_DROPOUT_HEAD

        self.num_features = num_features
        self.num_classes = num_classes
        self.sequence_length = sequence_length

        total_cnn_ch = cnn_base_filters * 3
        self.cnn = nn.Sequential(
            MultiScaleDilatedCNN(1, cnn_base_filters),
            nn.Dropout2d(dropout_cnn),
            ConvBlock(total_cnn_ch, total_cnn_ch * 2),
            nn.Dropout2d(dropout_cnn),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, sequence_length, num_features)
            cnn_out = self.cnn(dummy)
            _, C, T_out, F_out = cnn_out.shape
            self._cnn_flat = C * F_out
            self._cnn_T = T_out

        self.temporal = HybridTemporalBlock(
            input_size=self._cnn_flat,
            lstm_hidden=lstm_hidden,
            gru_hidden=gru_hidden,
            dropout=dropout_rnn,
        )

        self.attention = MultiHeadTemporalAttention(
            d_model=256,
            num_heads=num_heads,
            dropout=dropout_cnn,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout_head),
            nn.Linear(128, num_classes),
        )

    def forward(self, x, return_attention=False):
        B, T, N_F = x.shape

        x_img = x.view(B, 1, T, N_F)
        cnn_out = self.cnn(x_img)
        _, C, T_cnn, F_cnn = cnn_out.shape

        temporal_in = cnn_out.permute(0, 2, 1, 3).contiguous().view(B, T_cnn, C * F_cnn)

        seq_out, _ = self.temporal(temporal_in)

        if return_attention:
            pooled, attn_weights = self.attention(seq_out, return_weights=True)
            logits = self.classifier(pooled)
            return logits, attn_weights
        else:
            pooled = self.attention(seq_out)
            logits = self.classifier(pooled)
            return logits


def build_model_v2(num_features: int, sequence_length: int = None,
                   num_classes: int = None) -> SentinelV2:
    return SentinelV2(
        num_features=num_features,
        num_classes=num_classes,
        sequence_length=sequence_length,
    )


if __name__ == "__main__":
    batch_size = 8
    T = config.SEQUENCE_LENGTH
    n_feat = 30

    dummy = torch.randn(batch_size, T, n_feat)
    model = build_model_v2(num_features=n_feat)

    out = model(dummy)
    print(f"Model: SentinelV2 (MultiScale CNN-BiLSTM-GRU-MHA)")
    print(f"Input shape:  ({batch_size}, {T}, {n_feat})")
    print(f"Output shape: {out.shape}")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters:   {param_count:,}")

    out, attn = model(dummy, return_attention=True)
    print(f"Attention shape: {attn.shape}")
    print(f"Attention sum:   {attn[0].sum():.4f}")
