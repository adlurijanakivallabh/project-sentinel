import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src import config


class SelfAttention(nn.Module):

    def __init__(self, hidden_size: int):
        super().__init__()
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, hidden_size)
        self.W_v = nn.Linear(hidden_size, hidden_size)
        self.scale = hidden_size ** 0.5

    def forward(self, hidden_states: torch.Tensor, return_weights: bool = False):
        Q = self.W_q(hidden_states)
        K = self.W_k(hidden_states)
        V = self.W_v(hidden_states)

        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        weights = F.softmax(scores, dim=-1)

        context = torch.bmm(weights, V)
        context = context.mean(dim=1)

        if return_weights:
            attn_per_step = weights.mean(dim=1)
            return context, attn_per_step
        return context


class MultiScaleCNN(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        ch1 = out_channels // 4
        ch3 = out_channels // 2
        ch5 = out_channels - ch1 - ch3

        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1, kernel_size=1),
            nn.BatchNorm2d(ch1),
            nn.ReLU(inplace=True),
        )
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, ch3, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch3),
            nn.ReLU(inplace=True),
        )
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, ch5, kernel_size=5, padding=2),
            nn.BatchNorm2d(ch5),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        b1 = self.branch1x1(x)
        b3 = self.branch3x3(x)
        b5 = self.branch5x5(x)
        return torch.cat([b1, b3, b5], dim=1)


class ResidualConvBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.skip = (nn.Conv2d(in_channels, out_channels, kernel_size=1)
                     if in_channels != out_channels else nn.Identity())

    def forward(self, x):
        residual = self.skip(x)
        out = self.relu(self.bn(self.conv(x)))
        return out + residual


class CNNLSTM(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_classes: int = None,
        cnn_channels: int = 32,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 1,
        dropout: float = 0.3,
        sequence_length: int = None,
        use_attention: bool = None,
        use_multiscale_cnn: bool = None,
        use_residual_cnn: bool = None,
    ):
        super().__init__()

        if num_classes is None:
            num_classes = config.NUM_CLASSES
        if sequence_length is None:
            sequence_length = config.SEQUENCE_LENGTH
        if use_attention is None:
            use_attention = "attention" in config.MODEL_VARIANT
        if use_multiscale_cnn is None:
            use_multiscale_cnn = config.USE_MULTISCALE_CNN
        if use_residual_cnn is None:
            use_residual_cnn = config.USE_RESIDUAL_CNN

        self.num_features = num_features
        self.sequence_length = sequence_length
        self.use_attention = use_attention

        if use_multiscale_cnn:
            self.cnn = nn.Sequential(
                MultiScaleCNN(1, cnn_channels),
                nn.MaxPool2d(kernel_size=(2, 2)),
                MultiScaleCNN(cnn_channels, cnn_channels * 2),
                nn.MaxPool2d(kernel_size=(2, 2)),
            )
        elif use_residual_cnn:
            self.cnn = nn.Sequential(
                ResidualConvBlock(1, cnn_channels),
                nn.MaxPool2d(kernel_size=(2, 2)),
                ResidualConvBlock(cnn_channels, cnn_channels * 2),
                nn.MaxPool2d(kernel_size=(2, 2)),
            )
        else:
            self.cnn = nn.Sequential(
                nn.Conv2d(1, cnn_channels, kernel_size=(3, 3), padding=(1, 1)),
                nn.BatchNorm2d(cnn_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2)),
                nn.Conv2d(cnn_channels, cnn_channels * 2, kernel_size=(3, 3), padding=(1, 1)),
                nn.BatchNorm2d(cnn_channels * 2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2)),
            )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, sequence_length, num_features)
            cnn_out = self.cnn(dummy)
            _, C, T_out, F_out = cnn_out.shape
            self._lstm_input_size = C * F_out
            self._cnn_T_out = T_out

        self.lstm = nn.LSTM(
            input_size=self._lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=False,
        )

        if self.use_attention:
            self.attention = SelfAttention(lstm_hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        batch, T, F = x.shape

        x_img = x.view(batch, 1, T, F)
        cnn_out = self.cnn(x_img)
        b, C, T_cnn, F_cnn = cnn_out.shape

        cnn_out = cnn_out.permute(0, 2, 1, 3).contiguous()
        cnn_out = cnn_out.view(b, T_cnn, C * F_cnn)

        lstm_out, _ = self.lstm(cnn_out)

        if self.use_attention:
            if return_attention:
                context, attn_weights = self.attention(lstm_out, return_weights=True)
                x = self.dropout(context)
                logits = self.fc(x)
                return logits, attn_weights
            else:
                context = self.attention(lstm_out)
                x = self.dropout(context)
                logits = self.fc(x)
                return logits
        else:
            last_hidden = lstm_out[:, -1, :]
            x = self.dropout(last_hidden)
            logits = self.fc(x)
            return logits


def build_model(num_features: int, sequence_length: int = None,
                num_classes: int = None) -> CNNLSTM:
    model = CNNLSTM(
        num_features=num_features,
        sequence_length=sequence_length,
        num_classes=num_classes,
    )
    return model


if __name__ == "__main__":
    batch_size, T, F = 8, config.SEQUENCE_LENGTH, 30
    dummy = torch.randn(batch_size, T, F)

    model = build_model(num_features=F)
    out = model(dummy)
    print(f"Model variant: {config.MODEL_VARIANT}")
    print(f"Use attention: {model.use_attention}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if model.use_attention:
        out, attn = model(dummy, return_attention=True)
        print(f"Attention weights shape: {attn.shape}")
        print(f"Attention weights sum: {attn[0].sum():.4f}")
