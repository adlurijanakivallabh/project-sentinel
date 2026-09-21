import sys
from pathlib import Path
import torch
import torch.nn as nn

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src import config
from src.model_cnn_lstm import SelfAttention, MultiScaleCNN, ResidualConvBlock


class CNNBiLSTM(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_classes: int = None,
        cnn_channels: int = 32,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 1,
        dropout: float = 0.3,
        sequence_length: int = None,
        use_attention: bool = True,
        use_multiscale_cnn: bool = None,
        use_residual_cnn: bool = None,
    ):
        super().__init__()

        if num_classes is None:
            num_classes = config.NUM_CLASSES
        if sequence_length is None:
            sequence_length = config.SEQUENCE_LENGTH
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

        self.lstm = nn.LSTM(
            input_size=self._lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
        )

        bi_hidden = lstm_hidden_size * 2

        if self.use_attention:
            self.attention = SelfAttention(bi_hidden)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(bi_hidden, num_classes)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
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


def build_bilstm_model(num_features: int, sequence_length: int = None,
                       num_classes: int = None) -> CNNBiLSTM:
    return CNNBiLSTM(
        num_features=num_features,
        sequence_length=sequence_length,
        num_classes=num_classes,
    )


if __name__ == "__main__":
    batch_size, T, F = 8, config.SEQUENCE_LENGTH, 30
    dummy = torch.randn(batch_size, T, F)

    model = build_bilstm_model(num_features=F)
    out = model(dummy)
    print(f"CNN-BiLSTM Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if model.use_attention:
        out, attn = model(dummy, return_attention=True)
        print(f"Attention weights shape: {attn.shape}")
