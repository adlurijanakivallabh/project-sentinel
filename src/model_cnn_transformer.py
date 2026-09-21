import sys
import math
from pathlib import Path
import torch
import torch.nn as nn

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src import config
from src.model_cnn_lstm import MultiScaleCNN, ResidualConvBlock


class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 20, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
        x = x + self.pe(positions)
        return self.dropout(x)


class CNNTransformer(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_classes: int = None,
        cnn_channels: int = 32,
        d_model: int = 128,
        nhead: int = 2,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.3,
        sequence_length: int = None,
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
        self.d_model = d_model

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
            self._cnn_feat_size = C * F_out
            self._cnn_T_out = T_out

        self.input_projection = nn.Linear(self._cnn_feat_size, d_model)

        self.pos_encoder = LearnablePositionalEncoding(d_model, max_len=T_out + 1, dropout=dropout)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        batch, T, F = x.shape

        x_img = x.view(batch, 1, T, F)
        cnn_out = self.cnn(x_img)
        b, C, T_cnn, F_cnn = cnn_out.shape

        cnn_out = cnn_out.permute(0, 2, 1, 3).contiguous()
        cnn_out = cnn_out.view(b, T_cnn, C * F_cnn)

        projected = self.input_projection(cnn_out)

        cls_tokens = self.cls_token.expand(batch, -1, -1)
        
        projected = self.pos_encoder(projected)
        
        projected = torch.cat([cls_tokens, projected], dim=1)

        encoded = self.transformer_encoder(projected)

        cls_output = encoded[:, 0, :]

        x = self.dropout(cls_output)
        logits = self.fc(x)
        return logits


def build_transformer_model(num_features: int, sequence_length: int = None,
                            num_classes: int = None) -> CNNTransformer:
    return CNNTransformer(
        num_features=num_features,
        sequence_length=sequence_length,
        num_classes=num_classes,
    )


if __name__ == "__main__":
    batch_size, T, F = 8, config.SEQUENCE_LENGTH, 30
    dummy = torch.randn(batch_size, T, F)

    model = build_transformer_model(num_features=F)
    out = model(dummy)
    print(f"CNN-Transformer Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
