import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src import config


class CNNFpga(nn.Module):

    def __init__(
        self,
        num_features: int = len(config.FEATURE_COLUMNS),
        num_classes: int = len(config.META_CLASS_NAMES),
        cnn_channels: int = 32,
        temporal_channels: int = 128,
        dropout: float = 0.3,
        sequence_length: int = None,
    ):
        super().__init__()

        if sequence_length is None:
            sequence_length = config.SEQUENCE_LENGTH

        self.num_features = num_features
        self.sequence_length = sequence_length

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
            self._temporal_input_size = C * F_out
            self._temporal_steps = T_out

        self.temporal = nn.Sequential(
            nn.Conv1d(self._temporal_input_size, temporal_channels,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(temporal_channels),
            nn.ReLU(inplace=True),
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(temporal_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = 1
        T = self.sequence_length
        F = self.num_features

        x_img = x.view(batch, 1, T, F)
        cnn_out = self.cnn(x_img)
        
        b = 1
        C = 64
        T_cnn = self._temporal_steps
        F_cnn = int(self._temporal_input_size / C)

        cnn_out = cnn_out.permute(0, 2, 1, 3).contiguous()
        cnn_out = cnn_out.view(b, T_cnn, C * F_cnn)

        cnn_out = cnn_out.permute(0, 2, 1)
        temporal_out = self.temporal(cnn_out)

        pooled = temporal_out.mean(dim=2)

        x = self.dropout(pooled)
        logits = self.fc(x)
        return logits


def build_fpga_model(num_features: int = None, sequence_length: int = None) -> CNNFpga:
    if num_features is None:
        num_features = len(config.FEATURE_COLUMNS)
    return CNNFpga(num_features=num_features, sequence_length=sequence_length)


def transfer_cnn_weights(fpga_model: CNNFpga, lstm_checkpoint_path: Path = None):
    if lstm_checkpoint_path is None:
        lstm_checkpoint_path = config.MODEL_CHECKPOINT_PATH

    if not lstm_checkpoint_path.exists():
        raise FileNotFoundError(f"CNN-LSTM checkpoint not found: {lstm_checkpoint_path}")

    lstm_state = torch.load(lstm_checkpoint_path, map_location="cpu", weights_only=True)

    cnn_weights = {k: v for k, v in lstm_state.items() if k.startswith("cnn.")}

    missing, unexpected = fpga_model.load_state_dict(cnn_weights, strict=False)

    transferred = len(cnn_weights)
    print(f"[fpga_model] Transferred {transferred} CNN weight tensors")
    print(f"[fpga_model] Layers needing fine-tuning: {len(missing)}")
    print(f"[fpga_model]   → {', '.join(missing)}")

    return fpga_model


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    layer_counts = {}
    for name, param in model.named_parameters():
        layer = name.split(".")[0]
        if layer not in layer_counts:
            layer_counts[layer] = 0
        layer_counts[layer] += param.numel()

    return {
        "total": total,
        "trainable": trainable,
        "by_layer": layer_counts,
    }


def estimate_fpga_resources(model: CNNFpga) -> dict:
    params = count_parameters(model)
    total_params = params["total"]

    dsp_conv1 = 1 * 32 * 3 * 3
    dsp_conv2 = 32 * 64 * 3 * 3
    dsp_temporal = 384 * 128 * 3
    dsp_fc = 128 * 5
    total_dsp = dsp_conv1 + dsp_conv2 + dsp_temporal + dsp_fc

    weight_bytes = total_params * 1
    weight_kb = weight_bytes / 1024

    activation_kb = (1 * 64 * 5 * 6 * 1) / 1024

    bram_kb = weight_kb + activation_kb

    luts = total_params * 2

    resource_pct = {
        "DSP_used": total_dsp,
        "DSP_available": config.FPGA_DSP_BLOCKS,
        "DSP_pct": min(total_dsp / config.FPGA_DSP_BLOCKS * 100, 100),
        "BRAM_used_KB": bram_kb,
        "BRAM_available_KB": config.FPGA_BRAM_KB,
        "BRAM_pct": bram_kb / config.FPGA_BRAM_KB * 100,
        "LUT_used": luts,
        "LUT_available": config.FPGA_LUT_COUNT,
        "LUT_pct": luts / config.FPGA_LUT_COUNT * 100,
        "total_params": total_params,
        "weight_size_KB": weight_kb,
    }

    return resource_pct


def print_model_summary(model: CNNFpga):
    params = count_parameters(model)
    resources = estimate_fpga_resources(model)

    print(f"\n{'=' * 55}")
    print("  CNN-FPGA MODEL SUMMARY")
    print(f"{'=' * 55}")
    print(f"  Input shape     : (batch, {model.sequence_length}, {model.num_features})")
    print(f"  Output classes  : {len(config.META_CLASS_NAMES)}")
    print(f"  Total params    : {params['total']:,}")
    print(f"  Trainable params: {params['trainable']:,}")
    print("\n  Parameters by layer:")
    for layer, count in params["by_layer"].items():
        print(f"    {layer:15s}: {count:>8,}")

    print(f"\n{'=' * 55}")
    print("  ESTIMATED FPGA RESOURCES (Zynq-7020)")
    print(f"{'=' * 55}")
    print(f"  DSP Blocks : {resources['DSP_used']:>6,} / {resources['DSP_available']:>6,} "
          f"({resources['DSP_pct']:.1f}%)")
    print(f"  BRAM (KB)  : {resources['BRAM_used_KB']:>6.1f} / {resources['BRAM_available_KB']:>6.0f} "
          f"({resources['BRAM_pct']:.1f}%)")
    print(f"  LUTs       : {resources['LUT_used']:>6,} / {resources['LUT_available']:>6,} "
          f"({resources['LUT_pct']:.1f}%)")
    print(f"  Weight Size: {resources['weight_size_KB']:.1f} KB (INT8)")
    print(f"{'=' * 55}\n")


def fine_tune(model: CNNFpga, epochs: int = 10, freeze_cnn: bool = True):
    from torch.utils.data import DataLoader, TensorDataset
    from src.sessionization import load_sequences

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[fine-tune] Device: {device}")

    if freeze_cnn:
        for param in model.cnn.parameters():
            param.requires_grad = False
        print("[fine-tune] CNN layers frozen — only training temporal + FC")

    seqs, lbls, _ = load_sequences()
    n = len(lbls)
    indices = torch.randperm(n)
    split = int(n * 0.8)

    train_ds = TensorDataset(seqs[indices[:split]], lbls[indices[:split]])
    val_ds = TensorDataset(seqs[indices[split:]], lbls[indices[split:]])

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE)

    model.to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE,
    )
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(x)
            correct += (out.argmax(1) == y).sum().item()
            total += len(x)

        train_acc = correct / total * 100

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                val_correct += (out.argmax(1) == y).sum().item()
                val_total += len(x)

        val_acc = val_correct / val_total * 100
        print(f"  Epoch {epoch:2d}/{epochs}: Train={train_acc:.2f}% | "
              f"Val={val_acc:.2f}% | Loss={total_loss / total:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), config.FPGA_MODEL_PATH)

    print(f"\n[fine-tune] Best validation accuracy: {best_acc:.2f}%")
    print(f"[fine-tune] Model saved to {config.FPGA_MODEL_PATH}")
    return model


def main():
    parser = argparse.ArgumentParser(description="FPGA-friendly CNN model")
    parser.add_argument("--transfer", action="store_true",
                        help="Transfer CNN weights from trained CNN-LSTM and fine-tune")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Fine-tuning epochs (default: 10)")
    args = parser.parse_args()

    model = build_fpga_model()
    print_model_summary(model)

    batch_size = 4
    dummy = torch.randn(batch_size, config.SEQUENCE_LENGTH, len(config.FEATURE_COLUMNS))
    out = model(dummy)
    print(f"[test] Input: {dummy.shape} → Output: {out.shape}")
    print(f"[test] Predictions: {out.argmax(dim=1).tolist()}")

    if args.transfer:
        print("\n[Step 1] Transferring CNN weights from trained CNN-LSTM...")
        model = transfer_cnn_weights(model)

        print(f"\n[Step 2] Fine-tuning temporal + FC layers ({args.epochs} epochs)...")
        model = fine_tune(model, epochs=args.epochs, freeze_cnn=True)

        print("\n[Step 3] Exporting FPGA model to ONNX...")
        from src.fpga_export import export_onnx
        export_onnx(model, output_path=config.ONNX_FPGA_MODEL_PATH)

        print("\n✓ FPGA model ready for deployment!")


if __name__ == "__main__":
    main()
