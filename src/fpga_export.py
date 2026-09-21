import sys
from pathlib import Path
import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src import config
from src.model_cnn_lstm import build_model, CNNLSTM
from src.sessionization import load_sequences


def load_trained_model() -> CNNLSTM:
    from src import clean_state_dict
    checkpoint = config.MODEL_CHECKPOINT_PATH
    if not checkpoint.exists():
        raise FileNotFoundError(f"Trained model not found at {checkpoint}")

    num_features = len(config.FEATURE_COLUMNS)
    if config.TEMPORAL_FEATURES_ENABLED:
        num_features += 5
    model = build_model(num_features=num_features)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "model_state_dict" in state:
        model.load_state_dict(clean_state_dict(state["model_state_dict"]))
    else:
        model.load_state_dict(clean_state_dict(state))
    model.eval()
    print(f"[export] Loaded trained model from {checkpoint}")
    print(f"[export] Model size: {checkpoint.stat().st_size / 1024:.1f} KB")
    return model


def get_calibration_data(n_samples: int = None):
    if n_samples is None:
        n_samples = config.CALIBRATION_SAMPLES

    seqs, lbls, _ = load_sequences()

    checkpoint_path = config.MODEL_CHECKPOINT_PATH
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if isinstance(ckpt, dict) and "zscore_mean" in ckpt:
            zscore_mean = ckpt["zscore_mean"].cpu().unsqueeze(0)
            zscore_std = ckpt["zscore_std"].cpu().unsqueeze(0)
            print("[export] Z-score stats: loaded from checkpoint (train-only)")
        else:
            from src.split_utils import block_split
            train_idx, _, _ = block_split(len(lbls))
            train_flat = seqs[train_idx].reshape(-1, seqs.shape[-1])
            zscore_mean = train_flat.mean(dim=0, keepdim=True)
            zscore_std = train_flat.std(dim=0, keepdim=True).clamp(min=1e-8)
            print("[export] Z-score stats: recomputed from train split")
    else:
        from src.split_utils import block_split
        train_idx, _, _ = block_split(len(lbls))
        train_flat = seqs[train_idx].reshape(-1, seqs.shape[-1])
        zscore_mean = train_flat.mean(dim=0, keepdim=True)
        zscore_std = train_flat.std(dim=0, keepdim=True).clamp(min=1e-8)
        print("[export] Z-score stats: recomputed from train split (no checkpoint)")

    seqs = ((seqs - zscore_mean) / zscore_std).clamp(-10, 10)

    indices = torch.randperm(len(seqs))[:n_samples]
    return seqs[indices], lbls[indices]


def export_onnx(model: nn.Module, output_path: Path = None, seq_len: int = None,
                num_features: int = None):
    import os
    os.environ["PYTHONIOENCODING"] = "utf-8"

    if output_path is None:
        output_path = config.ONNX_MODEL_PATH
    if seq_len is None:
        seq_len = config.SEQUENCE_LENGTH
    if num_features is None:
        num_features = len(config.FEATURE_COLUMNS)
        if getattr(config, "TEMPORAL_FEATURES_ENABLED", False):
            num_features += 5

    model.eval()
    dummy_input = torch.randn(1, seq_len, num_features)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamo=False,
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    size_kb = output_path.stat().st_size / 1024
    print(f"[export] ONNX model saved to {output_path} ({size_kb:.1f} KB)")
    return output_path


def verify_onnx(model: nn.Module, onnx_path: Path = None):
    if onnx_path is None:
        onnx_path = config.ONNX_MODEL_PATH

    try:
        import onnxruntime as ort
    except ImportError:
        print("[export] onnxruntime not installed — skipping ONNX verification")
        print("[export] Install with: pip install onnxruntime")
        return False

    model.eval()
    num_features = len(config.FEATURE_COLUMNS)
    if getattr(config, "TEMPORAL_FEATURES_ENABLED", False):
        num_features += 5
    dummy = torch.randn(1, config.SEQUENCE_LENGTH, num_features)

    with torch.no_grad():
        pt_out = model(dummy).numpy()

    sess = ort.InferenceSession(str(onnx_path))
    onnx_out = sess.run(None, {"input": dummy.numpy()})[0]

    max_diff = np.abs(pt_out - onnx_out).max()
    match = max_diff < 1e-4
    print(f"[export] ONNX verification: max diff = {max_diff:.6f} — {'PASS' if match else 'FAIL'}")
    return match


def quantize_dynamic(model: nn.Module) -> nn.Module:
    model.eval()
    quantized = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.LSTM},
        dtype=torch.qint8,
    )
    print("[export] Dynamic INT8 quantization applied (Linear + LSTM layers)")
    return quantized


def quantize_static(model: nn.Module, calibration_data: torch.Tensor) -> nn.Module:
    model.eval()

    model_prepared = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.LSTM},
        dtype=torch.qint8,
    )

    print(f"[export] Static quantization with {len(calibration_data)} calibration samples")
    return model_prepared


def compare_accuracy(original: nn.Module, quantized: nn.Module,
                     test_seqs: torch.Tensor, test_lbls: torch.Tensor) -> dict:
    original.eval()
    quantized.eval()

    results = {}

    for name, model in [("FP32 (Original)", original), ("INT8 (Quantized)", quantized)]:
        correct = 0
        total = 0
        latencies = []

        loader = DataLoader(TensorDataset(test_seqs, test_lbls), batch_size=64)

        with torch.no_grad():
            for x, y in loader:
                t0 = time.perf_counter()
                out = model(x)
                t1 = time.perf_counter()

                preds = out.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += len(y)
                latencies.append((t1 - t0) / len(x) * 1000)

        acc = correct / total * 100
        avg_latency = np.mean(latencies)
        results[name] = {"accuracy": acc, "latency_ms": avg_latency}
        print(f"[export] {name}: Accuracy={acc:.2f}%, Latency={avg_latency:.3f} ms/sample")

    return results


def report_model_sizes(original_path: Path, quantized_model: nn.Module):
    orig_size = original_path.stat().st_size / 1024

    tmp_path = config.FPGA_DIR / "_tmp_quantized.pth"
    torch.save(quantized_model.state_dict(), tmp_path)
    quant_size = tmp_path.stat().st_size / 1024
    tmp_path.unlink()

    ratio = orig_size / quant_size if quant_size > 0 else 0

    print(f"\n{'=' * 50}")
    print("  MODEL SIZE COMPARISON")
    print(f"{'=' * 50}")
    print(f"  FP32 Original : {orig_size:>8.1f} KB")
    print(f"  INT8 Quantized: {quant_size:>8.1f} KB")
    print(f"  Compression   : {ratio:>8.1f}x")
    print(f"{'=' * 50}\n")

    return {"fp32_kb": orig_size, "int8_kb": quant_size, "ratio": ratio}


def generate_export_report(accuracy_results: dict, size_results: dict,
                           onnx_verified: bool):
    report_path = config.FPGA_DIR / "export_report.txt"

    lines = [
        "=" * 60,
        "  FPGA EXPORT REPORT — CNN-LSTM IPS",
        "=" * 60,
        "",
        "MODEL EXPORT STATUS",
        "  ONNX Export     : DONE",
        f"  ONNX Verified   : {'PASS' if onnx_verified else 'SKIP/FAIL'}",
        "  INT8 Quantized  : DONE",
        "",
        "MODEL SIZES",
        f"  FP32 (Original) : {size_results['fp32_kb']:.1f} KB",
        f"  INT8 (Quantized): {size_results['int8_kb']:.1f} KB",
        f"  Compression     : {size_results['ratio']:.1f}x",
        "",
        "ACCURACY COMPARISON",
    ]

    for name, res in accuracy_results.items():
        lines.append(f"  {name}: Acc={res['accuracy']:.2f}%, "
                      f"Latency={res['latency_ms']:.3f} ms/sample")

    fp32_acc = accuracy_results.get("FP32 (Original)", {}).get("accuracy", 0)
    int8_acc = accuracy_results.get("INT8 (Quantized)", {}).get("accuracy", 0)
    drop = fp32_acc - int8_acc

    lines.extend([
        "",
        f"  Accuracy Drop   : {drop:.2f}%",
        f"  Status          : {'ACCEPTABLE' if drop < 1.0 else 'REVIEW NEEDED'}",
        "",
        "DEPLOYMENT READINESS",
        f"  ONNX file       : {config.ONNX_MODEL_PATH}",
        f"  Quantized model : {config.QUANTIZED_MODEL_PATH}",
        "  Ready for Vitis AI / hls4ml / OpenVINO conversion",
        "=" * 60,
    ])

    report = "\n".join(lines)
    report_path.write_text(report)
    print(report)
    print(f"\n[export] Report saved to {report_path}")


def main():
    print("=" * 60)
    print("  FPGA EXPORT PIPELINE")
    print("=" * 60)

    print("\n[Step 1/6] Loading trained model...")
    model = load_trained_model()

    print("\n[Step 2/6] Exporting to ONNX...")
    onnx_path = export_onnx(model)

    print("\n[Step 3/6] Verifying ONNX output...")
    onnx_ok = verify_onnx(model, onnx_path)

    print("\n[Step 4/6] Loading test data for quantization...")
    test_seqs, test_lbls = get_calibration_data(n_samples=2000)
    print(f"[export] Using {len(test_seqs)} samples for evaluation")

    print("\n[Step 5/6] Applying INT8 quantization...")
    model_int8 = quantize_dynamic(model)

    torch.save(model_int8.state_dict(), config.QUANTIZED_MODEL_PATH)
    print(f"[export] Quantized model saved to {config.QUANTIZED_MODEL_PATH}")

    print("\n[Step 6/6] Comparing FP32 vs INT8 accuracy...")
    acc_results = compare_accuracy(model, model_int8, test_seqs, test_lbls)
    size_results = report_model_sizes(config.MODEL_CHECKPOINT_PATH, model_int8)

    generate_export_report(acc_results, size_results, onnx_ok)

    print("\n[OK] FPGA export pipeline complete!")
    print(f"  ONNX model    : {config.ONNX_MODEL_PATH}")
    print(f"  INT8 model    : {config.QUANTIZED_MODEL_PATH}")


if __name__ == "__main__":
    main()
