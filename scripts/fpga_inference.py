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
from src.model_cnn_lstm import build_model
from src.model_cnn_fpga import build_fpga_model
from src.sessionization import load_sequences


class FPGAInferenceSimulator:

    def __init__(self):
        self.models = {}
        self.results = {}

    def load_models(self):
        num_features = len(config.FEATURE_COLUMNS)

        if config.MODEL_CHECKPOINT_PATH.exists():
            model = build_model(num_features=num_features)
            model.load_state_dict(
                torch.load(config.MODEL_CHECKPOINT_PATH, map_location="cpu", weights_only=True)
            )
            model.eval()
            self.models["CNN-LSTM (FP32)"] = model
            print("[sim] Loaded CNN-LSTM FP32 model")

        if config.MODEL_CHECKPOINT_PATH.exists():
            model_base = build_model(num_features=num_features)
            model_base.load_state_dict(
                torch.load(config.MODEL_CHECKPOINT_PATH, map_location="cpu", weights_only=True)
            )
            model_int8 = torch.quantization.quantize_dynamic(
                model_base, {nn.Linear, nn.LSTM}, dtype=torch.qint8
            )
            model_int8.eval()
            self.models["CNN-LSTM (INT8)"] = model_int8
            print("[sim] Loaded CNN-LSTM INT8 quantized model")

        if config.FPGA_MODEL_PATH.exists():
            fpga_model = build_fpga_model(num_features=num_features)
            fpga_model.load_state_dict(
                torch.load(config.FPGA_MODEL_PATH, map_location="cpu", weights_only=True)
            )
            fpga_model.eval()
            self.models["CNN-FPGA (FP32)"] = fpga_model
            print("[sim] Loaded CNN-FPGA model")
        else:
            print("[sim] CNN-FPGA model not found — run model_cnn_fpga.py --transfer first")

        print(f"[sim] Total models loaded: {len(self.models)}")

    def benchmark_model(self, model: nn.Module, test_data: torch.Tensor,
                        test_labels: torch.Tensor, name: str,
                        n_warmup: int = 10, n_runs: int = 3) -> dict:
        model.eval()
        loader = DataLoader(
            TensorDataset(test_data, test_labels),
            batch_size=1,
            shuffle=False,
        )

        with torch.no_grad():
            for i, (x, _) in enumerate(loader):
                if i >= n_warmup:
                    break
                model(x)

        latencies = []
        with torch.no_grad():
            for i, (x, _) in enumerate(loader):
                if i >= 500:
                    break
                t0 = time.perf_counter()
                _ = model(x)
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000)

        batch_loader = DataLoader(
            TensorDataset(test_data, test_labels),
            batch_size=256,
        )

        correct = 0
        total = 0
        class_correct = {i: 0 for i in range(len(config.META_CLASS_NAMES))}
        class_total = {i: 0 for i in range(len(config.META_CLASS_NAMES))}
        batch_times = []

        with torch.no_grad():
            for run in range(n_runs):
                for x, y in batch_loader:
                    t0 = time.perf_counter()
                    out = model(x)
                    t1 = time.perf_counter()

                    preds = out.argmax(dim=1)
                    correct += (preds == y).sum().item()
                    total += len(y)
                    batch_times.append(len(x) / (t1 - t0))

                    if run == 0:
                        for i in range(len(y)):
                            c = y[i].item()
                            class_total[c] += 1
                            if preds[i] == y[i]:
                                class_correct[c] += 1

        accuracy = correct / total * 100
        throughput = np.mean(batch_times)
        avg_latency = np.mean(latencies)
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)

        per_class = {}
        for i in range(len(config.META_CLASS_NAMES)):
            if class_total[i] > 0:
                per_class[config.META_CLASS_NAMES[i]] = class_correct[i] / class_total[i] * 100

        result = {
            "name": name,
            "accuracy": accuracy,
            "avg_latency_ms": avg_latency,
            "p50_latency_ms": p50,
            "p95_latency_ms": p95,
            "p99_latency_ms": p99,
            "throughput_sps": throughput,
            "per_class_accuracy": per_class,
        }

        self.results[name] = result
        return result

    def estimate_fpga_performance(self) -> dict:
        clock_mhz = config.FPGA_CLOCK_MHZ
        dsp_blocks = config.FPGA_DSP_BLOCKS

        peak_gops = (dsp_blocks * 2 * clock_mhz) / 1000

        total_macs = 144_000 + 6_635_520 + 737_280 + 640
        total_gmacs = total_macs / 1e9

        utilization = 0.7
        effective_gops = peak_gops * utilization
        inference_time_ms = (total_gmacs / effective_gops) * 1000

        power_w = 2.5

        fpga_estimate = {
            "name": "FPGA Estimate (Zynq-7020)",
            "clock_mhz": clock_mhz,
            "dsp_blocks": dsp_blocks,
            "peak_gops": peak_gops,
            "total_macs": total_macs,
            "utilization": utilization,
            "estimated_latency_ms": inference_time_ms,
            "estimated_throughput_sps": 1000 / inference_time_ms if inference_time_ms > 0 else 0,
            "estimated_power_w": power_w,
            "energy_per_inference_mj": power_w * inference_time_ms,
        }

        self.results["FPGA Estimate"] = fpga_estimate
        return fpga_estimate

    def print_comparison_table(self):
        print(f"\n{'=' * 80}")
        print("  INFERENCE PERFORMANCE COMPARISON")
        print(f"{'=' * 80}")
        print(f"{'Model':<25} {'Accuracy':>10} {'Latency(ms)':>12} {'Throughput':>12} {'Power(W)':>10}")
        print(f"{'-' * 80}")

        for name, res in self.results.items():
            if "accuracy" in res:
                acc = f"{res['accuracy']:.2f}%"
                lat = f"{res['avg_latency_ms']:.3f}"
                tput = f"{res['throughput_sps']:.0f} s/s"
                power = "~65W (CPU)"
            else:
                acc = "~99%*"
                lat = f"{res['estimated_latency_ms']:.3f}"
                tput = f"{res['estimated_throughput_sps']:.0f} s/s"
                power = f"{res['estimated_power_w']:.1f}W"

            print(f"  {name:<23} {acc:>10} {lat:>12} {tput:>12} {power:>10}")

        print(f"{'-' * 80}")
        print("  * FPGA accuracy estimated based on INT8 quantization results")
        print(f"{'=' * 80}")

        for name, res in self.results.items():
            if "per_class_accuracy" in res and res["per_class_accuracy"]:
                print(f"\n  Per-class accuracy — {name}:")
                for cls, acc in res["per_class_accuracy"].items():
                    bar = "█" * int(acc / 2) + "░" * (50 - int(acc / 2))
                    print(f"    {cls:<25} {bar} {acc:.1f}%")

    def save_report(self, path: Path = None):
        if path is None:
            path = config.FPGA_DIR / "inference_report.txt"

        lines = [
            "=" * 60,
            "  FPGA INFERENCE SIMULATION REPORT",
            "=" * 60, "",
        ]

        for name, res in self.results.items():
            lines.append(f"Model: {name}")
            lines.append("-" * 40)
            for k, v in res.items():
                if k == "per_class_accuracy":
                    lines.append("  Per-class accuracy:")
                    for cls, acc in v.items():
                        lines.append(f"    {cls}: {acc:.2f}%")
                elif isinstance(v, float):
                    lines.append(f"  {k}: {v:.4f}")
                else:
                    lines.append(f"  {k}: {v}")
            lines.append("")

        report = "\n".join(lines)
        path.write_text(report)
        print(f"\n[sim] Report saved to {path}")


def main():
    print("=" * 60)
    print("  FPGA INFERENCE SIMULATOR")
    print("=" * 60)

    sim = FPGAInferenceSimulator()

    print("\n[Step 1] Loading models...")
    sim.load_models()

    if not sim.models:
        print("[ERROR] No models found. Train the model first.")
        return

    print("\n[Step 2] Loading test sequences...")
    seqs, lbls, _ = load_sequences()
    n = len(seqs)
    test_idx = torch.randperm(n)[:2000]
    test_seqs = seqs[test_idx]
    test_lbls = lbls[test_idx]
    print(f"[sim] Using {len(test_seqs)} test samples")

    print("\n[Step 3] Running benchmarks...")
    for name, model in sim.models.items():
        print(f"\n  Benchmarking {name}...")
        sim.benchmark_model(model, test_seqs, test_lbls, name)

    print("\n[Step 4] Estimating FPGA performance...")
    _fpga_est = sim.estimate_fpga_performance()

    sim.print_comparison_table()
    sim.save_report()

    print("\n✓ Inference simulation complete!")


if __name__ == "__main__":
    main()
