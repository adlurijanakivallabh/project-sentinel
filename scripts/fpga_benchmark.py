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
from src.model_cnn_fpga import build_fpga_model, estimate_fpga_resources, count_parameters
from src.sessionization import load_sequences


def measure_latency(model, data, n_samples=500, n_warmup=50):
    model.eval()
    latencies = []

    with torch.no_grad():
        for i in range(min(n_warmup, len(data))):
            model(data[i:i+1])

        for i in range(min(n_samples, len(data))):
            t0 = time.perf_counter()
            model(data[i:i+1])
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

    return np.array(latencies)


def measure_throughput(model, data, labels, batch_size=256, n_runs=5):
    model.eval()
    loader = DataLoader(TensorDataset(data, labels), batch_size=batch_size)
    throughputs = []
    accuracies = []

    with torch.no_grad():
        for run in range(n_runs):
            total_time = 0
            total_samples = 0
            correct = 0

            for x, y in loader:
                t0 = time.perf_counter()
                out = model(x)
                t1 = time.perf_counter()

                total_time += (t1 - t0)
                total_samples += len(x)
                correct += (out.argmax(1) == y).sum().item()

            throughputs.append(total_samples / total_time)
            accuracies.append(correct / total_samples * 100)

    return {
        "throughput_mean": np.mean(throughputs),
        "throughput_std": np.std(throughputs),
        "accuracy_mean": np.mean(accuracies),
        "accuracy_std": np.std(accuracies),
    }


def get_model_size(model):
    tmp = config.FPGA_DIR / "_benchmark_tmp.pth"
    torch.save(model.state_dict(), tmp)
    size = tmp.stat().st_size / 1024
    tmp.unlink()
    return size


def estimate_fpga_latency():
    clock_mhz = config.FPGA_CLOCK_MHZ
    dsp_blocks = config.FPGA_DSP_BLOCKS

    total_macs = 144_000 + 6_635_520 + 737_280 + 640

    macs_per_cycle = dsp_blocks * 2
    cycles_needed = total_macs / macs_per_cycle

    pipeline_stages = 4
    effective_cycles = cycles_needed / pipeline_stages

    latency_us = effective_cycles / clock_mhz
    latency_ms = latency_us / 1000

    return {
        "total_macs": total_macs,
        "macs_per_cycle": macs_per_cycle,
        "cycles_needed": int(cycles_needed),
        "pipeline_stages": pipeline_stages,
        "effective_cycles": int(effective_cycles),
        "latency_us": latency_us,
        "latency_ms": latency_ms,
        "throughput_sps": 1_000_000 / latency_us if latency_us > 0 else 0,
    }


def generate_benchmark_chart(results: dict):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[benchmark] matplotlib not available — skipping chart generation")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("FPGA Deployment Benchmark — CNN-LSTM IPS", fontsize=14, fontweight="bold")

    names = list(results.keys())
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6"][:len(names)]

    ax = axes[0]
    latencies = [results[n].get("latency_ms", 0) for n in names]
    bars = ax.bar(names, latencies, color=colors, edgecolor="white", linewidth=1.5)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Inference Latency\n(lower is better)")
    for bar, val in zip(bars, latencies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylim(0, max(latencies) * 1.3)

    ax = axes[1]
    throughputs = [results[n].get("throughput_sps", 0) for n in names]
    bars = ax.bar(names, throughputs, color=colors, edgecolor="white", linewidth=1.5)
    ax.set_ylabel("Samples / Second")
    ax.set_title("Throughput\n(higher is better)")
    for bar, val in zip(bars, throughputs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f"{val:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylim(0, max(throughputs) * 1.3)

    ax = axes[2]
    sizes = [results[n].get("model_size_kb", 0) for n in names]
    bars = ax.bar(names, sizes, color=colors, edgecolor="white", linewidth=1.5)
    ax.set_ylabel("Model Size (KB)")
    ax.set_title("Model Size\n(smaller is better)")
    for bar, val in zip(bars, sizes):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylim(0, max(sizes) * 1.3)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    chart_path = config.FPGA_DIR / "benchmark_chart.png"
    plt.savefig(str(chart_path), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[benchmark] Chart saved to {chart_path}")


def generate_report(results: dict, fpga_resources: dict):
    report_path = config.FPGA_BENCHMARK_PATH

    lines = [
        "=" * 65,
        "  FPGA DEPLOYMENT BENCHMARK REPORT",
        "  CNN-LSTM Intrusion Prevention System",
        "=" * 65,
        "",
        "PLATFORM COMPARISON",
        "-" * 65,
        f"{'Metric':<25} ",
    ]

    names = list(results.keys())
    header = f"{'Metric':<25}"
    for n in names:
        header += f" {n:>14}"
    lines.append(header)
    lines.append("-" * 65)

    metrics = [
        ("Accuracy (%)", "accuracy", ".2f"),
        ("Latency (ms)", "latency_ms", ".3f"),
        ("Throughput (s/s)", "throughput_sps", ".0f"),
        ("Model Size (KB)", "model_size_kb", ".0f"),
        ("Power (W)", "power_w", ".1f"),
    ]

    for label, key, fmt in metrics:
        row = f"  {label:<23}"
        for n in names:
            val = results[n].get(key, "N/A")
            if isinstance(val, (int, float)):
                row += f" {val:>14{fmt}}"
            else:
                row += f" {str(val):>14}"
        lines.append(row)

    lines.extend([
        "",
        "-" * 65,
        "",
        "FPGA RESOURCE UTILIZATION (Zynq-7020 Estimate)",
        "-" * 65,
        f"  DSP Blocks : {fpga_resources.get('DSP_used', 0):>6} / {fpga_resources.get('DSP_available', 0):>6} "
        f"({fpga_resources.get('DSP_pct', 0):.1f}%)",
        f"  BRAM (KB)  : {fpga_resources.get('BRAM_used_KB', 0):>6.1f} / {fpga_resources.get('BRAM_available_KB', 0):>6.0f} "
        f"({fpga_resources.get('BRAM_pct', 0):.1f}%)",
        f"  LUTs       : {fpga_resources.get('LUT_used', 0):>6} / {fpga_resources.get('LUT_available', 0):>6} "
        f"({fpga_resources.get('LUT_pct', 0):.1f}%)",
        f"  Weight Size: {fpga_resources.get('weight_size_KB', 0):.1f} KB (INT8)",
        "",
        "-" * 65,
        "",
        "KEY FINDINGS",
        "-" * 65,
    ])

    cpu_lat = results.get("CPU FP32", {}).get("latency_ms", 1)
    fpga_lat = results.get("FPGA (est.)", {}).get("latency_ms", 1)
    if cpu_lat > 0 and fpga_lat > 0:
        speedup = cpu_lat / fpga_lat
        lines.append(f"  FPGA speedup vs CPU FP32: {speedup:.1f}x faster")

    int8_lat = results.get("CPU INT8", {}).get("latency_ms", 1)
    if int8_lat > 0 and fpga_lat > 0:
        speedup = int8_lat / fpga_lat
        lines.append(f"  FPGA speedup vs CPU INT8: {speedup:.1f}x faster")

    cpu_power = results.get("CPU FP32", {}).get("power_w", 65)
    fpga_power = results.get("FPGA (est.)", {}).get("power_w", 2.5)
    if fpga_power > 0:
        power_saving = cpu_power / fpga_power
        lines.append(f"  Power efficiency vs CPU : {power_saving:.0f}x less power")

    cpu_acc = results.get("CPU FP32", {}).get("accuracy", 0)
    int8_acc = results.get("CPU INT8", {}).get("accuracy", 0)
    lines.append(f"  INT8 accuracy drop      : {cpu_acc - int8_acc:.2f}%")

    lines.extend([
        "",
        "-" * 65,
        "",
        "DEPLOYMENT RECOMMENDATION",
        "-" * 65,
        "  The CNN-LSTM model with INT8 quantization is suitable for",
        "  FPGA deployment on Xilinx Zynq-7020 or similar platforms.",
        "  For optimal FPGA performance, use the CNN-FPGA variant",
        "  (temporal Conv1D instead of LSTM) which eliminates",
        "  recurrent dependencies and maps efficiently to DSP blocks.",
        "",
        "=" * 65,
    ])

    report = "\n".join(lines)
    report_path.write_text(report)
    print(f"\n{report}")
    print(f"\n[benchmark] Report saved to {report_path}")


def main():
    print("=" * 60)
    print("  FPGA DEPLOYMENT BENCHMARK SUITE")
    print("=" * 60)

    num_features = len(config.FEATURE_COLUMNS)

    print("\n[Step 1] Loading test data...")
    seqs, lbls, _ = load_sequences()
    idx = torch.randperm(len(seqs))[:3000]
    test_seqs, test_lbls = seqs[idx], lbls[idx]
    print(f"  Using {len(test_seqs)} test samples")

    results = {}

    print("\n[Step 2] Benchmarking CPU FP32...")
    model_fp32 = build_model(num_features=num_features)
    model_fp32.load_state_dict(
        torch.load(config.MODEL_CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    )
    model_fp32.eval()

    lat_fp32 = measure_latency(model_fp32, test_seqs)
    tp_fp32 = measure_throughput(model_fp32, test_seqs, test_lbls)

    results["CPU FP32"] = {
        "accuracy": tp_fp32["accuracy_mean"],
        "latency_ms": np.mean(lat_fp32),
        "p50_ms": np.percentile(lat_fp32, 50),
        "p95_ms": np.percentile(lat_fp32, 95),
        "p99_ms": np.percentile(lat_fp32, 99),
        "throughput_sps": tp_fp32["throughput_mean"],
        "model_size_kb": get_model_size(model_fp32),
        "power_w": 65.0,
    }
    print(f"  Accuracy: {results['CPU FP32']['accuracy']:.2f}%")
    print(f"  Latency : {results['CPU FP32']['latency_ms']:.3f} ms (avg)")
    print(f"  Throughput: {results['CPU FP32']['throughput_sps']:.0f} samples/sec")

    print("\n[Step 3] Benchmarking CPU INT8...")
    model_int8 = torch.quantization.quantize_dynamic(
        model_fp32, {nn.Linear, nn.LSTM}, dtype=torch.qint8
    )
    model_int8.eval()

    lat_int8 = measure_latency(model_int8, test_seqs)
    tp_int8 = measure_throughput(model_int8, test_seqs, test_lbls)

    results["CPU INT8"] = {
        "accuracy": tp_int8["accuracy_mean"],
        "latency_ms": np.mean(lat_int8),
        "p50_ms": np.percentile(lat_int8, 50),
        "p95_ms": np.percentile(lat_int8, 95),
        "p99_ms": np.percentile(lat_int8, 99),
        "throughput_sps": tp_int8["throughput_mean"],
        "model_size_kb": get_model_size(model_int8),
        "power_w": 65.0,
    }
    print(f"  Accuracy: {results['CPU INT8']['accuracy']:.2f}%")
    print(f"  Latency : {results['CPU INT8']['latency_ms']:.3f} ms (avg)")
    print(f"  Throughput: {results['CPU INT8']['throughput_sps']:.0f} samples/sec")

    print("\n[Step 4] Estimating FPGA performance...")
    fpga_lat = estimate_fpga_latency()

    fpga_model = build_fpga_model(num_features=num_features)
    fpga_resources = estimate_fpga_resources(fpga_model)
    fpga_params = count_parameters(fpga_model)

    results["FPGA (est.)"] = {
        "accuracy": results["CPU INT8"]["accuracy"],
        "latency_ms": fpga_lat["latency_ms"],
        "throughput_sps": fpga_lat["throughput_sps"],
        "model_size_kb": fpga_params["total"] * 1 / 1024,
        "power_w": 2.5,
    }
    print(f"  Estimated latency   : {fpga_lat['latency_ms']:.3f} ms")
    print(f"  Estimated throughput: {fpga_lat['throughput_sps']:.0f} samples/sec")
    print(f"  Total MACs          : {fpga_lat['total_macs']:,}")
    print("  Estimated power     : 2.5W")

    print("\n[Step 5] Generating report and charts...")
    generate_report(results, fpga_resources)
    generate_benchmark_chart(results)

    print("\n✓ Benchmark complete!")
    print(f"  Report: {config.FPGA_BENCHMARK_PATH}")
    print(f"  Chart : {config.FPGA_DIR / 'benchmark_chart.png'}")


if __name__ == "__main__":
    main()
