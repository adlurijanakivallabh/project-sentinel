import sys
import time
import json
import argparse
import numpy as np
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src import config
from src.train import get_model, ALL_VARIANTS


BATCH_SIZES = [1, 4, 16, 64, 128, 256, 512, 1024]
WARMUP_ITERS = 20
BENCHMARK_ITERS = 50
NUM_REPETITIONS = 3
AVG_BYTES_PER_FLOW = 400
FLOWS_PER_WINDOW = config.SEQUENCE_LENGTH


def _load_model(variant: str, device: torch.device, use_swa: bool = True):
    num_features = len(config.FEATURE_COLUMNS)
    if config.TEMPORAL_FEATURES_ENABLED:
        num_features += len(config.TEMPORAL_FEATURE_NAMES)

    model = get_model(variant, num_features=num_features, num_classes=config.NUM_CLASSES)

    if variant == "model_v2":
        swa_path = config.MODELS_DIR / "best_model_v2_swa.pth"
        best_path = config.V2_MODEL_SAVE_PATH
        ckpt_path = swa_path if (use_swa and swa_path.exists()) else best_path
    else:
        swa_path = config.MODELS_DIR / f"best_{variant}_swa.pth"
        best_path = config.MODELS_DIR / f"best_{variant}.pth"
        ckpt_path = swa_path if (use_swa and swa_path.exists()) else best_path

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)
        print(f"  Loaded checkpoint: {ckpt_path.name}")
    else:
        print(f"  No checkpoint found — benchmarking fresh model (weights are random)")

    model.to(device)
    model.eval()
    return model


def _benchmark_batch(model, batch_size: int, num_features: int,
                     device: torch.device, use_cuda_events: bool):
    seq_len = config.SEQUENCE_LENGTH
    dummy = torch.randn(batch_size, seq_len, num_features, device=device)

    with torch.inference_mode():
        for _ in range(WARMUP_ITERS):
            _ = model(dummy)
    if use_cuda_events:
        torch.cuda.synchronize()

    timings = []
    for _ in range(NUM_REPETITIONS):
        if use_cuda_events:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            torch.cuda.synchronize()
            start_event.record()
            with torch.inference_mode():
                for _ in range(BENCHMARK_ITERS):
                    _ = model(dummy)
            end_event.record()
            torch.cuda.synchronize()

            elapsed_ms = start_event.elapsed_time(end_event)
        else:
            start = time.perf_counter()
            with torch.inference_mode():
                for _ in range(BENCHMARK_ITERS):
                    _ = model(dummy)
            elapsed_ms = (time.perf_counter() - start) * 1000.0

        ms_per_batch = elapsed_ms / BENCHMARK_ITERS
        timings.append(ms_per_batch)

    return np.mean(timings), np.std(timings)


def benchmark_model(variant: str, device_name: str = "auto"):
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    use_cuda_events = device.type == "cuda"

    print(f"\n{'=' * 80}")
    print(f"  THROUGHPUT BENCHMARK: {variant}")
    print(f"{'=' * 80}")
    print(f"  Device:           {device}")
    if device.type == "cuda":
        print(f"  GPU:              {torch.cuda.get_device_name(0)}")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU Memory:       {gpu_mem:.1f} GB")
    print(f"  Timing method:    {'CUDA Events (accurate)' if use_cuda_events else 'CPU time.perf_counter'}")
    print(f"  Warmup:           {WARMUP_ITERS} iterations")
    print(f"  Benchmark:        {BENCHMARK_ITERS} iterations × {NUM_REPETITIONS} repetitions")
    print(f"  Sequence length:  {config.SEQUENCE_LENGTH} flows/window")

    model = _load_model(variant, device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters:       {param_count:,}")

    num_features = len(config.FEATURE_COLUMNS)
    if config.TEMPORAL_FEATURES_ENABLED:
        num_features += len(config.TEMPORAL_FEATURE_NAMES)
    print(f"  Features:         {num_features}")

    results = []
    print(f"\n  {'Batch':>6}  {'ms/batch':>10}  {'±std':>8}  {'Win/sec':>12}  "
          f"{'Flows/sec':>14}  {'Gbps':>8}  {'ms/window':>10}")
    print(f"  {'-' * 82}")

    for bs in BATCH_SIZES:
        try:
            mean_ms, std_ms = _benchmark_batch(model, bs, num_features, device, use_cuda_events)

            batches_per_sec = 1000.0 / mean_ms
            windows_per_sec = batches_per_sec * bs
            flows_per_sec = windows_per_sec * FLOWS_PER_WINDOW
            gbps = flows_per_sec * AVG_BYTES_PER_FLOW * 8 / 1e9
            ms_per_window = mean_ms / bs

            result = {
                "batch_size": bs,
                "ms_per_batch": round(mean_ms, 4),
                "std_ms": round(std_ms, 4),
                "windows_per_sec": round(windows_per_sec, 1),
                "flows_per_sec": round(flows_per_sec, 1),
                "gbps_estimated": round(gbps, 3),
                "ms_per_window": round(ms_per_window, 4),
            }
            results.append(result)

            print(f"  {bs:>6}  {mean_ms:>10.3f}  {std_ms:>7.3f}  "
                  f"{windows_per_sec:>12,.0f}  {flows_per_sec:>14,.0f}  "
                  f"{gbps:>8.3f}  {ms_per_window:>10.4f}", flush=True)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  {bs:>6}  {'OOM':>10}  {'—':>8}  {'—':>12}  {'—':>14}  {'—':>8}  {'—':>10}")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                break
            else:
                raise

    if results:
        peak = max(results, key=lambda r: r["windows_per_sec"])
        print(f"\n  +-----------------------------------------------------+")
        print(f"  |  PEAK: {peak['windows_per_sec']:,.0f} windows/sec "
              f"@ batch={peak['batch_size']}  |")
        print(f"  |        {peak['flows_per_sec']:,.0f} flows/sec "
              f"= {peak['gbps_estimated']:.2f} Gbps est.    |")
        print(f"  |        {peak['ms_per_window']:.4f} ms/window latency"
              f"                   |")
        print(f"  +-----------------------------------------------------+")

    return {
        "model": variant,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "params": param_count,
        "num_features": num_features,
        "sequence_length": config.SEQUENCE_LENGTH,
        "avg_bytes_per_flow": AVG_BYTES_PER_FLOW,
        "results": results,
        "peak": peak if results else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Throughput benchmark for Sentinel IPS models")
    parser.add_argument("--model", type=str, default="model_v2",
                        choices=ALL_VARIANTS,
                        help="Model variant to benchmark (default: model_v2)")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Device to benchmark on (default: auto)")
    parser.add_argument("--all", action="store_true",
                        help="Benchmark ALL model variants")
    args = parser.parse_args()

    all_results = {}

    if args.all:
        for variant in ALL_VARIANTS:
            try:
                result = benchmark_model(variant, args.device)
                all_results[variant] = result
            except Exception as e:
                print(f"\n  ERROR benchmarking {variant}: {e}")
                all_results[variant] = {"error": str(e)}
    else:
        result = benchmark_model(args.model, args.device)
        all_results[args.model] = result

    if len(all_results) > 1:
        print(f"\n\n{'=' * 80}")
        print(f"  ALL MODELS COMPARISON (peak throughput)")
        print(f"{'=' * 80}")
        print(f"  {'Model':<30}  {'Params':>10}  {'Win/sec':>12}  "
              f"{'Gbps':>8}  {'ms/win':>8}")
        print(f"  {'-' * 72}")

        for variant, data in all_results.items():
            if "error" in data:
                print(f"  {variant:<30}  {'ERROR':>10}")
                continue
            peak = data.get("peak", {})
            if peak:
                print(f"  {variant:<30}  {data['params']:>10,}  "
                      f"{peak['windows_per_sec']:>12,.0f}  "
                      f"{peak['gbps_estimated']:>8.3f}  "
                      f"{peak['ms_per_window']:>8.4f}")

    out_path = config.EXPERIMENTS_DIR / "throughput_benchmark.json"
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "benchmarks": all_results,
        }, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
