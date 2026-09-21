import time
import numpy as np

import onnxruntime as ort


CLASS_NAMES = [
    "Normal", "DoS/DDoS", "PortScan/Recon", "Web/Injection",
    "Brute Force", "Botnet/C2", "Malware/Exploit", "Infiltration"
]
SEQUENCE_LENGTH = 20
NUM_FEATURES = 30


class SentinelPYNQ:

    def __init__(self, onnx_path="sentinel_v2.onnx", zscore_path="zscore_stats.npz"):
        print("[*] Loading ONNX model...")
        self.session = ort.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
        print(f"[OK] Model loaded: {onnx_path}")

        stats = np.load(zscore_path)
        self.zscore_mean = stats['mean']
        self.zscore_std = np.where(stats['std'] < 1e-8, 1.0, stats['std'])
        print(f"[OK] Z-score stats loaded: {zscore_path}")

    def classify(self, window):
        normalized = (window - self.zscore_mean) / self.zscore_std
        normalized = np.clip(normalized, -10, 10)

        input_data = normalized.astype(np.float32).reshape(1, SEQUENCE_LENGTH, NUM_FEATURES)

        t0 = time.perf_counter()
        output = self.session.run(None, {self.input_name: input_data})[0]
        t1 = time.perf_counter()

        pred_class = int(np.argmax(output[0]))

        logits = output[0] / 0.5
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        confidence = float(probs[pred_class]) * 100

        return {
            'class': CLASS_NAMES[pred_class],
            'class_id': pred_class,
            'confidence': round(confidence, 1),
            'latency_ms': round((t1 - t0) * 1000, 2),
            'all_probs': {CLASS_NAMES[i]: round(float(probs[i]) * 100, 1) for i in range(8)}
        }

    def classify_batch(self, windows):
        results = []
        for w in windows:
            results.append(self.classify(w))
        return results

    def benchmark(self, n_iterations=100):
        print(f"\n[*] Running {n_iterations} inference iterations...")
        dummy = np.random.randn(SEQUENCE_LENGTH, NUM_FEATURES).astype(np.float32)

        for _ in range(10):
            self.classify(dummy)

        latencies = []
        for _ in range(n_iterations):
            result = self.classify(dummy)
            latencies.append(result['latency_ms'])

        print(f"[OK] Benchmark complete:")
        print(f"     Mean latency : {np.mean(latencies):.2f} ms")
        print(f"     Median       : {np.median(latencies):.2f} ms")
        print(f"     Min / Max    : {np.min(latencies):.2f} / {np.max(latencies):.2f} ms")
        print(f"     Throughput   : {1000 / np.mean(latencies):.0f} windows/sec")
        return latencies


if __name__ == "__main__":
    ips = SentinelPYNQ()
    test = np.random.randn(SEQUENCE_LENGTH, NUM_FEATURES).astype(np.float32)
    result = ips.classify(test)
    print(f"\nTest result: {result['class']} ({result['confidence']}%) in {result['latency_ms']} ms")
    ips.benchmark(n_iterations=50)
