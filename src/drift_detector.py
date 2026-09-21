import sys
from pathlib import Path
import numpy as np
from collections import deque
from datetime import datetime

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src import config


class DriftDetector:

    def __init__(self, n_bins: int = 50, kl_threshold: float = None,
                 psi_threshold: float = None):
        self.n_bins = n_bins
        self.kl_threshold = kl_threshold or config.DRIFT_KL_THRESHOLD
        self.psi_threshold = psi_threshold or config.DRIFT_PSI_THRESHOLD

        self.baseline_hists = {}
        self.feature_names = list(config.FEATURE_COLUMNS)
        if config.TEMPORAL_FEATURES_ENABLED:
            self.feature_names += config.TEMPORAL_FEATURE_NAMES

        self.drift_history = deque(maxlen=1000)

    def set_baseline(self, X: np.ndarray):
        n_features = X.shape[1]
        for i in range(n_features):
            col = X[:, i]
            col = col[np.isfinite(col)]
            if len(col) < 10:
                continue
            hist, bin_edges = np.histogram(col, bins=self.n_bins, density=True)
            hist = hist + 1e-10
            self.baseline_hists[i] = (hist, bin_edges)

        print(f"[drift] Baseline set from {X.shape[0]:,} samples, "
              f"{len(self.baseline_hists)} features")

    def _kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        p = np.clip(p, 1e-10, None)
        q = np.clip(q, 1e-10, None)
        p = p / p.sum()
        q = q / q.sum()
        return float(np.sum(p * np.log(p / q)))

    def _psi(self, expected: np.ndarray, actual: np.ndarray) -> float:
        expected = np.clip(expected / expected.sum(), 1e-10, None)
        actual = np.clip(actual / actual.sum(), 1e-10, None)
        return float(np.sum((actual - expected) * np.log(actual / expected)))

    def check(self, X_new: np.ndarray) -> dict:
        if not self.baseline_hists:
            return {"status": "no_baseline", "message": "Baseline not set"}

        n_features = min(X_new.shape[1], len(self.baseline_hists))
        feature_results = []
        kl_values = []
        psi_values = []

        for i in range(n_features):
            if i not in self.baseline_hists:
                continue

            base_hist, bin_edges = self.baseline_hists[i]
            col = X_new[:, i]
            col = col[np.isfinite(col)]

            if len(col) < 5:
                continue

            new_hist, _ = np.histogram(col, bins=bin_edges, density=True)
            new_hist = new_hist + 1e-10

            kl = self._kl_divergence(new_hist, base_hist)
            psi = self._psi(base_hist, new_hist)

            name = self.feature_names[i] if i < len(self.feature_names) else f"Feature_{i}"

            feature_results.append({
                "feature": name,
                "feature_idx": i,
                "kl_divergence": kl,
                "psi": psi,
                "kl_alert": kl > self.kl_threshold,
                "psi_alert": psi > self.psi_threshold,
            })

            kl_values.append(kl)
            psi_values.append(psi)

        n_kl_alerts = sum(1 for r in feature_results if r["kl_alert"])
        n_psi_alerts = sum(1 for r in feature_results if r["psi_alert"])
        total_features = len(feature_results)

        is_critical = (n_kl_alerts / max(total_features, 1)) > 0.25

        report = {
            "timestamp": datetime.now().isoformat(),
            "status": "DRIFT_CRITICAL" if is_critical else (
                "DRIFT_WARNING" if n_kl_alerts > 0 else "STABLE"
            ),
            "n_samples": len(X_new),
            "n_features_checked": total_features,
            "n_kl_alerts": n_kl_alerts,
            "n_psi_alerts": n_psi_alerts,
            "mean_kl": float(np.mean(kl_values)) if kl_values else 0,
            "mean_psi": float(np.mean(psi_values)) if psi_values else 0,
            "max_kl": float(np.max(kl_values)) if kl_values else 0,
            "max_psi": float(np.max(psi_values)) if psi_values else 0,
            "feature_details": feature_results,
        }

        self.drift_history.append(report)
        return report

    def print_report(self, report: dict):
        status = report["status"]
        colors = {
            "STABLE": "\033[92m",
            "DRIFT_WARNING": "\033[93m",
            "DRIFT_CRITICAL": "\033[91m",
        }
        color = colors.get(status, "\033[0m")
        reset = "\033[0m"

        print("\n  -- Drift Detection --")
        print(f"  Status:            {color}{status}{reset}")
        print(f"  Samples checked:   {report['n_samples']}")
        print(f"  Features checked:  {report['n_features_checked']}")
        print(f"  KL alerts:         {report['n_kl_alerts']}")
        print(f"  PSI alerts:        {report['n_psi_alerts']}")
        print(f"  Mean KL:           {report['mean_kl']:.4f} (threshold: {self.kl_threshold})")
        print(f"  Mean PSI:          {report['mean_psi']:.4f} (threshold: {self.psi_threshold})")

        drifted = [f for f in report["feature_details"]
                   if f["kl_alert"] or f["psi_alert"]]
        if drifted:
            print("\n  Drifted Features:")
            drifted.sort(key=lambda x: x["kl_divergence"], reverse=True)
            for f in drifted[:10]:
                kl_flag = "!" if f["kl_alert"] else " "
                psi_flag = "!" if f["psi_alert"] else " "
                print(f"    {f['feature']:>30s}: KL={f['kl_divergence']:.4f} {kl_flag}  "
                      f"PSI={f['psi']:.4f} {psi_flag}")


if __name__ == "__main__":
    np.random.seed(42)
    baseline = np.random.randn(1000, 25)
    normal_new = np.random.randn(200, 25)
    drifted_new = np.random.randn(200, 25) + 2

    detector = DriftDetector()
    detector.set_baseline(baseline)

    print("=== No Drift ===")
    report = detector.check(normal_new)
    detector.print_report(report)

    print("\n=== With Drift ===")
    report = detector.check(drifted_new)
    detector.print_report(report)
