import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import threading

import numpy as np
import torch
import torch.nn as nn

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src import config
from src.model_cnn_lstm import build_model


class FlowTracker:

    def __init__(self, timeout: float = None, idle: float = None):
        self.timeout = timeout or config.FLOW_TIMEOUT_SEC
        self.idle = idle or config.FLOW_IDLE_SEC
        self.flows = {}
        self.completed_flows = []
        self.lock = threading.Lock()

    def _flow_key(self, pkt_info: dict) -> tuple:
        return (
            pkt_info.get("src_ip", ""),
            pkt_info.get("dst_ip", ""),
            pkt_info.get("src_port", 0),
            pkt_info.get("dst_port", 0),
            pkt_info.get("protocol", 0),
        )

    def process_packet(self, pkt_info: dict):
        key = self._flow_key(pkt_info)
        now = time.time()

        with self.lock:
            if key not in self.flows:
                self.flows[key] = {
                    "start_time": now,
                    "last_time": now,
                    "dst_port": pkt_info.get("dst_port", 0),
                    "protocol": pkt_info.get("protocol", 0),
                    "fwd_packets": [],
                    "bwd_packets": [],
                    "fwd_iats": [],
                    "bwd_iats": [],
                    "fwd_psh": 0,
                    "bwd_psh": 0,
                    "fwd_urg": 0,
                    "bwd_urg": 0,
                    "last_fwd_time": now,
                    "last_bwd_time": now,
                    "is_forward": True,
                }

            flow = self.flows[key]
            pkt_size = pkt_info.get("size", 0)
            is_fwd = pkt_info.get("is_forward", True)

            if is_fwd:
                flow["fwd_packets"].append(pkt_size)
                if flow["fwd_packets"]:
                    flow["fwd_iats"].append(now - flow["last_fwd_time"])
                flow["last_fwd_time"] = now
                if pkt_info.get("psh", False):
                    flow["fwd_psh"] += 1
                if pkt_info.get("urg", False):
                    flow["fwd_urg"] += 1
            else:
                flow["bwd_packets"].append(pkt_size)
                if flow["bwd_packets"]:
                    flow["bwd_iats"].append(now - flow["last_bwd_time"])
                flow["last_bwd_time"] = now
                if pkt_info.get("psh", False):
                    flow["bwd_psh"] += 1
                if pkt_info.get("urg", False):
                    flow["bwd_urg"] += 1

            flow["last_time"] = now

    def expire_flows(self):
        now = time.time()
        with self.lock:
            expired_keys = []
            for key, flow in self.flows.items():
                if (now - flow["start_time"] > self.timeout or
                        now - flow["last_time"] > self.idle):
                    features = self._extract_features(flow)
                    self.completed_flows.append(features)
                    expired_keys.append(key)

            for key in expired_keys:
                del self.flows[key]

    def _extract_features(self, flow: dict) -> np.ndarray:
        fwd = flow["fwd_packets"] or [0]
        bwd = flow["bwd_packets"] or [0]
        all_iats = flow.get("fwd_iats", []) + flow.get("bwd_iats", [])
        if not all_iats:
            all_iats = [0]

        duration = flow["last_time"] - flow["start_time"]

        features = np.array([
            float(flow["dst_port"]),
            float(flow["protocol"]),
            duration * 1e6,
            float(len(fwd)),
            float(len(bwd)),
            float(sum(fwd)),
            float(sum(bwd)),
            float(max(fwd)),
            float(min(fwd)),
            float(np.mean(fwd)),
            float(np.std(fwd)) if len(fwd) > 1 else 0.0,
            float(max(bwd)),
            float(min(bwd)),
            float(np.mean(bwd)),
            float(np.std(bwd)) if len(bwd) > 1 else 0.0,
            float(np.mean(all_iats)) * 1e6,
            float(np.std(all_iats)) * 1e6 if len(all_iats) > 1 else 0.0,
            float(max(all_iats)) * 1e6,
            float(min(all_iats)) * 1e6,
            float(flow["fwd_psh"]),
            float(flow["bwd_psh"]),
            float(flow["fwd_urg"]),
            float(flow["bwd_urg"]),
            sum(fwd + bwd) / duration if duration > 0 else 0,
            len(fwd + bwd) / duration if duration > 0 else 0,
        ], dtype=np.float32)

        return features

    def get_completed_flows(self) -> list:
        with self.lock:
            flows = self.completed_flows.copy()
            self.completed_flows.clear()
        return flows

    def force_close_all(self):
        with self.lock:
            for key, flow in self.flows.items():
                features = self._extract_features(flow)
                self.completed_flows.append(features)
            self.flows.clear()


class IPSEngine:

    def __init__(self, model: nn.Module, scaler=None):
        self.model = model
        self.model.eval()
        self.scaler = scaler
        self.flow_buffer = []
        self.alerts = []
        self.blocked_ips = {}
        self.last_window_time = time.time()
        self.stats = {
            "total_flows": 0,
            "total_windows": 0,
            "blocked": 0,
            "rate_limited": 0,
            "logged": 0,
            "allowed": 0,
            "detections": defaultdict(int),
        }

    def add_flow(self, features: np.ndarray, src_ip: str = None):
        self.flow_buffer.append({"features": features, "src_ip": src_ip})
        self.stats["total_flows"] += 1

    def _classify_severity(self, pred_id: int, confidence: float) -> str:
        if pred_id == 0:
            return "ALLOW"

        if confidence >= config.SEVERITY_BLOCK_THRESHOLD:
            return "BLOCK"
        elif confidence >= config.SEVERITY_RATE_LIMIT_THRESHOLD:
            return "RATE_LIMIT"
        elif confidence >= config.SEVERITY_LOG_THRESHOLD:
            return "LOG"
        else:
            return "ALLOW"

    def _should_trigger_window(self) -> bool:
        if len(self.flow_buffer) >= config.SEQUENCE_LENGTH:
            return True
        elapsed = time.time() - self.last_window_time
        if (elapsed >= config.WINDOW_TRIGGER_INTERVAL_SEC
                and len(self.flow_buffer) >= config.SEQUENCE_LENGTH // 2):
            return True
        return False

    def check_window(self) -> dict:
        if not self._should_trigger_window():
            return None

        n = min(len(self.flow_buffer), config.SEQUENCE_LENGTH)
        window_entries = self.flow_buffer[:n]
        self.flow_buffer = self.flow_buffer[config.SEQUENCE_STEP:]
        self.last_window_time = time.time()

        features_list = [e["features"] if isinstance(e, dict) else e
                         for e in window_entries]
        src_ips = [e.get("src_ip", "unknown") if isinstance(e, dict) else "unknown"
                   for e in window_entries]

        window = np.stack(features_list)
        if window.shape[0] < config.SEQUENCE_LENGTH:
            pad = np.zeros((config.SEQUENCE_LENGTH - window.shape[0], window.shape[1]),
                           dtype=np.float32)
            window = np.vstack([window, pad])

        if self.scaler is not None:
            window = self.scaler.transform(window)

        with torch.no_grad():
            x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
            t0 = time.perf_counter()
            logits = self.model(x)
            t1 = time.perf_counter()

            probs = torch.softmax(logits, dim=1)
            confidence, pred_class = probs.max(dim=1)

        pred_id = pred_class.item()
        class_names = config.get_class_names()
        pred_name = class_names[pred_id] if pred_id < len(class_names) else f"Class{pred_id}"
        conf = confidence.item()
        conf_pct = conf * 100
        latency_ms = (t1 - t0) * 1000

        action = self._classify_severity(pred_id, conf)

        ip_counts = defaultdict(int)
        for ip in src_ips:
            if ip:
                ip_counts[ip] += 1
        top_src_ip = max(ip_counts, key=ip_counts.get) if ip_counts else "unknown"

        result = {
            "timestamp": datetime.now().isoformat(),
            "prediction": pred_name,
            "class_id": pred_id,
            "confidence": conf_pct,
            "action": action,
            "severity": action,
            "latency_ms": latency_ms,
            "src_ip": top_src_ip,
            "window_size": n,
        }

        self.stats["total_windows"] += 1
        if action == "BLOCK":
            self.stats["blocked"] += 1
            self.stats["detections"][pred_name] += 1
            self.alerts.append(result)
            if top_src_ip != "unknown" and top_src_ip not in config.IP_WHITELIST:
                expire_time = time.time() + config.BLOCK_TIMEOUT_MINUTES * 60
                self.blocked_ips[top_src_ip] = expire_time
        elif action == "RATE_LIMIT":
            self.stats["rate_limited"] += 1
            self.alerts.append(result)
        elif action == "LOG":
            self.stats["logged"] += 1
            self.alerts.append(result)
        else:
            self.stats["allowed"] += 1

        self._expire_blocks()

        return result

    def _expire_blocks(self):
        now = time.time()
        expired = [ip for ip, t in self.blocked_ips.items() if t <= now]
        for ip in expired:
            del self.blocked_ips[ip]

    def print_result(self, result: dict):
        if result is None:
            return

        action = result["action"]
        colors = {
            "BLOCK": "\033[91m",
            "RATE_LIMIT": "\033[93m",
            "LOG": "\033[96m",
            "ALLOW": "\033[92m",
        }
        color_start = colors.get(action, "\033[0m")
        color_end = "\033[0m"

        ip_str = f" | IP: {result.get('src_ip', 'N/A'):>15s}" if result.get("src_ip") else ""

        print(f"  {result['timestamp']} | "
              f"{color_start}{action:10s}{color_end} | "
              f"{result['prediction']:<25s} | "
              f"Confidence: {result['confidence']:5.1f}%{ip_str} | "
              f"Latency: {result['latency_ms']:.2f}ms")

    def print_summary(self):
        s = self.stats
        print(f"\n{'=' * 70}")
        print("  IPS SESSION SUMMARY")
        print(f"{'=' * 70}")
        print(f"  Total flows processed  : {s['total_flows']}")
        print(f"  Total windows analyzed : {s['total_windows']}")
        print("  ── Actions ──")
        print(f"  \033[91mBLOCKED\033[0m              : {s['blocked']}")
        print(f"  \033[93mRATE_LIMITED\033[0m          : {s['rate_limited']}")
        print(f"  \033[96mLOGGED\033[0m               : {s['logged']}")
        print(f"  \033[92mALLOWED\033[0m              : {s['allowed']}")
        if s['total_windows'] > 0:
            threat_rate = (s['blocked'] + s['rate_limited']) / s['total_windows'] * 100
            print(f"  Threat rate            : {threat_rate:.1f}%")
        if s['detections']:
            print("\n  Detections by type:")
            for name, count in sorted(s['detections'].items(), key=lambda x: -x[1]):
                print(f"    {name:<25s}: {count}")
        if self.blocked_ips:
            print(f"\n  Currently blocked IPs: {len(self.blocked_ips)}")
            for ip, expire in self.blocked_ips.items():
                remaining = max(0, (expire - time.time()) / 60)
                print(f"    {ip:>15s} — expires in {remaining:.0f} min")
        print(f"{'=' * 70}\n")

    def save_alerts(self, path: Path = None):
        if path is None:
            path = config.FPGA_DIR / "capture_alerts.json"

        with open(path, "w") as f:
            json.dump(self.alerts, f, indent=2)
        print(f"[ips] {len(self.alerts)} alerts saved to {path}")


def run_simulation(model: nn.Module, duration: int = 30):
    from src.sessionization import load_sequences

    print(f"\n{'=' * 60}")
    print("  LIVE IPS SIMULATION (using test data)")
    print(f"{'=' * 60}")

    seqs, lbls, _ = load_sequences()
    n = min(len(seqs), 200)

    ips = IPSEngine(model)
    print(f"\n  Processing {n} test sequences...\n")
    print(f"  {'Timestamp':<26} | {'Action':5} | {'Classification':<25} | "
          f"{'Confidence':>12} | {'Latency':>10}")
    print(f"  {'-' * 90}")

    for i in range(n):
        seq = seqs[i].numpy()

        for row in seq:
            ips.add_flow(row)

        result = ips.check_window()
        if result:
            ips.print_result(result)

        time.sleep(0.05)

    ips.print_summary()
    ips.save_alerts()


def run_live_capture(model: nn.Module, interface: str = None,
                     duration: int = None):
    try:
        from scapy.all import sniff, IP, TCP, UDP
    except ImportError:
        print("[ERROR] Scapy not installed. Install with: pip install scapy")
        print("[INFO]  Falling back to simulation mode...")
        run_simulation(model, duration or 30)
        return

    if interface is None:
        interface = config.CAPTURE_INTERFACE
    if duration is None:
        duration = config.CAPTURE_DURATION_SEC

    print(f"\n{'=' * 60}")
    print("  LIVE PACKET CAPTURE IPS")
    print(f"{'=' * 60}")
    print(f"  Interface : {interface or 'auto-detect'}")
    print(f"  Duration  : {duration} seconds")
    print("  Model     : CNN-LSTM")
    print(f"{'=' * 60}\n")

    tracker = FlowTracker()
    ips = IPSEngine(model)

    packet_count = [0]

    def process_pkt(pkt):
        if not pkt.haslayer(IP):
            return

        ip = pkt[IP]
        pkt_info = {
            "src_ip": ip.src,
            "dst_ip": ip.dst,
            "protocol": ip.proto,
            "size": len(pkt),
            "is_forward": True,
            "psh": False,
            "urg": False,
        }

        if pkt.haslayer(TCP):
            tcp = pkt[TCP]
            pkt_info["src_port"] = tcp.sport
            pkt_info["dst_port"] = tcp.dport
            pkt_info["psh"] = bool(tcp.flags & 0x08)
            pkt_info["urg"] = bool(tcp.flags & 0x20)
        elif pkt.haslayer(UDP):
            udp = pkt[UDP]
            pkt_info["src_port"] = udp.sport
            pkt_info["dst_port"] = udp.dport
        else:
            pkt_info["src_port"] = 0
            pkt_info["dst_port"] = 0

        tracker.process_packet(pkt_info)
        packet_count[0] += 1

    stop_event = threading.Event()

    def flow_processor():
        while not stop_event.is_set():
            tracker.expire_flows()
            flows = tracker.get_completed_flows()
            for features in flows:
                ips.add_flow(features)
                result = ips.check_window()
                if result:
                    ips.print_result(result)
            time.sleep(1)

    processor_thread = threading.Thread(target=flow_processor, daemon=True)
    processor_thread.start()

    print("  Capturing packets... (Ctrl+C to stop)\n")
    print(f"  {'Timestamp':<26} | {'Action':5} | {'Classification':<25} | "
          f"{'Confidence':>12} | {'Latency':>10}")
    print(f"  {'-' * 90}")

    try:
        sniff(
            iface=interface,
            prn=process_pkt,
            timeout=duration,
            store=False,
        )
    except KeyboardInterrupt:
        print("\n  [Capture stopped by user]")
    except Exception as e:
        print(f"\n  [Capture error: {e}]")
        print("  [Falling back to simulation mode]\n")
        run_simulation(model, 30)
        return

    stop_event.set()
    tracker.force_close_all()
    flows = tracker.get_completed_flows()
    for features in flows:
        ips.add_flow(features)
        result = ips.check_window()
        if result:
            ips.print_result(result)

    processor_thread.join(timeout=5)

    print(f"\n  Total packets captured: {packet_count[0]}")
    ips.print_summary()
    ips.save_alerts()


def main():
    parser = argparse.ArgumentParser(description="Live packet capture IPS")
    parser.add_argument("--interface", type=str, default=None,
                        help="Network interface name (e.g., 'Wi-Fi', 'Ethernet')")
    parser.add_argument("--duration", type=int, default=60,
                        help="Capture duration in seconds")
    parser.add_argument("--simulate", action="store_true",
                        help="Use test data instead of live capture")
    args = parser.parse_args()

    print("[ips] Loading trained model...")
    model = build_model(num_features=len(config.FEATURE_COLUMNS))
    model.load_state_dict(
        torch.load(config.MODEL_CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    )
    model.eval()
    print("[ips] Model loaded successfully")

    if args.simulate:
        run_simulation(model, duration=args.duration)
    else:
        run_live_capture(model, interface=args.interface, duration=args.duration)


if __name__ == "__main__":
    main()
