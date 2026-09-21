import sys
import time
import random
import threading
from pathlib import Path
from datetime import datetime

import numpy as np
from sentinel_numpy import SentinelNumPy

from flask import Flask, jsonify, request, send_from_directory

PROJECT_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = PROJECT_DIR / "sentinel_weights.npz"
ZSCORE_STATS_PATH = PROJECT_DIR / "zscore_stats.npz"
TRAFFIC_SAMPLES_PATH = PROJECT_DIR / "traffic_samples.npz"
WEB_DIR = PROJECT_DIR / "web"
PORT = 8082

CLASS_NAMES = [
    "Normal", "DoS/DDoS", "PortScan/Recon", "Web/Injection",
    "Brute Force", "Botnet/C2", "Malware/Exploit", "Infiltration"
]
NUM_CLASSES = 8
SEQUENCE_LENGTH = 20
NUM_FEATURES = 30

ATTACK_CATEGORIES = {
    "Normal": {"emoji": "OK", "meta_class": 0},
    "DoS-Hulk": {"emoji": "DOS", "meta_class": 1},
    "DoS-Slowloris": {"emoji": "DOS", "meta_class": 1},
    "DoS-GoldenEye": {"emoji": "DOS", "meta_class": 1},
    "DDoS-HOIC": {"emoji": "DDOS", "meta_class": 1},
    "DDoS-LOIC": {"emoji": "DDOS", "meta_class": 1},
    "DDoS-SYN-Flood": {"emoji": "DDOS", "meta_class": 1},
    "PortScan": {"emoji": "SCAN", "meta_class": 2},
    "Reconnaissance": {"emoji": "RECON", "meta_class": 2},
    "SQL-Injection": {"emoji": "SQLI", "meta_class": 3},
    "XSS-Attack": {"emoji": "XSS", "meta_class": 3},
    "FTP-Brute": {"emoji": "BRUTE", "meta_class": 4},
    "SSH-Brute": {"emoji": "BRUTE", "meta_class": 4},
    "Botnet": {"emoji": "BOT", "meta_class": 5},
    "Mirai": {"emoji": "MIRAI", "meta_class": 5},
    "Backdoor": {"emoji": "BACK", "meta_class": 6},
    "Exploit": {"emoji": "EXPL", "meta_class": 6},
    "Infiltration": {"emoji": "INFIL", "meta_class": 7},
}

ATTACKER_IP_POOLS = {
    "DoS-Hulk": ["192.0.2.{}"],
    "DoS-Slowloris": ["192.0.2.{}"],
    "DoS-GoldenEye": ["192.0.2.{}"],
    "DDoS-HOIC": ["45.33.32.{}", "198.51.100.{}", "203.0.113.{}"],
    "DDoS-LOIC": ["45.33.32.{}", "198.51.100.{}"],
    "DDoS-SYN-Flood": ["45.33.32.{}", "198.51.100.{}"],
    "PortScan": ["10.0.0.{}", "172.16.0.{}"],
    "Reconnaissance": ["10.0.0.{}", "172.16.0.{}"],
    "SQL-Injection": ["185.220.101.{}", "162.247.74.{}"],
    "XSS-Attack": ["185.220.101.{}"],
    "FTP-Brute": ["185.220.101.{}", "23.129.64.{}"],
    "SSH-Brute": ["185.220.101.{}", "23.129.64.{}"],
    "Botnet": ["77.247.181.{}", "185.220.102.{}"],
    "Backdoor": ["77.247.181.{}"],
    "Mirai": ["77.247.181.{}", "185.220.102.{}"],
    "Exploit": ["198.51.100.{}", "203.0.113.{}"],
    "Infiltration": ["10.10.10.{}", "172.16.0.{}"],
}

META_TO_SPECIFIC = {
    0: ["Normal"],
    1: ["DoS-Hulk", "DoS-Slowloris", "DoS-GoldenEye", "DDoS-HOIC", "DDoS-LOIC", "DDoS-SYN-Flood"],
    2: ["PortScan", "Reconnaissance"],
    3: ["SQL-Injection", "XSS-Attack"],
    4: ["FTP-Brute", "SSH-Brute"],
    5: ["Botnet", "Mirai"],
    6: ["Backdoor", "Exploit"],
    7: ["Infiltration"],
}

SPECIFIC_TO_META = {
    "DoS-Hulk": "DoS/DDoS", "DoS-Slowloris": "DoS/DDoS", "DoS-GoldenEye": "DoS/DDoS",
    "DDoS-HOIC": "DoS/DDoS", "DDoS-LOIC": "DoS/DDoS", "DDoS-SYN-Flood": "DoS/DDoS",
    "PortScan": "PortScan/Recon", "Reconnaissance": "PortScan/Recon",
    "SQL-Injection": "Web/Injection", "XSS-Attack": "Web/Injection",
    "FTP-Brute": "Brute Force", "SSH-Brute": "Brute Force",
    "Botnet": "Botnet/C2", "Mirai": "Botnet/C2",
    "Backdoor": "Malware/Exploit", "Exploit": "Malware/Exploit",
    "Infiltration": "Infiltration",
}


class NumpyEngine:

    def __init__(self, weights_path, zscore_path):
        self.engine = SentinelNumPy(
            weights_path=str(weights_path),
            zscore_path=str(zscore_path)
        )
        print(f"[OK] NumPy inference engine ready (no ONNX Runtime needed)")

    def predict(self, window):
        if window.ndim == 2:
            window = window.reshape(1, SEQUENCE_LENGTH, NUM_FEATURES)

        input_data = window.astype(np.float32)

        t0 = time.perf_counter()
        output = self.engine.forward(input_data)
        t1 = time.perf_counter()

        pred_class = int(np.argmax(output[0]))

        logits = output[0] / 0.5
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        confidence = float(probs[pred_class]) * 100

        return pred_class, confidence, (t1 - t0) * 1000


class TrafficSampleLoader:

    def __init__(self, samples_path):
        print(f"[*] Loading traffic samples: {samples_path}")
        data = np.load(str(samples_path))
        self.class_sequences = {}
        for name in CLASS_NAMES:
            if name in data:
                self.class_sequences[name] = data[name].astype(np.float32)
                print(f"    {name}: {len(self.class_sequences[name])} windows")
            else:
                print(f"    {name}: 0 windows (not in file)")

        total = sum(len(v) for v in self.class_sequences.values())
        print(f"[OK] Loaded {total} real traffic windows across {len(self.class_sequences)} classes")

    def get_sample(self, class_name):
        seqs = self.class_sequences.get(class_name)
        if seqs is None or len(seqs) == 0:
            seqs = self.class_sequences.get("Normal")
        idx = random.randint(0, len(seqs) - 1)
        return seqs[idx]


class LightweightIPS:

    def __init__(self, dry_run=True):
        self.dry_run = dry_run
        self.blocked_ips = set()
        self.ip_strike_counter = {}
        self.recent_actions = []
        self.stats = {"total_ips_blocked": 0, "total_attacks_detected": 0}
        self.strike_threshold = 3
        self.confidence_threshold = 0.7

    def process_detection(self, source_ip, attack_class, confidence, window_id=0):
        self.stats["total_attacks_detected"] += 1

        if confidence < self.confidence_threshold:
            return

        if source_ip not in self.ip_strike_counter:
            self.ip_strike_counter[source_ip] = {"count": 0, "attack_types": set()}

        self.ip_strike_counter[source_ip]["count"] += 1
        self.ip_strike_counter[source_ip]["attack_types"].add(attack_class)

        if self.ip_strike_counter[source_ip]["count"] >= self.strike_threshold:
            if source_ip not in self.blocked_ips:
                self.blocked_ips.add(source_ip)
                self.stats["total_ips_blocked"] += 1

                action = {
                    "action_taken": "BLOCKED",
                    "source_ip": source_ip,
                    "attack_type": attack_class,
                    "confidence": confidence,
                    "timestamp": datetime.now().isoformat(),
                }
                self.recent_actions.append(action)

                if not self.dry_run:
                    self._block_ip(source_ip)

                print(f"[IPS] {'[DRY-RUN] ' if self.dry_run else ''}BLOCKED {source_ip} for {attack_class}")

    def _block_ip(self, ip):
        import subprocess
        try:
            subprocess.run(
                ["iptables", "-A", "INPUT", "-s", ip, "-j", "DROP"],
                check=True, capture_output=True
            )
        except Exception as e:
            print(f"[IPS] iptables block failed: {e}")

    def cleanup_all_rules(self):
        self.blocked_ips.clear()
        self.ip_strike_counter.clear()
        self.recent_actions.append({
            "action_taken": "UNBLOCKED",
            "source_ip": "ALL",
            "attack_type": "Cleanup",
            "confidence": 1.0,
            "timestamp": datetime.now().isoformat(),
        })

    def get_status(self):
        return {
            "blocked_ips": list(self.blocked_ips),
            "blocked_count": len(self.blocked_ips),
            "recent_actions": self.recent_actions[-10:],
            "dry_run": self.dry_run,
            "stats": self.stats,
        }


def get_simulated_ip(attack_type):
    if attack_type == "Normal":
        return f"192.168.1.{random.randint(100, 200)}"
    pool = ATTACKER_IP_POOLS.get(attack_type, ["unknown.{}.{}"])
    template = random.choice(pool)
    return template.format(random.randint(1, 254))


def map_meta_to_specific(meta_class_id):
    options = META_TO_SPECIFIC.get(meta_class_id, ["Normal"])
    return random.choice(options)


print("\n" + "=" * 60)
print("  PROJECT SENTINEL V2 — PYNQ EDGE IPS")
print("=" * 60)

for fpath, fname in [(WEIGHTS_PATH, "Model weights"), (TRAFFIC_SAMPLES_PATH, "Traffic samples")]:
    if not fpath.exists():
        print(f"[!] {fname} not found at {fpath}")
        print("    Run export_pynq.py on your PC first.")
        sys.exit(1)

engine = NumpyEngine(WEIGHTS_PATH, ZSCORE_STATS_PATH)

traffic_loader = TrafficSampleLoader(TRAFFIC_SAMPLES_PATH)

ips_engine = LightweightIPS(dry_run=True)
print("[OK] IPS Prevention Engine (dry_run=True)")

print("=" * 60 + "\n")

app = Flask(__name__, static_folder=str(WEB_DIR))

state_lock = threading.Lock()
STATE = {
    "running": False,
    "attack_active": {"type": None},
    "stats": {"total": 0, "attacks": 0, "latency": 0.0, "blocked": 0},
    "alerts": [],
    "throughput_history": [0.0] * 30,
    "drift": {"status": "normal", "severity": 0.0, "message": "Baseline stable"},
    "ips_enabled": True,
    "class_counts": {name: 0 for name in CLASS_NAMES},
    "live_mode": False,
}

window_counter = 0


def inference_loop():
    global window_counter

    while True:
        with state_lock:
            running = STATE["running"]
            attack_active = dict(STATE["attack_active"])
            ips_enabled = STATE["ips_enabled"]

        if not running:
            time.sleep(0.5)
            continue

        packets_added = random.randint(1000, 1500)

        if attack_active["type"] is not None:
            meta_id = ATTACK_CATEGORIES[attack_active["type"]]["meta_class"]
            meta_class_name = CLASS_NAMES[meta_id]
            if random.random() < 0.8:
                window = traffic_loader.get_sample(meta_class_name)
                true_type = attack_active["type"]
            else:
                window = traffic_loader.get_sample("Normal")
                true_type = "Normal"
        else:
            window = traffic_loader.get_sample("Normal")
            true_type = "Normal"

        pred_meta, confidence, latency_ms = engine.predict(window)
        window_counter += 1

        if pred_meta != 0:
            pred_type = map_meta_to_specific(pred_meta) if true_type == "Normal" else true_type
        else:
            pred_type = "Normal"

        ip = get_simulated_ip(pred_type)
        meta_class_name = CLASS_NAMES[pred_meta] if pred_meta < NUM_CLASSES else "Normal"

        ips_action = "ALLOW"
        ips_blocked = False
        if ips_enabled and pred_meta != 0:
            policy_class = SPECIFIC_TO_META.get(pred_type, meta_class_name)
            ips_engine.process_detection(
                source_ip=ip,
                attack_class=policy_class,
                confidence=confidence / 100.0,
                window_id=window_counter,
            )
            if ip in ips_engine.blocked_ips:
                ips_action = "BLOCKED"
                ips_blocked = True
            else:
                ips_action = "ALERT"

        if ips_blocked:
            severity = "BLOCKED"
            badge_class = "badge-blocked"
        elif pred_meta != 0:
            if any(x in pred_type for x in ["DDoS", "DoS", "SQL", "Brute", "Botnet", "Mirai", "Backdoor"]):
                severity = "ALERT"
                badge_class = "badge-alert"
            else:
                severity = "LOG"
                badge_class = "badge-log"
        else:
            severity = "OK"
            badge_class = "badge-ok"

        alert = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "type": pred_type,
            "meta_class": meta_class_name,
            "ip": ip,
            "emoji": ATTACK_CATEGORIES.get(pred_type, {}).get("emoji", "?"),
            "severity": severity,
            "badge_class": badge_class,
            "confidence": f"{confidence:.1f}",
            "ips_action": ips_action,
        }

        with state_lock:
            STATE["stats"]["total"] += packets_added
            STATE["stats"]["latency"] = latency_ms
            STATE["throughput_history"].pop(0)
            STATE["throughput_history"].append(random.uniform(8.0, 10.5))
            STATE["class_counts"][meta_class_name] = STATE["class_counts"].get(meta_class_name, 0) + 1
            if pred_meta != 0:
                STATE["stats"]["attacks"] += 1
            if ips_blocked:
                STATE["stats"]["blocked"] = len(ips_engine.blocked_ips)
            STATE["alerts"].insert(0, alert)
            if len(STATE["alerts"]) > 100:
                STATE["alerts"] = STATE["alerts"][:100]

        time.sleep(0.3)


worker = threading.Thread(target=inference_loop, daemon=True)
worker.start()


@app.route("/")
def index():
    return send_from_directory(str(WEB_DIR), "index.html")


@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(str(WEB_DIR), path)


@app.route("/api/state", methods=["GET"])
def get_state():
    with state_lock:
        ips_status = ips_engine.get_status()
        return jsonify({
            "running": STATE["running"],
            "attack_active": STATE["attack_active"]["type"],
            "stats": STATE["stats"],
            "alerts": STATE["alerts"][:12],
            "throughput": STATE["throughput_history"],
            "drift": STATE["drift"],
            "class_counts": STATE["class_counts"],
            "ips": {
                "enabled": STATE["ips_enabled"],
                "blocked_ips": ips_status["blocked_ips"],
                "blocked_count": ips_status["blocked_count"],
                "recent_actions": ips_status["recent_actions"][-8:],
                "dry_run": ips_status["dry_run"],
                "stats": {
                    "total_blocked": ips_status["stats"]["total_ips_blocked"],
                    "total_alerts": ips_status["stats"]["total_attacks_detected"],
                },
            },
            "model": {
                "variant": "model_v2",
                "name": "SentinelV2 (NumPy)",
                "params": "2.87M",
                "available": {},
            },
        })


@app.route("/api/control", methods=["POST"])
def control():
    req = request.json
    action = req.get("action")
    with state_lock:
        if action == "start":
            STATE["running"] = True
        elif action == "stop":
            STATE["running"] = False
        elif action == "inject":
            STATE["attack_active"]["type"] = req.get("type", None)
        elif action == "toggle_ips":
            STATE["ips_enabled"] = not STATE["ips_enabled"]
        elif action == "unblock_all":
            ips_engine.cleanup_all_rules()
            STATE["stats"]["blocked"] = 0
    return jsonify({"status": "ok", "state": action})


@app.route("/api/live_mode", methods=["POST"])
def toggle_live_mode():
    return jsonify({"status": "error", "message": "Live capture not available on PYNQ (use desktop app)"}), 400


@app.route("/api/live_status", methods=["GET"])
def live_status():
    return jsonify({"available": False, "active": False, "stats": {}})


@app.route("/api/interfaces", methods=["GET"])
def list_interfaces():
    return jsonify({"interfaces": [], "available": False})


@app.route("/api/switch_model", methods=["POST"])
def switch_model():
    return jsonify({"status": "ok", "model": "SentinelV2 (NumPy)", "params": "2.87M"})


if __name__ == "__main__":
    print(f"[*] Serving PYNQ IPS Dashboard on http://0.0.0.0:{PORT}")
    try:
        from waitress import serve
        serve(app, host="0.0.0.0", port=PORT)
    except ImportError:
        print("[*] waitress not found, using Flask dev server")
        app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
