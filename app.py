import sys
from pathlib import Path
import time
import random
import threading
from datetime import datetime

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from flask import Flask, jsonify, request, send_from_directory
import torch
import numpy as np

from src import config
from src import clean_state_dict
from src.sessionization import load_sequences
from src.model_v2 import build_model_v2
from src.train import get_model
from src.drift_detector import DriftDetector
from src.prevention_engine import PreventionEngine
from src.live_capture import LiveCaptureEngine, SCAPY_AVAILABLE

MODEL_REGISTRY = {
    "model_v2": {
        "name": "SentinelV2 (Primary)",
        "params": "2.87M",
        "checkpoint": config.MODELS_DIR / "best_model_v2_swa.pth",
        "fallback": config.MODELS_DIR / "best_model_v2.pth",
        "is_v2": True,
    },
    "cnn_lstm": {
        "name": "CNN-LSTM",
        "params": "318K",
        "checkpoint": config.MODELS_DIR / "best_cnn_lstm.pth",
        "fallback": None,
        "is_v2": False,
    },
    "cnn_lstm_attention": {
        "name": "CNN-LSTM-Attention",
        "params": "368K",
        "checkpoint": config.MODELS_DIR / "best_cnn_lstm_attention.pth",
        "fallback": None,
        "is_v2": False,
    },
    "cnn_bilstm": {
        "name": "CNN-BiLSTM",
        "params": "812K",
        "checkpoint": config.MODELS_DIR / "best_cnn_bilstm.pth",
        "fallback": None,
        "is_v2": False,
    },
    "cnn_transformer": {
        "name": "CNN-Transformer",
        "params": "346K",
        "checkpoint": config.MODELS_DIR / "best_cnn_transformer.pth",
        "fallback": None,
        "is_v2": False,
    },
}

ATTACK_CATEGORIES = {
    "Normal": {"color": "#00aa00", "emoji": "OK", "meta_class": 0},
    "DoS-Hulk": {"color": "#ff8888", "emoji": "DOS", "meta_class": 1},
    "DoS-Slowloris": {"color": "#ffaaaa", "emoji": "DOS", "meta_class": 1},
    "DoS-GoldenEye": {"color": "#ff9999", "emoji": "DOS", "meta_class": 1},
    "DDoS-HOIC": {"color": "#ff4444", "emoji": "DDOS", "meta_class": 1},
    "DDoS-LOIC": {"color": "#ff6666", "emoji": "DDOS", "meta_class": 1},
    "DrDoS-Amplification": {"color": "#ff3333", "emoji": "DDOS", "meta_class": 1},
    "DDoS-SYN-Flood": {"color": "#ff5555", "emoji": "DDOS", "meta_class": 1},
    "PortScan": {"color": "#ffaa00", "emoji": "SCAN", "meta_class": 2},
    "Reconnaissance": {"color": "#ffbb33", "emoji": "RECON", "meta_class": 2},
    "SQL-Injection": {"color": "#aa00aa", "emoji": "SQLI", "meta_class": 3},
    "XSS-Attack": {"color": "#cc44cc", "emoji": "XSS", "meta_class": 3},
    "FTP-Brute": {"color": "#e879f9", "emoji": "BRUTE", "meta_class": 4},
    "SSH-Brute": {"color": "#d946ef", "emoji": "BRUTE", "meta_class": 4},
    "Botnet": {"color": "#0066aa", "emoji": "BOT", "meta_class": 5},
    "Mirai": {"color": "#44aadd", "emoji": "MIRAI", "meta_class": 5},
    "Backdoor": {"color": "#0088cc", "emoji": "BACK", "meta_class": 6},
    "Exploit": {"color": "#22d3ee", "emoji": "EXPL", "meta_class": 6},
    "Infiltration": {"color": "#facc15", "emoji": "INFIL", "meta_class": 7},
}

ATTACKER_IP_POOLS = {
    "DoS-Hulk": ["192.0.2.{}"],
    "DoS-Slowloris": ["192.0.2.{}"],
    "DoS-GoldenEye": ["192.0.2.{}"],
    "DDoS-HOIC": ["45.33.32.{}", "198.51.100.{}", "203.0.113.{}"],
    "DDoS-LOIC": ["45.33.32.{}", "198.51.100.{}"],
    "DrDoS-Amplification": ["45.33.32.{}", "198.51.100.{}", "203.0.113.{}"],
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


def get_simulated_ip(attack_type):
    if attack_type == "Normal":
        return f"192.168.1.{random.randint(100, 200)}"
    pool = ATTACKER_IP_POOLS.get(attack_type, ["unknown.{}.{}"])
    template = random.choice(pool)
    return template.format(random.randint(1, 254))


def map_meta_to_specific(meta_class_id):
    mapping = {
        0: ["Normal"],
        1: ["DoS-Hulk", "DoS-Slowloris", "DoS-GoldenEye", "DDoS-HOIC", "DDoS-LOIC", "DrDoS-Amplification", "DDoS-SYN-Flood"],
        2: ["PortScan", "Reconnaissance"],
        3: ["SQL-Injection", "XSS-Attack"],
        4: ["FTP-Brute", "SSH-Brute"],
        5: ["Botnet", "Mirai"],
        6: ["Backdoor", "Exploit"],
        7: ["Infiltration"],
    }
    options = mapping.get(meta_class_id, ["Normal"])
    return random.choice(options)


SPECIFIC_TO_META = {
    "DoS-Hulk": "DoS/DDoS", "DoS-Slowloris": "DoS/DDoS", "DoS-GoldenEye": "DoS/DDoS",
    "DDoS-HOIC": "DoS/DDoS", "DDoS-LOIC": "DoS/DDoS", "DrDoS-Amplification": "DoS/DDoS",
    "DDoS-SYN-Flood": "DoS/DDoS",
    "PortScan": "PortScan/Recon", "Reconnaissance": "PortScan/Recon",
    "SQL-Injection": "Web/Injection", "XSS-Attack": "Web/Injection",
    "FTP-Brute": "Brute Force", "SSH-Brute": "Brute Force",
    "Botnet": "Botnet/C2", "Mirai": "Botnet/C2",
    "Backdoor": "Malware/Exploit", "Exploit": "Malware/Exploit",
    "Infiltration": "Infiltration",
}


def get_traffic(class_sequences, attack_active):
    if attack_active["type"] is None:
        return class_sequences["Normal"][random.randint(0, len(class_sequences["Normal"])-1):][0:1], "Normal"
    meta_name = ATTACK_CATEGORIES[attack_active["type"]]["meta_class"]
    meta_class_name = v2_class_names[meta_name] if meta_name < len(v2_class_names) else "Normal"
    if random.random() < 0.8:
        seqs = class_sequences.get(meta_class_name, class_sequences.get("Normal", list(class_sequences.values())[0]))
        return seqs[random.randint(0, len(seqs)-1):][0:1], attack_active["type"]
    else:
        return class_sequences["Normal"][random.randint(0, len(class_sequences["Normal"])-1):][0:1], "Normal"


def load_model_and_data(variant="model_v2"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sequences, labels, _ = load_sequences()

    seq_flat = sequences.reshape(-1, sequences.shape[-1])
    mean = seq_flat.mean(dim=0, keepdim=True)
    std = seq_flat.std(dim=0, keepdim=True).clamp(min=1e-8)
    sequences = ((sequences - mean) / std).clamp(-10, 10)

    v2_class_names = config.get_class_names()
    class_sequences = {}
    for i, name in enumerate(v2_class_names):
        mask = labels == i
        if mask.sum() > 0:
            class_sequences[name] = sequences[mask]

    reg = MODEL_REGISTRY[variant]
    ckpt_path = reg["checkpoint"]
    if reg["fallback"] and reg["fallback"].exists() and not ckpt_path.exists():
        ckpt_path = reg["fallback"]
    elif reg["is_v2"] and reg["fallback"] and reg["fallback"].exists():
        ckpt_path = reg["checkpoint"] if reg["checkpoint"].exists() else reg["fallback"]

    num_features = sequences.shape[-1]

    if reg["is_v2"]:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        mdl = build_model_v2(
            num_features=ckpt["num_features"],
            num_classes=ckpt.get("num_classes", config.NUM_CLASSES),
        )
        mdl.load_state_dict(clean_state_dict(ckpt["model_state_dict"]))
    else:
        mdl = get_model(variant, num_features=num_features, num_classes=config.NUM_CLASSES)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "model_state_dict" in ckpt:
            mdl.load_state_dict(clean_state_dict(ckpt["model_state_dict"]))
        else:
            mdl.load_state_dict(clean_state_dict(ckpt))

    mdl.to(device)
    mdl.eval()
    return {"model": mdl, "device": device, "class_sequences": class_sequences,
            "class_names": v2_class_names, "variant": variant,
            "model_name": reg["name"], "model_params": reg["params"]}


app = Flask(__name__, static_folder="web")

state_lock = threading.Lock()
STATE = {
    "running": False,
    "attack_active": {"type": None},
    "stats": {"total": 0, "attacks": 0, "latency": 0.0, "blocked": 0},
    "alerts": [],
    "throughput_history": [0.0] * 30,
    "drift": {"status": "normal", "severity": 0.0, "message": "Baseline stable"},
    "ips_enabled": True,
    "class_counts": {
        "Normal": 0, "DoS/DDoS": 0, "PortScan/Recon": 0,
        "Web/Injection": 0, "Brute Force": 0, "Botnet/C2": 0,
        "Malware/Exploit": 0, "Infiltration": 0,
    },
    "timeline": [],
    "live_mode": False,
}

print("\n" + "=" * 60)
print("  PROJECT SENTINEL V2 — 8-CLASS IPS COMMAND CENTER")
print("=" * 60)
print("[*] Loading SentinelV2 MultiScale CNN-BiLSTM-GRU-MHA...")
model_data = load_model_and_data("model_v2")
model = model_data["model"]
device = model_data["device"]
class_sequences = model_data["class_sequences"]
v2_class_names = model_data["class_names"]
current_variant = model_data["variant"]
current_model_name = model_data["model_name"]
current_model_params = model_data["model_params"]
print(f"[OK] {current_model_name} loaded to {device} ({len(v2_class_names)} classes)")

print("[*] Initializing Prevention Engine (dry_run=True)...")
ips_engine = PreventionEngine(dry_run=True)
print("[OK] IPS Prevention Engine ACTIVE")

print("[*] Initializing Concept Drift Detector...")
drift_detector = DriftDetector()
if "Normal" in class_sequences and len(class_sequences["Normal"]) > 0:
    baseline_data = class_sequences["Normal"][:5000].numpy()
    baseline_data_flat = baseline_data.reshape(-1, baseline_data.shape[-1])
    drift_detector.set_baseline(baseline_data_flat)

live_capture_engine = None
if SCAPY_AVAILABLE:
    _ckpt_path = config.MODELS_DIR / "best_model_v2_swa.pth"
    if not _ckpt_path.exists():
        _ckpt_path = config.V2_MODEL_SAVE_PATH
    _zs_mean, _zs_std = None, None
    if _ckpt_path.exists():
        _ckpt = torch.load(_ckpt_path, map_location=device, weights_only=False)
        _zs_mean = _ckpt.get("zscore_mean")
        _zs_std = _ckpt.get("zscore_std")
        if _zs_mean is not None:
            _zs_mean = _zs_mean.cpu().numpy() if torch.is_tensor(_zs_mean) else _zs_mean
            _zs_std = _zs_std.cpu().numpy() if torch.is_tensor(_zs_std) else _zs_std

    def _live_alert_callback(alert):
        meta_class = alert["meta_class"]
        ip = alert["ip"]
        confidence = alert["confidence"]
        pred_class_id = alert.get("pred_class_id", 0)

        ips_action = "ALLOW"
        ips_blocked = False
        if STATE["ips_enabled"] and pred_class_id != 0:
            ips_engine.process_detection(
                source_ip=ip,
                attack_class=meta_class,
                confidence=confidence / 100.0,
            )
            if ip in ips_engine.blocked_ips:
                ips_action = "BLOCKED"
                ips_blocked = True
            else:
                ips_action = "ALERT"

        severity = "BLOCKED" if ips_blocked else ("ALERT" if pred_class_id != 0 else "OK")
        badge_class = "badge-blocked" if ips_blocked else ("badge-alert" if pred_class_id != 0 else "badge-ok")

        dashboard_alert = {
            "time": alert["time"],
            "type": meta_class,
            "meta_class": meta_class,
            "ip": ip,
            "emoji": "LIVE",
            "severity": severity,
            "badge_class": badge_class,
            "confidence": f"{confidence:.1f}",
            "ips_action": ips_action,
            "source": "live",
        }

        with state_lock:
            STATE["class_counts"][meta_class] = STATE["class_counts"].get(meta_class, 0) + 1
            if pred_class_id != 0:
                STATE["stats"]["attacks"] += 1
            if ips_blocked:
                STATE["stats"]["blocked"] = len(ips_engine.blocked_ips)
            STATE["alerts"].insert(0, dashboard_alert)
            if len(STATE["alerts"]) > 100:
                STATE["alerts"] = STATE["alerts"][:100]

    live_capture_engine = LiveCaptureEngine(
        model=model, device=device,
        zscore_mean=_zs_mean, zscore_std=_zs_std,
        on_alert=_live_alert_callback,
    )
    print("[OK] Live Capture Engine ready (scapy available)")
else:
    print("[!] Live Capture unavailable (scapy not installed)")

print("=" * 60 + "\n")

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

        with state_lock:
            STATE["stats"]["total"] += packets_added
            STATE["throughput_history"].pop(0)
            STATE["throughput_history"].append(random.uniform(8.0, 10.5))

        t0 = time.time()
        seq, true_type = get_traffic(class_sequences, attack_active)

        with torch.no_grad():
            output = model(seq.to(device))
            temperature = 0.5
            scaled_logits = output / temperature
            probabilities = torch.nn.functional.softmax(scaled_logits, dim=1)
            pred_meta = output.argmax(dim=1).item()
            confidence = probabilities[0][pred_meta].item() * 100

        seq_flat = seq.cpu().numpy().reshape(-1, seq.shape[-1])
        drift_report = drift_detector.check(seq_flat)

        t1 = time.time()
        latency_ms = (t1 - t0) * 1000
        window_counter += 1

        if pred_meta != 0:
            pred_type = map_meta_to_specific(pred_meta) if true_type == "Normal" else true_type
        else:
            pred_type = "Normal"

        ip = get_simulated_ip(pred_type)
        meta_class_name = v2_class_names[pred_meta] if pred_meta < len(v2_class_names) else "Normal"

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

        frontend_meta = v2_class_names[pred_meta] if pred_meta < len(v2_class_names) else "Normal"

        alert = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "type": pred_type,
            "meta_class": frontend_meta,
            "ip": ip,
            "emoji": ATTACK_CATEGORIES.get(pred_type, {}).get("emoji", "?"),
            "severity": severity,
            "badge_class": badge_class,
            "confidence": f"{confidence:.1f}",
            "ips_action": ips_action,
        }

        with state_lock:
            STATE["stats"]["latency"] = latency_ms
            STATE["class_counts"][frontend_meta] = STATE["class_counts"].get(frontend_meta, 0) + 1
            if pred_meta != 0:
                STATE["stats"]["attacks"] += 1
            if ips_blocked:
                STATE["stats"]["blocked"] = len(ips_engine.blocked_ips)
            STATE["alerts"].insert(0, alert)
            if len(STATE["alerts"]) > 100:
                STATE["alerts"] = STATE["alerts"][:100]
            if drift_report.get("status") in ["DRIFT_WARNING", "DRIFT_CRITICAL"]:
                STATE["drift"] = drift_report

        time.sleep(0.3)

worker = threading.Thread(target=inference_loop, daemon=True)
worker.start()

@app.route("/")
def index():
    return send_from_directory("web", "index.html")

@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory("web", path)

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
                "variant": current_variant,
                "name": current_model_name,
                "params": current_model_params,
                "available": {k: {"name": v["name"], "params": v["params"],
                              "available": v["checkpoint"].exists()}
                              for k, v in MODEL_REGISTRY.items()},
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
    global live_capture_engine
    req = request.json or {}
    action = req.get("action", "toggle")
    interface = req.get("interface", None)

    if live_capture_engine is None:
        return jsonify({"status": "error", "message": "Live capture unavailable (scapy not installed)"}), 400

    with state_lock:
        if action == "start" or (action == "toggle" and not STATE["live_mode"]):
            STATE["running"] = False
            STATE["live_mode"] = True
            time.sleep(0.3)
            live_capture_engine.start(interface=interface)
            return jsonify({"status": "ok", "live_mode": True, "message": f"Live capture started on {interface or 'default'}"})
        elif action == "stop" or (action == "toggle" and STATE["live_mode"]):
            live_capture_engine.stop()
            STATE["live_mode"] = False
            return jsonify({"status": "ok", "live_mode": False, "message": "Live capture stopped"})

    return jsonify({"status": "ok", "live_mode": STATE.get("live_mode", False)})

@app.route("/api/live_status", methods=["GET"])
def live_status():
    if live_capture_engine is None:
        return jsonify({"available": False, "active": False, "stats": {}})

    return jsonify({
        "available": True,
        "active": STATE.get("live_mode", False),
        "active_flows": live_capture_engine.get_active_flows(),
        "stats": live_capture_engine.get_stats(),
    })

@app.route("/api/interfaces", methods=["GET"])
def list_interfaces():
    if live_capture_engine is None:
        return jsonify({"interfaces": [], "available": False})

    return jsonify({
        "interfaces": live_capture_engine.list_interfaces(),
        "available": True,
    })

@app.route("/api/switch_model", methods=["POST"])
def switch_model():
    global model, current_variant, current_model_name, current_model_params
    req = request.json
    variant = req.get("variant", "model_v2")
    if variant not in MODEL_REGISTRY:
        return jsonify({"status": "error", "message": f"Unknown variant: {variant}"}), 400
    if not MODEL_REGISTRY[variant]["checkpoint"].exists():
        return jsonify({"status": "error", "message": f"Checkpoint not found for {variant}"}), 404
    try:
        with state_lock:
            was_running = STATE["running"]
            STATE["running"] = False
        import time as _t; _t.sleep(0.5)
        print(f"[*] Switching model to {MODEL_REGISTRY[variant]['name']}...")
        data = load_model_and_data(variant)
        model = data["model"]
        current_variant = data["variant"]
        current_model_name = data["model_name"]
        current_model_params = data["model_params"]
        print(f"[OK] Model switched to {current_model_name} ({current_model_params} params)")
        with state_lock:
            STATE["running"] = was_running
        return jsonify({"status": "ok", "model": current_model_name, "params": current_model_params})
    except Exception as e:
        with state_lock:
            STATE["running"] = True
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    from waitress import serve
    print("[*] Serving IPS Dashboard on http://0.0.0.0:8082")
    serve(app, host="0.0.0.0", port=8082)
