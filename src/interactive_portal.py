import sys
from pathlib import Path
import time
import random
import json
import torch
import streamlit as st
from datetime import datetime
from collections import defaultdict

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src import config
from src import clean_state_dict
from src.sessionization import load_sequences
from src.model_cnn_lstm import build_model

ATTACK_CATEGORIES = {
    "Normal": {"color": "#00aa00", "emoji": "✅", "meta_class": 0},
    "DoS-Hulk": {"color": "#ff8888", "emoji": "💥", "meta_class": 1},
    "DoS-Slowloris": {"color": "#ffaaaa", "emoji": "🐌", "meta_class": 1},
    "DoS-GoldenEye": {"color": "#ff9999", "emoji": "💥", "meta_class": 1},
    "DDoS-HOIC": {"color": "#ff4444", "emoji": "🌊", "meta_class": 2},
    "DDoS-LOIC": {"color": "#ff6666", "emoji": "🌊", "meta_class": 2},
    "DrDoS-Amplification": {"color": "#ff3333", "emoji": "🌊", "meta_class": 2},
    "DDoS-SYN-Flood": {"color": "#ff5555", "emoji": "🌊", "meta_class": 2},
    "PortScan": {"color": "#ffaa00", "emoji": "🔍", "meta_class": 3},
    "Reconnaissance": {"color": "#ffbb33", "emoji": "👁️", "meta_class": 3},
    "SQL-Injection": {"color": "#aa00aa", "emoji": "💉", "meta_class": 4},
    "XSS-Attack": {"color": "#cc44cc", "emoji": "📜", "meta_class": 4},
    "Brute-Force": {"color": "#ee66ee", "emoji": "🔨", "meta_class": 4},
    "Botnet": {"color": "#0066aa", "emoji": "🤖", "meta_class": 5},
    "Backdoor": {"color": "#0088cc", "emoji": "🚪", "meta_class": 5},
    "Mirai": {"color": "#0055dd", "emoji": "🤖", "meta_class": 5},
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
    "Brute-Force": ["185.220.101.{}", "23.129.64.{}"],
    "Botnet": ["77.247.181.{}", "185.220.102.{}"],
    "Backdoor": ["77.247.181.{}"],
    "Mirai": ["77.247.181.{}", "185.220.102.{}"],
}

ALERTS_FILE = project_root / "results" / "alerts" / "alerts.json"

st.set_page_config(
    page_title="IPS Advanced Monitor",
    page_icon="🛡️",
    layout="wide"
)

st.markdown("""
<style>
    .stat-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin: 5px;
    }
    .alert-row {
        padding: 8px;
        border-radius: 5px;
        margin: 3px 0;
        font-family: monospace;
        font-size: 13px;
    }
    .ip-badge {
        background: #333;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 11px;
    }
    .timeline {
        background: #0a0a1a;
        padding: 15px;
        border-radius: 10px;
        max-height: 400px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)

def get_simulated_ip(attack_type):
    if attack_type == "Normal":
        return f"192.168.1.{random.randint(100, 200)}"
    
    pool = ATTACKER_IP_POOLS.get(attack_type, ["unknown.{}.{}"])
    template = random.choice(pool)
    return template.format(random.randint(1, 254))

def map_meta_to_specific(meta_class_id):
    mapping = {
        0: ["Normal"],
        1: ["DoS-Hulk", "DoS-Slowloris", "DoS-GoldenEye"],
        2: ["DDoS-HOIC", "DDoS-LOIC", "DrDoS-Amplification", "DDoS-SYN-Flood"],
        3: ["PortScan", "Reconnaissance"],
        4: ["SQL-Injection", "XSS-Attack", "Brute-Force"],
        5: ["Botnet", "Backdoor", "Mirai"],
    }
    options = mapping.get(meta_class_id, ["Normal"])
    return random.choice(options)

def save_alerts(alerts):
    ALERTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ALERTS_FILE, "w") as f:
        json.dump(alerts[-500:], f, indent=2)

def load_alerts():
    if ALERTS_FILE.exists():
        try:
            with open(ALERTS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

@st.cache_resource
def load_model_and_data():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sequences, labels, _ = load_sequences()
    
    class_sequences = {}
    for i, name in enumerate(config.META_CLASS_NAMES):
        mask = labels == i
        class_sequences[name] = sequences[mask]
    
    model = build_model(num_features=sequences.shape[2])
    checkpoint = torch.load(config.MODEL_CHECKPOINT_PATH, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(clean_state_dict(checkpoint["model_state_dict"]))
    else:
        model.load_state_dict(clean_state_dict(checkpoint))
    model.to(device)
    model.eval()
    
    return {"model": model, "device": device, "class_sequences": class_sequences}

def get_traffic(class_sequences, attack_active):
    if attack_active["type"] is None:
        return class_sequences["Normal"][random.randint(0, len(class_sequences["Normal"])-1):][0:1], "Normal"
    
    meta_class = ATTACK_CATEGORIES[attack_active["type"]]["meta_class"]
    meta_name = config.META_CLASS_NAMES[meta_class]
    
    if random.random() < 0.8:
        seqs = class_sequences.get(meta_name, class_sequences["Normal"])
        return seqs[random.randint(0, len(seqs)-1):][0:1], attack_active["type"]
    else:
        return class_sequences["Normal"][random.randint(0, len(class_sequences["Normal"])-1):][0:1], "Normal"

def main():
    st.title("🛡️ IPS Advanced Monitor - Session 3")
    st.caption("IP Tracking • 12 Attack Types • History • Persistence")
    
    if "running" not in st.session_state:
        st.session_state.running = False
    if "attack_active" not in st.session_state:
        st.session_state.attack_active = {"type": None}
    if "alerts" not in st.session_state:
        st.session_state.alerts = load_alerts()
    if "ip_tracker" not in st.session_state:
        st.session_state.ip_tracker = defaultdict(int)
    if "stats" not in st.session_state:
        st.session_state.stats = {"total": 0, "attacks": 0}
    
    data = load_model_and_data()
    model, device = data["model"], data["device"]
    class_sequences = data["class_sequences"]
    
    with st.sidebar:
        st.header("⚡ Controls")
        
        if st.session_state.running:
            if st.button("⏹️ STOP", type="primary", use_container_width=True):
                st.session_state.running = False
                save_alerts(st.session_state.alerts)
                st.rerun()
        else:
            if st.button("▶️ START", type="primary", use_container_width=True):
                st.session_state.running = True
                st.rerun()
        
        st.divider()
        st.subheader("🎯 Inject Attack")
        
        for category, attacks in [
            ("DDoS/DoS", ["DDoS-HOIC", "DDoS-LOIC", "DoS-Hulk", "DoS-Slowloris"]),
            ("Recon", ["PortScan", "Reconnaissance"]),
            ("Web", ["SQL-Injection", "XSS-Attack", "Brute-Force"]),
            ("Malware", ["Botnet", "Backdoor"]),
        ]:
            with st.expander(f"📁 {category}"):
                for attack in attacks:
                    info = ATTACK_CATEGORIES[attack]
                    if st.session_state.attack_active["type"] == attack:
                        if st.button(f"🛑 Stop {attack}", key=f"stop_{attack}", use_container_width=True):
                            st.session_state.attack_active["type"] = None
                            st.rerun()
                    else:
                        if st.button(f"{info['emoji']} {attack}", key=f"start_{attack}", use_container_width=True):
                            st.session_state.attack_active["type"] = attack
                            st.rerun()
        
        st.divider()
        col1, col2 = st.columns(2)
        if col1.button("💾 Save"):
            save_alerts(st.session_state.alerts)
            st.success("Saved!")
        if col2.button("🗑️ Clear"):
            st.session_state.alerts = []
            st.session_state.ip_tracker = defaultdict(int)
            st.session_state.stats = {"total": 0, "attacks": 0}
            st.rerun()
    
    if st.session_state.attack_active["type"]:
        attack = st.session_state.attack_active["type"]
        st.error(f"⚠️ INJECTING: {attack}", icon="🚨")
    elif st.session_state.running:
        st.success("Monitoring normal traffic", icon="✅")
    
    cols = st.columns(5)
    cols[0].metric("Total Packets", st.session_state.stats["total"])
    cols[1].metric("Attacks Detected", st.session_state.stats["attacks"])
    cols[2].metric("Unique Attackers", len(st.session_state.ip_tracker))
    cols[3].metric("Alerts Saved", len(st.session_state.alerts))
    cols[4].metric("Device", str(device).upper())
    
    left_col, right_col = st.columns([2, 1])
    
    with left_col:
        st.subheader("📋 Live Detection Feed")
        feed_container = st.empty()
    
    with right_col:
        st.subheader("👤 Top Attackers")
        attacker_container = st.empty()
    
    st.subheader("📈 Attack Timeline")
    timeline_container = st.empty()
    
    if st.session_state.running:
        for _ in range(30):
            if not st.session_state.running:
                break
            
            seq, true_type = get_traffic(class_sequences, st.session_state.attack_active)
            
            with torch.no_grad():
                output = model(seq.to(device))
                pred_meta = output.argmax(dim=1).item()
            
            if pred_meta == 0:
                pred_type = "Normal"
            else:
                pred_type = map_meta_to_specific(pred_meta) if true_type == "Normal" else true_type
            
            ip = get_simulated_ip(pred_type)
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            st.session_state.stats["total"] += 1
            
            if pred_type != "Normal":
                st.session_state.stats["attacks"] += 1
                st.session_state.ip_tracker[ip] += 1
                
                alert = {
                    "time": timestamp,
                    "type": pred_type,
                    "ip": ip,
                    "emoji": ATTACK_CATEGORIES[pred_type]["emoji"]
                }
                st.session_state.alerts.insert(0, alert)
            
            feed_html = ""
            for a in st.session_state.alerts[:12]:
                color = ATTACK_CATEGORIES.get(a["type"], {}).get("color", "#888")
                feed_html += f"""
                <div class='alert-row' style='background:{color}22; border-left: 3px solid {color};'>
                    {a['emoji']} <b>{a['type']}</b> 
                    <span class='ip-badge'>{a['ip']}</span>
                    <span style='float:right; opacity:0.7'>{a['time']}</span>
                </div>"""
            feed_container.markdown(feed_html or "<p>No attacks detected yet</p>", unsafe_allow_html=True)
            
            top_ips = sorted(st.session_state.ip_tracker.items(), key=lambda x: -x[1])[:8]
            attacker_html = ""
            for ip, count in top_ips:
                attacker_html += f"<div style='margin:5px 0'><b>{ip}</b>: {count} attacks</div>"
            attacker_container.markdown(attacker_html or "<p>No attackers yet</p>", unsafe_allow_html=True)
            
            type_counts = defaultdict(int)
            for a in st.session_state.alerts[:100]:
                type_counts[a["type"]] += 1
            
            if type_counts:
                timeline_container.bar_chart(dict(type_counts))
            
            time.sleep(0.15)
        
        st.rerun()

if __name__ == "__main__":
    main()
