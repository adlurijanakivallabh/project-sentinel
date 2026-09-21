import sys
import glob
from pathlib import Path

import pandas as pd
import numpy as np

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src import config


LITNET_COLUMNS = [
    "id",
    "start_year", "start_month", "start_day", "start_hour", "start_min", "start_sec",
    "end_year", "end_month", "end_day", "end_hour", "end_min", "end_sec",
    "duration",
    "src_ip", "dst_ip", "src_port", "dst_port",
    "protocol",
    "flag_1", "flag_2", "flag_3", "flag_4", "flag_5", "flag_6",
    "fwd_status",
    "tos",
    "in_pkts", "in_bytes",
    "fwd_tos",
    "out_tos",
    "out_pkts", "out_bytes",
    "src_as", "dst_as",
    "input_interface", "output_interface",
    "src_mask", "dst_mask",
    "router_ip_1", "router_ip_2",
    "engine_type", "engine_id",
    "src_mac_1", "src_mac_2",
    "dst_mac_1", "dst_mac_2",
    "mpls_1", "mpls_2", "mpls_3", "mpls_4", "mpls_5",
    "mpls_6", "mpls_7", "mpls_8", "mpls_9", "mpls_10",
    "icmp_type_1", "icmp_type_2", "icmp_type_3",
    "next_hop_ip",
    "bgp_next_hop",
    "event_flag",
    "event_time",
    "fwd_class", "fwd_label_1",
    "fwd_label_2", "fwd_label_3",
    "fwd_label_4", "fwd_label_5",
    "fwd_label_6", "fwd_label_7",
    "fwd_label_8", "fwd_label_9",
    "fwd_label_10", "fwd_label_11",
    "fwd_label_12", "fwd_label_13",
    "fwd_label_14", "fwd_label_15",
    "fwd_label_16", "fwd_label_17",
    "tcp_udp_win_p_1",
    "fwd_label_18",
    "tcp_udp_win_p_2",
    "attack_label",
]

LITNET_ATTACK_MAP = {
    "HTTP_FLOOD": "DoS",
    "LAND_ATTACK": "DoS",

    "SYN_FLOOD": "DDoS",
    "UDP_FLOOD": "DDoS",
    "ICMP_FLOOD": "DDoS",
    "SMURF": "DDoS",
    "FRAGMENTATION": "DDoS",

    "BLASTER_WORM": "Malware/Botnet/Exploit",
    "REAPER_WORM": "Malware/Botnet/Exploit",
    "RED_WORM": "Malware/Botnet/Exploit",

    "SCANNING_SPREAD": "PortScan/Recon",

    "SPAM": "Malware/Botnet/Exploit",
}

LITNET_PROTO_MAP = {
    "TCP": 6, "UDP": 17, "ICMP": 1, "GRE": 47,
    "ESP": 50, "AH": 51, "IGMP": 2,
}


def _extract_attack_type(filename):
    name = Path(filename).stem
    for suffix in ["_v2_ATTACKERS_FLOWS", "_v2", "_ATTACKERS_FLOWS",
                    "_FLOWS", "_ATTACKERS_ONLY"]:
        name = name.replace(suffix, "")
    return name


def load_litnet(use_attackers_only=True, sample_frac=0.05, max_rows_per_file=500000,
                debug=False):
    data_dir = config.DATA_DIR / "litnet"
    all_files = sorted(glob.glob(str(data_dir / "*.csv")))

    if not all_files:
        print("[loader_litnet] LITNET not found, skipping.")
        return pd.DataFrame()

    if use_attackers_only:
        attacker_files = [f for f in all_files if "ATTACKER" in f.upper()]
        mixed_files = [f for f in all_files if "ATTACKER" not in f.upper()]
        files_to_load = attacker_files + mixed_files
    else:
        files_to_load = all_files

    print(f"[loader_litnet] Loading LITNET-2020 ({len(files_to_load)} files, "
          f"sample={sample_frac*100:.0f}%)...")

    df_list = []
    for f in files_to_load:
        try:
            fname = Path(f).name
            attack_type = _extract_attack_type(f)
            meta_label = LITNET_ATTACK_MAP.get(attack_type)

            if meta_label is None:
                if debug:
                    print(f"  {fname}: unknown attack type '{attack_type}', skipping")
                continue

            is_attacker = "ATTACKER" in fname.upper()

            n_cols = len(LITNET_COLUMNS)
            chunk = pd.read_csv(
                f, header=None, nrows=max_rows_per_file,
                low_memory=False, on_bad_lines="skip",
            )

            if chunk.shape[1] >= n_cols:
                chunk.columns = LITNET_COLUMNS[:chunk.shape[1]]
            elif chunk.shape[1] < n_cols:
                chunk.columns = LITNET_COLUMNS[:chunk.shape[1]]

            if sample_frac < 1.0 and len(chunk) > 10000:
                chunk = chunk.sample(frac=sample_frac, random_state=42)

            if is_attacker:
                chunk["Label"] = meta_label
            else:
                if "attack_label" in chunk.columns:
                    chunk["Label"] = chunk["attack_label"].apply(
                        lambda x: "Normal" if x == 0 else meta_label
                    )
                else:
                    chunk["Label"] = meta_label

            if "protocol" in chunk.columns:
                chunk["Protocol"] = (
                    chunk["protocol"].astype(str).str.upper()
                    .map(LITNET_PROTO_MAP).fillna(0).astype(int)
                )
            else:
                chunk["Protocol"] = 0

            chunk["Destination Port"] = pd.to_numeric(
                chunk.get("dst_port", 0), errors="coerce"
            ).fillna(0).astype(int)

            chunk["Flow Duration"] = pd.to_numeric(
                chunk.get("duration", 0), errors="coerce"
            ).fillna(0) * 1e6

            chunk["Total Fwd Packets"] = pd.to_numeric(
                chunk.get("in_pkts", 0), errors="coerce"
            ).fillna(0)
            chunk["Total Backward Packets"] = pd.to_numeric(
                chunk.get("out_pkts", 0), errors="coerce"
            ).fillna(0)

            chunk["Total Length of Fwd Packets"] = pd.to_numeric(
                chunk.get("in_bytes", 0), errors="coerce"
            ).fillna(0)
            chunk["Total Length of Bwd Packets"] = pd.to_numeric(
                chunk.get("out_bytes", 0), errors="coerce"
            ).fillna(0)

            fwd_pkts = chunk["Total Fwd Packets"].clip(lower=1)
            bwd_pkts = chunk["Total Backward Packets"].clip(lower=1)

            chunk["Fwd Packet Length Mean"] = (
                chunk["Total Length of Fwd Packets"] / fwd_pkts
            )
            chunk["Fwd Packet Length Max"] = chunk["Fwd Packet Length Mean"]
            chunk["Fwd Packet Length Min"] = chunk["Fwd Packet Length Mean"]
            chunk["Fwd Packet Length Std"] = 0

            chunk["Bwd Packet Length Mean"] = (
                chunk["Total Length of Bwd Packets"] / bwd_pkts
            )
            chunk["Bwd Packet Length Max"] = chunk["Bwd Packet Length Mean"]
            chunk["Bwd Packet Length Min"] = chunk["Bwd Packet Length Mean"]
            chunk["Bwd Packet Length Std"] = 0

            dur_sec = chunk["Flow Duration"].clip(lower=1) / 1e6
            total_bytes = (chunk["Total Length of Fwd Packets"] +
                           chunk["Total Length of Bwd Packets"])
            total_pkts = (chunk["Total Fwd Packets"] +
                          chunk["Total Backward Packets"])

            chunk["Flow Bytes/s"] = total_bytes / dur_sec
            chunk["Flow Packets/s"] = total_pkts / dur_sec

            chunk["Flow IAT Mean"] = chunk["Flow Duration"] / total_pkts.clip(lower=1)
            chunk["Flow IAT Std"] = 0
            chunk["Flow IAT Max"] = chunk["Flow IAT Mean"]
            chunk["Flow IAT Min"] = chunk["Flow IAT Mean"]

            chunk["Fwd PSH Flags"] = 0
            chunk["Bwd PSH Flags"] = 0
            chunk["Fwd URG Flags"] = 0
            chunk["Bwd URG Flags"] = 0

            for col in config.FEATURE_COLUMNS:
                if col not in chunk.columns:
                    chunk[col] = 0

            cols = config.FEATURE_COLUMNS + ["Label"]
            chunk = chunk[[c for c in cols if c in chunk.columns]]

            df_list.append(chunk)

            if debug:
                print(f"  {fname}: {len(chunk):,} rows → {meta_label}")

        except Exception as e:
            print(f"  Skipping {Path(f).name}: {e}")
            continue

    if not df_list:
        return pd.DataFrame()

    df = pd.concat(df_list, ignore_index=True)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    print(f"[loader_litnet] → {len(df):,} rows loaded")
    if debug:
        print(f"  Labels: {df['Label'].value_counts().to_dict()}")

    return df


if __name__ == "__main__":
    df = load_litnet(use_attackers_only=True, sample_frac=0.1, debug=True)
    if not df.empty:
        print(f"\nShape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nLabel distribution:")
        print(df["Label"].value_counts())
