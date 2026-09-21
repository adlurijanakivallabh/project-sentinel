import sys
import glob
import gc
from pathlib import Path

import pandas as pd
import numpy as np

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src import config


BOTIOT_TO_CIC = {
    "dport": "Destination Port",
    "proto": "Protocol",
    "dur": "Flow Duration",
    "spkts": "Total Fwd Packets",
    "dpkts": "Total Backward Packets",
    "sbytes": "Total Length of Fwd Packets",
    "dbytes": "Total Length of Bwd Packets",
    "rate": "Flow Packets/s",
    "srate": "Flow Bytes/s",
}

BOTIOT_PROTO_MAP = {
    "tcp": 6, "udp": 17, "icmp": 1, "arp": 0,
    "igmp": 2, "rtp": 0, "ipv6-icmp": 58,
}

BOTIOT_LABEL_MAP = {
    "Normal": "Normal",
    "DDoS": "DDoS",
    "DoS": "DoS",
    "Reconnaissance": "PortScan/Recon",
    "Theft": "Malware/Botnet/Exploit",
}


def load_botiot(max_files=None, sample_frac=0.1, debug=False):
    data_dir = config.DATA_DIR / "bot_iot"
    all_files = sorted(glob.glob(str(data_dir / "data_*.csv")))

    if not all_files:
        print("[loader_botiot] Bot-IoT not found, skipping.")
        return pd.DataFrame()

    if max_files:
        all_files = all_files[:max_files]

    print(f"[loader_botiot] Loading Bot-IoT ({len(all_files)} files, "
          f"sample={sample_frac*100:.0f}%)...")

    df_list = []
    for i, f in enumerate(all_files):
        try:
            chunk = pd.read_csv(f, low_memory=False)

            if sample_frac < 1.0 and len(chunk) > 1000:
                chunk = chunk.sample(frac=sample_frac, random_state=42 + i)

            chunk.columns = [c.strip() for c in chunk.columns]

            if "category" in chunk.columns:
                chunk["Label"] = chunk["category"].map(BOTIOT_LABEL_MAP)
                chunk = chunk.dropna(subset=["Label"])
            else:
                if debug:
                    print(f"  {Path(f).name}: no 'category' column, skipping")
                continue

            if "proto" in chunk.columns:
                chunk["Protocol"] = (chunk["proto"].astype(str).str.lower()
                                     .map(BOTIOT_PROTO_MAP).fillna(0).astype(int))
            else:
                chunk["Protocol"] = 0

            chunk["Destination Port"] = pd.to_numeric(
                chunk.get("dport", 0), errors="coerce"
            ).fillna(0).astype(int)

            chunk["Flow Duration"] = pd.to_numeric(
                chunk.get("dur", 0), errors="coerce"
            ).fillna(0) * 1e6

            chunk["Total Fwd Packets"] = pd.to_numeric(
                chunk.get("spkts", 0), errors="coerce"
            ).fillna(0)
            chunk["Total Backward Packets"] = pd.to_numeric(
                chunk.get("dpkts", 0), errors="coerce"
            ).fillna(0)

            chunk["Total Length of Fwd Packets"] = pd.to_numeric(
                chunk.get("sbytes", 0), errors="coerce"
            ).fillna(0)
            chunk["Total Length of Bwd Packets"] = pd.to_numeric(
                chunk.get("dbytes", 0), errors="coerce"
            ).fillna(0)

            total_fwd = chunk["Total Fwd Packets"].clip(lower=1)
            total_bwd = chunk["Total Backward Packets"].clip(lower=1)

            chunk["Fwd Packet Length Mean"] = (
                chunk["Total Length of Fwd Packets"] / total_fwd
            )
            chunk["Fwd Packet Length Max"] = chunk["Fwd Packet Length Mean"]
            chunk["Fwd Packet Length Min"] = chunk["Fwd Packet Length Mean"]
            chunk["Fwd Packet Length Std"] = pd.to_numeric(
                chunk.get("stddev", 0), errors="coerce"
            ).fillna(0)

            chunk["Bwd Packet Length Mean"] = (
                chunk["Total Length of Bwd Packets"] / total_bwd
            )
            chunk["Bwd Packet Length Max"] = chunk["Bwd Packet Length Mean"]
            chunk["Bwd Packet Length Min"] = chunk["Bwd Packet Length Mean"]
            chunk["Bwd Packet Length Std"] = 0

            chunk["Flow Bytes/s"] = pd.to_numeric(
                chunk.get("srate", 0), errors="coerce"
            ).fillna(0)
            chunk["Flow Packets/s"] = pd.to_numeric(
                chunk.get("rate", 0), errors="coerce"
            ).fillna(0)

            total_pkts = (chunk["Total Fwd Packets"] +
                          chunk["Total Backward Packets"]).clip(lower=1)
            chunk["Flow IAT Mean"] = chunk["Flow Duration"] / total_pkts
            chunk["Flow IAT Std"] = 0
            chunk["Flow IAT Max"] = chunk["Flow IAT Mean"]
            chunk["Flow IAT Min"] = chunk["Flow IAT Mean"]

            chunk["Fwd PSH Flags"] = 0
            chunk["Bwd PSH Flags"] = 0
            chunk["Fwd URG Flags"] = 0
            chunk["Bwd URG Flags"] = 0

            available = [c for c in config.FEATURE_COLUMNS if c in chunk.columns]
            available.append("Label")
            chunk = chunk[available]

            for col in config.FEATURE_COLUMNS:
                if col not in chunk.columns:
                    chunk[col] = 0

            df_list.append(chunk)

            if debug and (i + 1) % 10 == 0:
                print(f"  Loaded {i+1}/{len(all_files)} files...")

        except Exception as e:
            print(f"  Skipping {Path(f).name}: {e}")
            continue

    if not df_list:
        return pd.DataFrame()

    df = pd.concat(df_list, ignore_index=True)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    print(f"[loader_botiot] → {len(df):,} rows loaded")
    if debug:
        print(f"  Labels: {df['Label'].value_counts().to_dict()}")

    gc.collect()
    return df


if __name__ == "__main__":
    df = load_botiot(max_files=3, sample_frac=0.05, debug=True)
    if not df.empty:
        print(f"\nShape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nLabel distribution:")
        print(df["Label"].value_counts())
