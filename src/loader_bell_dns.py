import sys
import glob
from pathlib import Path

import pandas as pd
import numpy as np

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src import config


BELL_DNS_LABEL_MAP = {
    "Heavy Benign 3": "Normal",
    "Light Benign": "Normal",
    "BenignExtra": "Normal",
    "Heavy Attack text": "Malware/Botnet/Exploit",
    "Heavy Attack video": "Malware/Botnet/Exploit",
    "Heavy Attack image": "Malware/Botnet/Exploit",
    "Heavy Attack audio": "Malware/Botnet/Exploit",
    "Heavy Attack compressed": "Malware/Botnet/Exploit",
    "Heavy Attack exe": "Malware/Botnet/Exploit",
    "Light Attack text": "Malware/Botnet/Exploit",
    "Light Attack video": "Malware/Botnet/Exploit",
    "Light Attack image": "Malware/Botnet/Exploit",
    "Light Attack audio": "Malware/Botnet/Exploit",
    "Light Attack compressed": "Malware/Botnet/Exploit",
    "Light Attack exe": "Malware/Botnet/Exploit",
}

DNS_FEATURE_MAP = {
    "rr_count": "Total Fwd Packets",
    "FQDN_count": "Total Backward Packets",
    "subdomain_length": "Fwd Packet Length Mean",
    "len": "Fwd Packet Length Max",
    "labels": "Total Length of Fwd Packets",
    "labels_max": "Total Length of Bwd Packets",
    "labels_average": "Bwd Packet Length Mean",
    "entropy": "Flow IAT Mean",
    "rr_name_entropy": "Flow IAT Std",
    "rr_name_length": "Flow IAT Max",
    "numeric": "Flow IAT Min",
    "upper": "Fwd PSH Flags",
    "lower": "Bwd PSH Flags",
}


def load_bell_dns(sample_frac=1.0, debug=False):
    data_dir = config.DATA_DIR / "bell_dns"
    all_files = sorted(glob.glob(str(data_dir / "*.parquet")))

    if not all_files:
        print("[loader_bell_dns] Bell-DNS not found, skipping.")
        return pd.DataFrame()

    print(f"[loader_bell_dns] Loading Bell-DNS ({len(all_files)} files)...")

    df_list = []
    for f in all_files:
        try:
            chunk = pd.read_parquet(f)

            if debug:
                print(f"  {Path(f).name}: {len(chunk):,} rows, "
                      f"cols={list(chunk.columns[:5])}...")

            label_col = None
            for candidate in ["SubClass", "GlobalClass", "label"]:
                if candidate in chunk.columns:
                    label_col = candidate
                    break

            if label_col is None:
                print(f"  {Path(f).name}: no label column found, skipping")
                continue

            chunk["Label"] = chunk[label_col].map(BELL_DNS_LABEL_MAP)

            if "GlobalClass" in chunk.columns:
                mask = chunk["Label"].isna()
                chunk.loc[mask & chunk["GlobalClass"].str.contains("Benign", na=False), "Label"] = "Normal"
                chunk.loc[mask & chunk["GlobalClass"].str.contains("Attack", na=False), "Label"] = "Malware/Botnet/Exploit"

            chunk = chunk.dropna(subset=["Label"])

            for dns_col, cic_col in DNS_FEATURE_MAP.items():
                if dns_col in chunk.columns:
                    chunk[cic_col] = pd.to_numeric(
                        chunk[dns_col], errors="coerce"
                    ).fillna(0)

            for col in config.FEATURE_COLUMNS:
                if col not in chunk.columns:
                    chunk[col] = 0

            chunk["Protocol"] = 17
            chunk["Destination Port"] = 53

            cols = config.FEATURE_COLUMNS + ["Label"]
            chunk = chunk[[c for c in cols if c in chunk.columns]]

            if sample_frac < 1.0:
                chunk = chunk.sample(frac=sample_frac, random_state=42)

            df_list.append(chunk)

        except Exception as e:
            print(f"  Skipping {Path(f).name}: {e}")
            continue

    if not df_list:
        return pd.DataFrame()

    df = pd.concat(df_list, ignore_index=True)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    print(f"[loader_bell_dns] → {len(df):,} rows loaded")
    if debug:
        print(f"  Labels: {df['Label'].value_counts().to_dict()}")

    return df


if __name__ == "__main__":
    df = load_bell_dns(debug=True)
    if not df.empty:
        print(f"\nShape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nLabel distribution:")
        print(df["Label"].value_counts())
