import glob
import warnings
import pandas as pd
import numpy as np
import gc
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import joblib

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from . import config
    from .loader_botiot import load_botiot
    from .loader_bell_dns import load_bell_dns
    from .loader_litnet import load_litnet
except ImportError:
    import config
    from loader_botiot import load_botiot
    from loader_bell_dns import load_bell_dns
    from loader_litnet import load_litnet


def load_cicids2017(debug=False):
    print("[preprocess] Loading CICIDS2017...")
    all_files = glob.glob(str(config.DATA_CICIDS2017_DIR / "*.csv"))
    if not all_files:
        print("[preprocess]   → Not found, skipping.")
        return pd.DataFrame()

    df_list = []
    for f in all_files:
        try:
            temp = pd.read_csv(
                f,
                usecols=lambda c: (c.strip() in config.FEATURE_COLUMNS
                                   or c.strip() in ("Label", " Label")),
            )
            temp.columns = [c.strip() for c in temp.columns]
            df_list.append(temp)
        except Exception as e:
            print(f"  Skipping {Path(f).name}: {e}")

    if not df_list:
        return pd.DataFrame()

    df = pd.concat(df_list, ignore_index=True)
    print(f"[preprocess]   → {len(df):,} rows loaded (100% — preserving rare classes)")
    return df


def load_cse_cic_ids2018(debug=False):
    sample_frac = config.CSE_CIC_2018_SAMPLE_FRAC
    print(f"[preprocess] Loading CSE-CIC-IDS2018 (Sampled {sample_frac*100:.0f}%)...")
    all_files = glob.glob(str(config.DATA_CSE_CIC_IDS2018_DIR / "*.csv"))
    if not all_files:
        print("[preprocess]   → Not found, skipping.")
        return pd.DataFrame()

    df_list = []
    rename_map = {
        "Dst Port": "Destination Port",
        "Tot Fwd Pkts": "Total Fwd Packets",
        "Tot Bwd Pkts": "Total Backward Packets",
        "TotLen Fwd Pkts": "Total Length of Fwd Packets",
        "TotLen Bwd Pkts": "Total Length of Bwd Packets",
        "Fwd Pkt Len Max": "Fwd Packet Length Max",
        "Fwd Pkt Len Min": "Fwd Packet Length Min",
        "Fwd Pkt Len Mean": "Fwd Packet Length Mean",
        "Fwd Pkt Len Std": "Fwd Packet Length Std",
        "Bwd Pkt Len Max": "Bwd Packet Length Max",
        "Bwd Pkt Len Min": "Bwd Packet Length Min",
        "Bwd Pkt Len Mean": "Bwd Packet Length Mean",
        "Bwd Pkt Len Std": "Bwd Packet Length Std",
        "Flow Byts/s": "Flow Bytes/s",
        "Flow Pkts/s": "Flow Packets/s",
    }

    for f in all_files:
        try:
            chunks = pd.read_csv(f, chunksize=100_000)
            for chunk in chunks:
                if len(chunk) > 0:
                    chunk = chunk.sample(frac=sample_frac, random_state=42)
                chunk.columns = [c.strip() for c in chunk.columns]
                chunk.rename(columns=rename_map, inplace=True)

                available_cols = [c for c in config.FEATURE_COLUMNS if c in chunk.columns]
                if "Label" in chunk.columns:
                    available_cols.append("Label")
                chunk = chunk[available_cols]

                for col in config.FEATURE_COLUMNS:
                    if col not in chunk.columns:
                        chunk[col] = 0

                df_list.append(chunk)
            del chunks
            gc.collect()
        except Exception as e:
            print(f"  Skipping {Path(f).name}: {e}")
            continue

    if not df_list:
        return pd.DataFrame()

    df = pd.concat(df_list, ignore_index=True)
    print(f"[preprocess]   → {len(df):,} rows loaded")
    return df


def load_unsw_nb15(debug=False):
    print("[preprocess] Loading UNSW-NB15 (Sampled 20%)...")
    all_files = sorted(glob.glob(str(config.DATA_UNSW_NB15_DIR / "UNSW-NB15_*.csv")))
    raw_files = [f for f in all_files
                 if "List_Events" not in f and "LIST_EVENTS" not in f
                 and "features" not in f and "training" not in f and "testing" not in f]

    if not raw_files:
        for alt in ("UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv"):
            p = config.DATA_UNSW_NB15_DIR / alt
            if p.exists():
                raw_files.append(str(p))
        if not raw_files:
            print("[preprocess]   → Not found, skipping.")
            return pd.DataFrame()

    unsw_cols = [
        "srcip", "sport", "dstip", "dsport", "proto", "state", "dur",
        "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss", "service",
        "sload", "dload", "spkts", "dpkts", "swin", "dwin", "stcpb", "dtcpb",
        "smeansz", "dmeansz", "trans_depth", "res_bdy_len", "sjit", "djit",
        "stime", "ltime", "sintpkt", "dintpkt", "tcprtt", "synack", "ackdat",
        "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login",
        "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
        "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat", "Label",
    ]

    df_list = []
    for f in raw_files:
        try:
            is_preprocessed = "training-set" in f or "testing-set" in f
            temp = pd.read_csv(f) if is_preprocessed else pd.read_csv(f, names=unsw_cols, header=None)

            if len(temp) > 0:
                temp = temp.sample(frac=0.2, random_state=42)

            temp["Destination Port"] = pd.to_numeric(temp.get("dsport", 0), errors="coerce").fillna(0)

            proto_map = {"tcp": 6, "udp": 17, "icmp": 1}
            if "proto" in temp.columns:
                temp["Protocol"] = temp["proto"].astype(str).str.lower().map(proto_map).fillna(0)
            else:
                temp["Protocol"] = 0

            temp["Flow Duration"] = temp.get("dur", 0) * 1e6

            temp["Total Fwd Packets"] = temp.get("spkts", 0)
            temp["Total Backward Packets"] = temp.get("dpkts", 0)
            temp["Total Length of Fwd Packets"] = temp.get("sbytes", 0)
            temp["Total Length of Bwd Packets"] = temp.get("dbytes", 0)

            temp["Fwd Packet Length Mean"] = temp.get("smeansz", 0)
            temp["Fwd Packet Length Max"] = temp.get("smeansz", 0)
            temp["Fwd Packet Length Min"] = temp.get("smeansz", 0)
            temp["Fwd Packet Length Std"] = 0

            temp["Bwd Packet Length Mean"] = temp.get("dmeansz", 0)
            temp["Bwd Packet Length Max"] = temp.get("dmeansz", 0)
            temp["Bwd Packet Length Min"] = temp.get("dmeansz", 0)
            temp["Bwd Packet Length Std"] = 0

            sint = temp.get("sintpkt", 0)
            dint = temp.get("dintpkt", 0)
            if isinstance(sint, pd.Series):
                temp["Flow IAT Mean"] = (sint + dint) / 2 * 1000
                temp["Flow IAT Std"] = temp.get("sjit", 0) * 1000
                temp["Flow IAT Max"] = temp["Flow IAT Mean"]
                temp["Flow IAT Min"] = temp["Flow IAT Mean"]
            else:
                temp["Flow IAT Mean"] = 0
                temp["Flow IAT Std"] = 0
                temp["Flow IAT Max"] = 0
                temp["Flow IAT Min"] = 0

            temp["Fwd PSH Flags"] = 0
            temp["Bwd PSH Flags"] = 0
            temp["Fwd URG Flags"] = 0
            temp["Bwd URG Flags"] = 0

            dur = temp.get("dur", pd.Series([1e-9]))
            sb = temp.get("sbytes", 0)
            db = temp.get("dbytes", 0)
            sp = temp.get("spkts", 0)
            dp = temp.get("dpkts", 0)
            if isinstance(dur, pd.Series):
                temp["Flow Bytes/s"] = (sb + db) / (dur + 1e-9)
                temp["Flow Packets/s"] = (sp + dp) / (dur + 1e-9)
            else:
                temp["Flow Bytes/s"] = 0
                temp["Flow Packets/s"] = 0

            if "attack_cat" in temp.columns:
                temp["Label"] = temp["attack_cat"].fillna("Normal").astype(str).str.strip()
            else:
                temp["Label"] = "Normal"

            keep = [c for c in config.FEATURE_COLUMNS + ["Label"] if c in temp.columns]
            df_list.append(temp[keep])

        except Exception as e:
            print(f"  Skipping {Path(f).name}: {e}")
            continue

    if not df_list:
        return pd.DataFrame()

    df = pd.concat(df_list, ignore_index=True)
    print(f"[preprocess]   → {len(df):,} rows loaded")
    return df


def load_cic_ddos2019(debug=False):
    print("[preprocess] Loading CIC-DDoS2019...")
    ddos_dir = config.DATA_DIR / "cicddos2019"
    all_files = sorted(glob.glob(str(ddos_dir / "*.parquet")))
    if not all_files:
        print("[preprocess]   → Not found, skipping.")
        return pd.DataFrame()

    rename_map = {
        "Flow Duration": "Flow Duration",
        "Total Fwd Packets": "Total Fwd Packets",
        "Total Backward Packets": "Total Backward Packets",
        "Fwd Packets Length Total": "Total Length of Fwd Packets",
        "Bwd Packets Length Total": "Total Length of Bwd Packets",
        "Fwd Packet Length Max": "Fwd Packet Length Max",
        "Fwd Packet Length Min": "Fwd Packet Length Min",
        "Fwd Packet Length Mean": "Fwd Packet Length Mean",
        "Fwd Packet Length Std": "Fwd Packet Length Std",
        "Bwd Packet Length Max": "Bwd Packet Length Max",
        "Bwd Packet Length Min": "Bwd Packet Length Min",
        "Bwd Packet Length Mean": "Bwd Packet Length Mean",
        "Bwd Packet Length Std": "Bwd Packet Length Std",
        "Flow Bytes/s": "Flow Bytes/s",
        "Flow Packets/s": "Flow Packets/s",
        "Flow IAT Mean": "Flow IAT Mean",
        "Flow IAT Std": "Flow IAT Std",
        "Flow IAT Max": "Flow IAT Max",
        "Flow IAT Min": "Flow IAT Min",
        "Fwd PSH Flags": "Fwd PSH Flags",
        "Bwd PSH Flags": "Bwd PSH Flags",
        "Fwd URG Flags": "Fwd URG Flags",
        "Bwd URG Flags": "Bwd URG Flags",
    }

    df_list = []
    for f in all_files:
        try:
            temp = pd.read_parquet(f)
            temp.columns = [c.strip() for c in temp.columns]
            temp.rename(columns=rename_map, inplace=True)

            if "Protocol" in temp.columns:
                temp["Protocol"] = pd.to_numeric(temp["Protocol"], errors="coerce").fillna(0)

            if "Destination Port" not in temp.columns:
                temp["Destination Port"] = 0

            for col in config.FEATURE_COLUMNS:
                if col not in temp.columns:
                    temp[col] = 0

            keep = [c for c in config.FEATURE_COLUMNS + ["Label"] if c in temp.columns]
            temp = temp[keep]

            if len(temp) > 10000:
                temp = temp.sample(frac=0.2, random_state=42)

            df_list.append(temp)
        except Exception as e:
            print(f"  Skipping {Path(f).name}: {e}")
            continue

    if not df_list:
        return pd.DataFrame()

    df = pd.concat(df_list, ignore_index=True)
    print(f"[preprocess]   → {len(df):,} rows loaded")
    return df


def load_ciciot2023(debug=False):
    print("[preprocess] Loading CICIoT2023...")
    csv_dir = config.DATA_DIR / "ciciot2023" / "wataiData" / "csv" / "CICIoT2023"
    if not csv_dir.exists():
        csv_dir = config.DATA_DIR / "ciciot2023"
    
    all_files = sorted(glob.glob(str(csv_dir / "part-*.csv")))
    if not all_files:
        print("[preprocess]   → Not found, skipping.")
        return pd.DataFrame()

    print(f"[preprocess]   Found {len(all_files)} part files")

    np.random.seed(42)
    sample_files = np.random.choice(all_files, size=min(30, len(all_files)), replace=False)
    print(f"[preprocess]   Sampling {len(sample_files)} files (~18% of data)")

    df_list = []
    for f in sample_files:
        try:
            raw = pd.read_csv(f)
            raw.columns = [c.strip() for c in raw.columns]

            mapped = pd.DataFrame()

            mapped["Destination Port"] = 0

            if "Protocol Type" in raw.columns:
                mapped["Protocol"] = pd.to_numeric(raw["Protocol Type"], errors="coerce").fillna(0)
            elif "TCP" in raw.columns and "UDP" in raw.columns:
                mapped["Protocol"] = np.where(
                    raw.get("TCP", 0) == 1, 6,
                    np.where(raw.get("UDP", 0) == 1, 17,
                             np.where(raw.get("ICMP", 0) == 1, 1, 0)))
            else:
                mapped["Protocol"] = 0

            if "flow_duration" in raw.columns:
                mapped["Flow Duration"] = pd.to_numeric(raw["flow_duration"], errors="coerce").fillna(0) * 1e6
            elif "Duration" in raw.columns:
                mapped["Flow Duration"] = pd.to_numeric(raw["Duration"], errors="coerce").fillna(0) * 1e6
            else:
                mapped["Flow Duration"] = 0

            total_pkts = pd.to_numeric(raw.get("Number", 0), errors="coerce").fillna(0)
            mapped["Total Fwd Packets"] = (total_pkts * 0.6).astype(int)
            mapped["Total Backward Packets"] = (total_pkts * 0.4).astype(int)

            tot_bytes = pd.to_numeric(raw.get("Tot sum", raw.get("Tot size", 0)), errors="coerce").fillna(0)
            mapped["Total Length of Fwd Packets"] = (tot_bytes * 0.6).astype(int)
            mapped["Total Length of Bwd Packets"] = (tot_bytes * 0.4).astype(int)

            avg_pkt = pd.to_numeric(raw.get("AVG", 0), errors="coerce").fillna(0)
            min_pkt = pd.to_numeric(raw.get("Min", 0), errors="coerce").fillna(0)
            max_pkt = pd.to_numeric(raw.get("Max", 0), errors="coerce").fillna(0)
            std_pkt = pd.to_numeric(raw.get("Std", 0), errors="coerce").fillna(0)

            mapped["Fwd Packet Length Max"] = max_pkt
            mapped["Fwd Packet Length Min"] = min_pkt
            mapped["Fwd Packet Length Mean"] = avg_pkt
            mapped["Fwd Packet Length Std"] = std_pkt
            mapped["Bwd Packet Length Max"] = max_pkt * 0.8
            mapped["Bwd Packet Length Min"] = min_pkt
            mapped["Bwd Packet Length Mean"] = avg_pkt * 0.8
            mapped["Bwd Packet Length Std"] = std_pkt * 0.8

            iat = pd.to_numeric(raw.get("IAT", 0), errors="coerce").fillna(0)
            mapped["Flow IAT Mean"] = iat * 1000
            mapped["Flow IAT Std"] = pd.to_numeric(raw.get("Variance", 0), errors="coerce").fillna(0).apply(np.sqrt) * 1000
            mapped["Flow IAT Max"] = iat * 1500
            mapped["Flow IAT Min"] = iat * 500

            mapped["Fwd PSH Flags"] = (pd.to_numeric(raw.get("psh_flag_number", 0), errors="coerce").fillna(0) > 0).astype(int)
            mapped["Bwd PSH Flags"] = 0
            mapped["Fwd URG Flags"] = (pd.to_numeric(raw.get("urg_count", 0), errors="coerce").fillna(0) > 0).astype(int)
            mapped["Bwd URG Flags"] = 0

            rate = pd.to_numeric(raw.get("Rate", 0), errors="coerce").fillna(0)
            duration_sec = mapped["Flow Duration"] / 1e6 + 1e-9
            mapped["Flow Bytes/s"] = tot_bytes / duration_sec
            mapped["Flow Packets/s"] = rate

            if "label" in raw.columns:
                mapped["Label"] = raw["label"].astype(str).str.strip()
                mapped["Label"] = mapped["Label"].replace("BenignTraffic", "BENIGN")
            else:
                mapped["Label"] = "BENIGN"

            if len(mapped) > 10000:
                mapped = mapped.sample(frac=0.2, random_state=42)

            df_list.append(mapped)

        except Exception as e:
            print(f"  Skipping {Path(f).name}: {e}")
            continue

    if not df_list:
        return pd.DataFrame()

    df = pd.concat(df_list, ignore_index=True)
    print(f"[preprocess]   → {len(df):,} rows loaded")
    print(f"[preprocess]   Labels: {sorted(df['Label'].unique())}")
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    print(f"[preprocess] Cleaning {len(df):,} rows...")
    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    keep_cols = [c for c in config.FEATURE_COLUMNS + ["Label"] if c in df.columns]
    df = df[keep_cols]

    for c in config.FEATURE_COLUMNS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df.dropna(inplace=True)
    print(f"[preprocess]   → {len(df):,} rows after cleaning")
    return df


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    label_map = config.get_label_map()
    df = df.copy()
    df["meta_label"] = df["Label"].map(
        lambda x: label_map.get(str(x).strip(), "Normal") if isinstance(x, str) else "Normal"
    )
    df["class_id"] = df["meta_label"].map(config.meta_label_to_id)
    return df


def remove_noise_isolation_forest(df: pd.DataFrame) -> pd.DataFrame:
    if not config.USE_ISOLATION_FOREST:
        return df

    try:
        from sklearn.ensemble import IsolationForest
    except ImportError:
        print("[preprocess] scikit-learn IsolationForest not available, skipping noise removal.")
        return df

    print(f"[preprocess] Removing noise with Isolation Forest (contamination={config.ISOLATION_FOREST_CONTAMINATION})...")
    feature_cols = [c for c in config.FEATURE_COLUMNS if c in df.columns]
    cleaned_dfs = []

    for class_name, group in df.groupby("meta_label"):
        if len(group) < 50:
            cleaned_dfs.append(group)
            continue

        iso = IsolationForest(
            contamination=config.ISOLATION_FOREST_CONTAMINATION,
            random_state=42,
            n_jobs=-1,
        )
        X = group[feature_cols].values
        preds = iso.fit_predict(X)
        clean = group[preds == 1]
        removed = len(group) - len(clean)
        if removed > 0:
            print(f"  {class_name}: removed {removed:,} noisy samples ({removed/len(group)*100:.1f}%)")
        cleaned_dfs.append(clean)

    result = pd.concat(cleaned_dfs, ignore_index=True)
    print(f"[preprocess]   → {len(result):,} rows after noise removal")
    return result


def balance_dataset(df: pd.DataFrame, target_count: int = 50000) -> pd.DataFrame:
    if not config.USE_BORDERLINE_SMOTE:
        print("[preprocess] Balancing DISABLED (class-weighted loss handles imbalance in train.py)")
        print("[preprocess]   → Downsampling very large classes only (>500K)")
        feature_cols = [c for c in config.FEATURE_COLUMNS if c in df.columns]
        class_counts = df["class_id"].value_counts()
        max_allowed = 500_000
        if class_counts.max() > max_allowed:
            keep_dfs = []
            for cls_id, group in df.groupby("class_id"):
                if len(group) > max_allowed:
                    print(f"  Downsampling class {cls_id} from {len(group):,} to {max_allowed:,}")
                    keep_dfs.append(group.sample(n=max_allowed, random_state=42))
                else:
                    keep_dfs.append(group)
            df = pd.concat(keep_dfs, ignore_index=True)
        print(f"[preprocess]   → {len(df):,} rows (natural distribution preserved)")
        return df
    
    print("[preprocess] Balancing classes with Borderline-SMOTE...")
    return _balance_borderline_smote(df, target_count)


def _balance_borderline_smote(df: pd.DataFrame, target_count: int) -> pd.DataFrame:
    try:
        from imblearn.over_sampling import BorderlineSMOTE
    except ImportError:
        print("[preprocess] imbalanced-learn not installed. Falling back to random oversample.")
        print("[preprocess] Install with: pip install imbalanced-learn")
        return _balance_random_oversample(df, target_count)

    feature_cols = [c for c in config.FEATURE_COLUMNS if c in df.columns]
    X = df[feature_cols].values
    y = df["class_id"].values

    class_counts = pd.Series(y).value_counts()
    max_allowed = 500_000

    if class_counts.max() > max_allowed:
        keep_mask = np.ones(len(y), dtype=bool)
        for cls_id, count in class_counts.items():
            if count > max_allowed:
                cls_indices = np.where(y == cls_id)[0]
                drop_n = count - max_allowed
                drop_indices = np.random.RandomState(42).choice(cls_indices, drop_n, replace=False)
                keep_mask[drop_indices] = False
                print(f"  Downsampling class {cls_id} from {count:,} to {max_allowed:,}")
        X = X[keep_mask]
        y = y[keep_mask]
        df = df[keep_mask].copy()

    class_counts = pd.Series(y).value_counts()
    sampling_strategy = {}
    for cls_id, count in class_counts.items():
        if count < target_count:
            sampling_strategy[cls_id] = target_count

    if not sampling_strategy:
        print("[preprocess]   → All classes already above target, no SMOTE needed.")
        return df

    print("[preprocess] Applying Borderline-SMOTE (k=5)...")
    try:
        smote = BorderlineSMOTE(
            sampling_strategy=sampling_strategy,
            random_state=42,
            k_neighbors=min(5, min(class_counts.values) - 1),
        )
        X_res, y_res = smote.fit_resample(X, y)
    except Exception as e:
        print(f"[preprocess] Borderline-SMOTE failed: {e}. Falling back to random oversample.")
        return _balance_random_oversample(df, target_count)

    result = pd.DataFrame(X_res, columns=feature_cols)
    result["class_id"] = y_res
    class_names = config.get_class_names()
    result["meta_label"] = result["class_id"].map(
        lambda x: class_names[x] if 0 <= x < len(class_names) else "Normal"
    )
    result["Label"] = result["meta_label"]

    print(f"[preprocess]   → {len(result):,} rows after Borderline-SMOTE")
    for cls_id in sorted(result["class_id"].unique()):
        name = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"
        count = (result["class_id"] == cls_id).sum()
        print(f"      {name}: {count:,}")

    return result


def _balance_random_oversample(df: pd.DataFrame, target_count: int) -> pd.DataFrame:
    groups = df.groupby("meta_label")
    balanced_dfs = []

    for name, group in groups:
        count = len(group)
        if count < target_count:
            print(f"  Oversampling {name} ({count:,} → {target_count:,})")
            over = group.sample(n=target_count, replace=True, random_state=42)
            balanced_dfs.append(over)
        elif count > 500_000:
            print(f"  Downsampling {name} ({count:,} → 500,000)")
            down = group.sample(n=500_000, random_state=42)
            balanced_dfs.append(down)
        else:
            balanced_dfs.append(group)

    return pd.concat(balanced_dfs, ignore_index=True)


def scale_features(df: pd.DataFrame):
    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    feature_cols = [c for c in config.FEATURE_COLUMNS if c in df.columns]
    scaler_dict = {}

    skip_prescale = getattr(config, 'SKIP_PRESCALE', True)
    
    if skip_prescale:
        print(f"[preprocess] Pre-split scaling SKIPPED (train-only z-score in train.py handles normalization)")
        print(f"[preprocess]   → No data leakage from scaler fitting on full dataset")
        return df, scaler_dict

    if config.USE_ROBUST_SCALER:
        robust_cols = [c for c in config.ROBUST_SCALE_FEATURES if c in feature_cols]
        minmax_cols = [c for c in feature_cols if c not in robust_cols]

        if robust_cols:
            robust_scaler = RobustScaler()
            df[robust_cols] = robust_scaler.fit_transform(df[robust_cols])
            scaler_dict["robust"] = robust_scaler
            scaler_dict["robust_cols"] = robust_cols
            print(f"[preprocess] RobustScaler applied to {len(robust_cols)} features")

        if minmax_cols:
            minmax_scaler = MinMaxScaler()
            df[minmax_cols] = minmax_scaler.fit_transform(df[minmax_cols])
            scaler_dict["minmax"] = minmax_scaler
            scaler_dict["minmax_cols"] = minmax_cols
            print(f"[preprocess] MinMaxScaler applied to {len(minmax_cols)} features")
    else:
        scaler = MinMaxScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        scaler_dict["minmax"] = scaler
        scaler_dict["minmax_cols"] = feature_cols
        print(f"[preprocess] MinMaxScaler applied to all {len(feature_cols)} features")

    return df, scaler_dict


def load_and_preprocess(save_path=None):

    print("\n" + "=" * 60)
    print("  STEP 1: Loading Datasets")
    print("=" * 60)
    datasets = []

    df1 = load_cicids2017()
    if len(df1) > 0:
        datasets.append(("CICIDS2017", df1))

    df2 = load_cse_cic_ids2018()
    if len(df2) > 0:
        datasets.append(("CSE-CIC-IDS2018", df2))

    df3 = load_unsw_nb15()
    if len(df3) > 0:
        datasets.append(("UNSW-NB15", df3))

    df4 = load_cic_ddos2019()
    if len(df4) > 0:
        datasets.append(("CIC-DDoS2019", df4))

    df5 = load_ciciot2023()
    if len(df5) > 0:
        datasets.append(("CICIoT2023", df5))

    df6 = load_botiot(sample_frac=config.BOTIOT_SAMPLE_FRAC)
    if len(df6) > 0:
        datasets.append(("Bot-IoT", df6))

    df7 = load_bell_dns(sample_frac=1.0)
    if len(df7) > 0:
        datasets.append(("Bell-DNS", df7))

    df8 = load_litnet(sample_frac=config.LITNET_SAMPLE_FRAC)
    if len(df8) > 0:
        datasets.append(("LITNET-2020", df8))

    if not datasets:
        raise RuntimeError("No datasets loaded! Check data/ directory.")

    print(f"\n  Datasets loaded: {len(datasets)}")
    for name, df in datasets:
        print(f"    {name}: {len(df):,} rows")

    print("\n" + "=" * 60)
    print("  STEP 2: Merging")
    print("=" * 60)
    full_df = pd.concat([df for _, df in datasets], ignore_index=True)
    print(f"  Merged shape: {full_df.shape}")

    for _, df in datasets:
        del df
    del datasets
    gc.collect()

    print("\n" + "=" * 60)
    print("  STEP 3: Cleaning")
    print("=" * 60)
    full_df = clean_dataset(full_df)

    if len(full_df) == 0:
        raise RuntimeError("Dataset is empty after cleaning!")

    print("\n" + "=" * 60)
    print(f"  STEP 4: Encoding Labels ({config.NUM_CLASSES}-class)")
    print("=" * 60)
    full_df = encode_labels(full_df)

    print("  Class distribution:")
    for name, count in full_df["meta_label"].value_counts().items():
        print(f"    {name}: {count:,}")

    print("\n" + "=" * 60)
    print("  STEP 5: Noise Removal")
    print("=" * 60)
    full_df = remove_noise_isolation_forest(full_df)

    print("\n" + "=" * 60)
    print("  STEP 6: Balancing")
    print("=" * 60)
    full_df = balance_dataset(full_df, target_count=50000)
    print(f"  Balanced shape: {full_df.shape}")

    print("\n" + "=" * 60)
    print("  STEP 7: Scaling")
    print("=" * 60)
    full_df, scaler_dict = scale_features(full_df)

    joblib.dump(scaler_dict, config.SCALER_PATH)
    print(f"  Scalers saved to {config.SCALER_PATH}")

    if save_path:
        print(f"\n  Saving to {save_path}...")
        full_df.to_csv(save_path, index=False)
        print("  Saved.")

    print("\n" + "=" * 60)
    print(f"  PREPROCESSING COMPLETE — {len(full_df):,} rows, {config.NUM_CLASSES} classes")
    print("=" * 60 + "\n")

    return full_df


if __name__ == "__main__":
    df = load_and_preprocess(save_path=config.DATASETS_DIR / "combined_preprocessed.csv")
    print("Final shape:", df.shape)
