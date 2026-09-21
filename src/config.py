from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

DATA_CICIDS2017_DIR = DATA_DIR / "cicids2017"
DATA_CSE_CIC_IDS2018_DIR = DATA_DIR / "cse-cic-ids2018"
DATA_UNSW_NB15_DIR = DATA_DIR / "unsw-nb15"

MODELS_DIR = RESULTS_DIR / "models"
LOGS_DIR = RESULTS_DIR / "logs"
FIGURES_DIR = RESULTS_DIR / "figures"
DATASETS_DIR = RESULTS_DIR / "datasets"
EXPLAINABILITY_DIR = RESULTS_DIR / "explainability"
EXPERIMENTS_DIR = RESULTS_DIR / "experiments"

for _p in (RESULTS_DIR, MODELS_DIR, LOGS_DIR, FIGURES_DIR, DATASETS_DIR,
           EXPLAINABILITY_DIR, EXPERIMENTS_DIR):
    _p.mkdir(parents=True, exist_ok=True)

FLOW_ID_COL = "flow_id"
TIMESTAMP_COL = "timestamp"
RAW_LABEL_COL = "Label"

FEATURE_COLUMNS = [
    "Destination Port",
    "Protocol",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "Flow Bytes/s",
    "Flow Packets/s",
]

ROBUST_SCALE_FEATURES = [
    "Flow Bytes/s",
    "Flow Packets/s",
    "Fwd Packet Length Max",
    "Bwd Packet Length Max",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
]

TEMPORAL_FEATURES_ENABLED = True

TEMPORAL_FEATURE_NAMES = [
    "delta_bytes_per_s",
    "delta_packets_per_s",
    "rolling_mean_fwd_pkts",
    "rolling_mean_bwd_pkts",
    "var_flow_duration",
]

META_CLASS_NAMES = [
    "Normal",
    "DoS",
    "DDoS",
    "PortScan/Recon",
    "Web/SQLi",
    "Malware/Botnet/Exploit",
]

META_CLASS_NAMES_FINE = [
    "Normal",
    "DoS-HTTP",
    "DoS-TCP",
    "DDoS-UDP",
    "DDoS-HTTP",
    "PortScan/Recon",
    "BruteForce-FTP",
    "BruteForce-SSH",
    "WebAttack-SQLi",
    "WebAttack-XSS",
    "Botnet/Malware",
    "IoT-Mirai",
]

NUM_CLASSES = 8

def get_class_names():
    if NUM_CLASSES in (8, 10):
        return META_CLASS_NAMES_10
    if NUM_CLASSES == 12:
        return META_CLASS_NAMES_FINE
    return META_CLASS_NAMES

RAW_TO_META_LABEL = {
    "BENIGN": "Normal",
    "Normal": "Normal",

    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "Heartbleed": "DoS",
    "DoS": "DoS",
    "DoS attacks-Hulk": "DoS",
    "DoS attacks-GoldenEye": "DoS",
    "DoS attacks-Slowloris": "DoS",
    "DoS attacks-SlowHTTPTest": "DoS",
    "DoS-TCP_Flood": "DoS",
    "DoS-UDP_Flood": "DoS",
    "DoS-SYN_Flood": "DoS",
    "DoS-HTTP_Flood": "DoS",

    "DDoS": "DDoS",
    "DDOS attack-HOIC": "DDoS",
    "DDOS attack-LOIC-UDP": "DDoS",
    "DrDoS_DNS": "DDoS",
    "DrDoS_LDAP": "DDoS",
    "DrDoS_MSSQL": "DDoS",
    "DrDoS_NetBIOS": "DDoS",
    "DrDoS_NTP": "DDoS",
    "DrDoS_SNMP": "DDoS",
    "DrDoS_SSDP": "DDoS",
    "DrDoS_UDP": "DDoS",
    "Syn": "DDoS",
    "UDP-lag": "DDoS",
    "UDPLag": "DDoS",
    "WebDDoS": "DDoS",
    "TFTP": "DDoS",
    "LDAP": "DDoS",
    "MSSQL": "DDoS",
    "NetBIOS": "DDoS",
    "Portmap": "DDoS",
    "UDP": "DDoS",
    "DDoS-ICMP_Flood": "DDoS",
    "DDoS-UDP_Flood": "DDoS",
    "DDoS-TCP_Flood": "DDoS",
    "DDoS-PSHACK_Flood": "DDoS",
    "DDoS-SYN_Flood": "DDoS",
    "DDoS-RSTFINFlood": "DDoS",
    "DDoS-SynonymousIP_Flood": "DDoS",
    "DDoS-ICMP_Fragmentation": "DDoS",
    "DDoS-ACK_Fragmentation": "DDoS",
    "DDoS-UDP_Fragmentation": "DDoS",
    "DDoS-HTTP_Flood": "DDoS",
    "DDoS-SlowLoris": "DDoS",

    "PortScan": "PortScan/Recon",
    "Port Scan": "PortScan/Recon",
    "Reconnaissance": "PortScan/Recon",
    "Analysis": "PortScan/Recon",
    "Fuzzers": "PortScan/Recon",
    "Vulnerability_scanner": "PortScan/Recon",
    "OS_Fingerprinting": "PortScan/Recon",
    "Ping_Sweep": "PortScan/Recon",
    "Port_Scanning": "PortScan/Recon",
    "Recon-HostDiscovery": "PortScan/Recon",
    "Recon-OSScan": "PortScan/Recon",
    "Recon-PortScan": "PortScan/Recon",
    "Recon-PingSweep": "PortScan/Recon",
    "VulnerabilityScan": "PortScan/Recon",

    "Web Attack  Brute Force": "Web/SQLi",
    "Web Attack  XSS": "Web/SQLi",
    "Web Attack  Sql Injection": "Web/SQLi",
    "Web Attack - Brute Force": "Web/SQLi",
    "Web Attack - XSS": "Web/SQLi",
    "Web Attack - Sql Injection": "Web/SQLi",
    "Brute Force -Web": "Web/SQLi",
    "Brute Force -XSS": "Web/SQLi",
    "SQL Injection": "Web/SQLi",
    "XSS": "Web/SQLi",
    "SqlInjection": "Web/SQLi",

    "Bot": "Malware/Botnet/Exploit",
    "Infiltration": "Malware/Botnet/Exploit",
    "FTP-Patator": "Malware/Botnet/Exploit",
    "SSH-Patator": "Malware/Botnet/Exploit",
    "Infilteration": "Malware/Botnet/Exploit",
    "FTP-BruteForce": "Malware/Botnet/Exploit",
    "SSH-Bruteforce": "Malware/Botnet/Exploit",
    "Exploits": "Malware/Botnet/Exploit",
    "Backdoor": "Malware/Botnet/Exploit",
    "Backdoors": "Malware/Botnet/Exploit",
    "Shellcode": "Malware/Botnet/Exploit",
    "Worms": "Malware/Botnet/Exploit",
    "Generic": "Malware/Botnet/Exploit",
    "Mirai": "Malware/Botnet/Exploit",
    "Mirai-greeth_flood": "Malware/Botnet/Exploit",
    "Mirai-greip_flood": "Malware/Botnet/Exploit",
    "Mirai-udpplain": "Malware/Botnet/Exploit",
    "BrowserHijacking": "Malware/Botnet/Exploit",
    "CommandInjection": "Web/SQLi",
    "Backdoor_Malware": "Malware/Botnet/Exploit",
    "Uploading_Attack": "Web/SQLi",
    "DictionaryBruteForce": "Malware/Botnet/Exploit",
    "MITM-ArpSpoofing": "Malware/Botnet/Exploit",
    "DNS_Spoofing": "Malware/Botnet/Exploit",
    "Spoofing": "Malware/Botnet/Exploit",
}

RAW_TO_META_LABEL_FINE = {
    "BENIGN": "Normal",
    "Normal": "Normal",
    "DoS Hulk": "DoS-HTTP",
    "DoS GoldenEye": "DoS-HTTP",
    "DoS slowloris": "DoS-HTTP",
    "DoS Slowhttptest": "DoS-HTTP",
    "DoS attacks-Hulk": "DoS-HTTP",
    "DoS attacks-GoldenEye": "DoS-HTTP",
    "DoS attacks-Slowloris": "DoS-HTTP",
    "DoS attacks-SlowHTTPTest": "DoS-HTTP",
    "DoS-HTTP_Flood": "DoS-HTTP",
    "DoS-TCP_Flood": "DoS-TCP",
    "DoS-UDP_Flood": "DoS-TCP",
    "DoS-SYN_Flood": "DoS-TCP",
    "DoS": "DoS-TCP",
    "DDOS attack-LOIC-UDP": "DDoS-UDP",
    "DrDoS_DNS": "DDoS-UDP",
    "DrDoS_LDAP": "DDoS-UDP",
    "DrDoS_MSSQL": "DDoS-UDP",
    "DrDoS_NetBIOS": "DDoS-UDP",
    "DrDoS_NTP": "DDoS-UDP",
    "DrDoS_SNMP": "DDoS-UDP",
    "DrDoS_SSDP": "DDoS-UDP",
    "DrDoS_UDP": "DDoS-UDP",
    "UDP-lag": "DDoS-UDP",
    "UDPLag": "DDoS-UDP",
    "DDoS-UDP_Flood": "DDoS-UDP",
    "DDoS-UDP_Fragmentation": "DDoS-UDP",
    "DDoS-ICMP_Flood": "DDoS-UDP",
    "DDoS-ICMP_Fragmentation": "DDoS-UDP",
    "DDoS": "DDoS-HTTP",
    "DDOS attack-HOIC": "DDoS-HTTP",
    "WebDDoS": "DDoS-HTTP",
    "DDoS-HTTP_Flood": "DDoS-HTTP",
    "DDoS-SlowLoris": "DDoS-HTTP",
    "DDoS-TCP_Flood": "DDoS-HTTP",
    "DDoS-PSHACK_Flood": "DDoS-HTTP",
    "DDoS-SYN_Flood": "DDoS-HTTP",
    "DDoS-RSTFINFlood": "DDoS-HTTP",
    "DDoS-SynonymousIP_Flood": "DDoS-HTTP",
    "DDoS-ACK_Fragmentation": "DDoS-HTTP",
    "Syn": "DDoS-HTTP",
    "TFTP": "DDoS-HTTP",
    "Heartbleed": "DDoS-HTTP",
    "PortScan": "PortScan/Recon",
    "Port Scan": "PortScan/Recon",
    "Reconnaissance": "PortScan/Recon",
    "Analysis": "PortScan/Recon",
    "Fuzzers": "PortScan/Recon",
    "Vulnerability_scanner": "PortScan/Recon",
    "OS_Fingerprinting": "PortScan/Recon",
    "Ping_Sweep": "PortScan/Recon",
    "Port_Scanning": "PortScan/Recon",
    "Recon-HostDiscovery": "PortScan/Recon",
    "Recon-OSScan": "PortScan/Recon",
    "Recon-PortScan": "PortScan/Recon",
    "FTP-Patator": "BruteForce-FTP",
    "FTP-BruteForce": "BruteForce-FTP",
    "SSH-Patator": "BruteForce-SSH",
    "SSH-Bruteforce": "BruteForce-SSH",
    "DictionaryBruteForce": "BruteForce-FTP",
    "Web Attack  Sql Injection": "WebAttack-SQLi",
    "Web Attack - Sql Injection": "WebAttack-SQLi",
    "SQL Injection": "WebAttack-SQLi",
    "SqlInjection": "WebAttack-SQLi",
    "Web Attack  XSS": "WebAttack-XSS",
    "Web Attack - XSS": "WebAttack-XSS",
    "Brute Force -XSS": "WebAttack-XSS",
    "XSS": "WebAttack-XSS",
    "Web Attack  Brute Force": "WebAttack-SQLi",
    "Web Attack - Brute Force": "WebAttack-SQLi",
    "Brute Force -Web": "WebAttack-SQLi",
    "Bot": "Botnet/Malware",
    "Infiltration": "Botnet/Malware",
    "Infilteration": "Botnet/Malware",
    "Exploits": "Botnet/Malware",
    "Backdoor": "Botnet/Malware",
    "Backdoors": "Botnet/Malware",
    "Shellcode": "Botnet/Malware",
    "Worms": "Botnet/Malware",
    "Generic": "Botnet/Malware",
    "Backdoor_Malware": "Botnet/Malware",
    "BrowserHijacking": "Botnet/Malware",
    "CommandInjection": "Botnet/Malware",
    "Uploading_Attack": "Botnet/Malware",
    "MITM-ArpSpoofing": "Botnet/Malware",
    "DNS_Spoofing": "Botnet/Malware",
    "Spoofing": "Botnet/Malware",
    "Mirai": "IoT-Mirai",
}

def get_label_map():
    if NUM_CLASSES in (8, 10):
        return RAW_TO_META_LABEL_10
    if NUM_CLASSES == 12:
        return RAW_TO_META_LABEL_FINE
    return RAW_TO_META_LABEL

def meta_label_to_id(meta_label: str) -> int:
    names = get_class_names()
    if meta_label in names:
        return names.index(meta_label)
    return 0

def id_to_meta_label(idx: int) -> str:
    names = get_class_names()
    if 0 <= idx < len(names):
        return names[idx]
    return "Normal"

SEQUENCE_LENGTH = 20
SEQUENCE_STEP = 10

TEST_SIZE = 0.2
VAL_SIZE = 0.1
RANDOM_STATE = 42
WINDOW_BLOCK_SIZE = 2

BATCH_SIZE = 2048
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 10

USE_FOCAL_LOSS = True
FOCAL_LOSS_GAMMA = 2.0
FOCAL_LOSS_ALPHA = None
LABEL_SMOOTHING = 0.1

USE_BORDERLINE_SMOTE = False
USE_ROBUST_SCALER = True
SKIP_PRESCALE = True
USE_ISOLATION_FOREST = True
ISOLATION_FOREST_CONTAMINATION = 0.01
CSE_CIC_2018_SAMPLE_FRAC = 0.50
BOTIOT_SAMPLE_FRAC = 0.10
LITNET_SAMPLE_FRAC = 0.10

MODEL_VARIANT = "model_v2"
USE_MULTISCALE_CNN = False
USE_RESIDUAL_CNN = True

ADVERSARIAL_TRAINING = False
ADVERSARIAL_NOISE_STD = 0.05
ADVERSARIAL_SAMPLE_RATIO = 0.10

MODEL_CHECKPOINT_PATH = MODELS_DIR / "best_model_v2_swa.pth"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
SEQUENCES_PATH = DATASETS_DIR / "combined_sequences.pt"
SPLIT_INDICES_PATH = DATASETS_DIR / "combined_splits.pt"

FPGA_DIR = RESULTS_DIR / "fpga"
FPGA_DIR.mkdir(parents=True, exist_ok=True)

ONNX_MODEL_PATH = FPGA_DIR / "cnn_lstm.onnx"
ONNX_FPGA_MODEL_PATH = FPGA_DIR / "cnn_fpga.onnx"
QUANTIZED_MODEL_PATH = FPGA_DIR / "cnn_lstm_int8.pth"
FPGA_MODEL_PATH = FPGA_DIR / "cnn_fpga.pth"
FPGA_BENCHMARK_PATH = FPGA_DIR / "benchmark_report.txt"

QUANTIZATION_BITS = 8
CALIBRATION_SAMPLES = 500

CAPTURE_INTERFACE = None
FLOW_TIMEOUT_SEC = 120
FLOW_IDLE_SEC = 30
CAPTURE_DURATION_SEC = 60
WINDOW_TRIGGER_INTERVAL_SEC = 10

SEVERITY_BLOCK_THRESHOLD = 0.95
SEVERITY_RATE_LIMIT_THRESHOLD = 0.75
SEVERITY_LOG_THRESHOLD = 0.50

BLOCK_TIMEOUT_MINUTES = 10
IP_WHITELIST = []

DRIFT_KL_THRESHOLD = 0.15
DRIFT_PSI_THRESHOLD = 0.20
DRIFT_CHECK_INTERVAL_SEC = 300

FPGA_CLOCK_MHZ = 100
FPGA_DSP_BLOCKS = 220
FPGA_BRAM_KB = 560
FPGA_LUT_COUNT = 53200

DASHBOARD_PORT = 8502
DASHBOARD_REFRESH_SEC = 2


DATA_BOT_IOT_DIR = DATA_DIR / "bot_iot"
DATA_BELL_DNS_DIR = DATA_DIR / "bell_dns"
DATA_LITNET_DIR = DATA_DIR / "litnet"

CICIOT2023_SAMPLE_FRAC = 0.1
CIC_DDOS2019_SAMPLE_FRAC = 1.0
BOTIOT_SAMPLE_FRAC = 0.05
LITNET_SAMPLE_FRAC = 0.05

META_CLASS_NAMES_10 = [
    "Normal",
    "DoS/DDoS",
    "PortScan/Recon",
    "Web/Injection",
    "Brute Force",
    "Botnet/C2",
    "Malware/Exploit",
    "Infiltration",
]

RAW_TO_META_LABEL_10 = {
    "BENIGN": "Normal",
    "Normal": "Normal",
    "Benign": "Normal",
    "DoS Hulk": "DoS/DDoS",
    "DoS GoldenEye": "DoS/DDoS",
    "DoS slowloris": "DoS/DDoS",
    "DoS Slowhttptest": "DoS/DDoS",
    "DoS attacks-Hulk": "DoS/DDoS",
    "DoS attacks-GoldenEye": "DoS/DDoS",
    "DoS attacks-Slowloris": "DoS/DDoS",
    "DoS attacks-SlowHTTPTest": "DoS/DDoS",
    "DoS-TCP_Flood": "DoS/DDoS",
    "DoS-UDP_Flood": "DoS/DDoS",
    "DoS-SYN_Flood": "DoS/DDoS",
    "DoS-HTTP_Flood": "DoS/DDoS",
    "Heartbleed": "DoS/DDoS",
    "DoS": "DoS/DDoS",
    "DDoS": "DoS/DDoS",
    "DDOS attack-HOIC": "DoS/DDoS",
    "DDOS attack-LOIC-UDP": "DoS/DDoS",
    "DDoS attacks-LOIC-HTTP": "DoS/DDoS",
    "DDoS attacks-LOIC-UDP": "DoS/DDoS",
    "DrDoS_DNS": "DoS/DDoS", "DrDoS_LDAP": "DoS/DDoS",
    "DrDoS_MSSQL": "DoS/DDoS", "DrDoS_NetBIOS": "DoS/DDoS",
    "DrDoS_NTP": "DoS/DDoS", "DrDoS_SNMP": "DoS/DDoS",
    "DrDoS_SSDP": "DoS/DDoS", "DrDoS_UDP": "DoS/DDoS",
    "Syn": "DoS/DDoS", "UDP-lag": "DoS/DDoS", "UDPLag": "DoS/DDoS",
    "WebDDoS": "DoS/DDoS", "TFTP": "DoS/DDoS",
    "DDoS-ICMP_Flood": "DoS/DDoS", "DDoS-UDP_Flood": "DoS/DDoS",
    "DDoS-TCP_Flood": "DoS/DDoS", "DDoS-PSHACK_Flood": "DoS/DDoS",
    "DDoS-SYN_Flood": "DoS/DDoS", "DDoS-RSTFINFlood": "DoS/DDoS",
    "DDoS-SynonymousIP_Flood": "DoS/DDoS",
    "DDoS-ICMP_Fragmentation": "DoS/DDoS",
    "DDoS-ACK_Fragmentation": "DoS/DDoS",
    "DDoS-UDP_Fragmentation": "DoS/DDoS",
    "DDoS-HTTP_Flood": "DoS/DDoS", "DDoS-SlowLoris": "DoS/DDoS",
    "NetBIOS": "DoS/DDoS", "LDAP": "DoS/DDoS",
    "MSSQL": "DoS/DDoS", "Portmap": "DoS/DDoS",
    "UDP": "DoS/DDoS", "SNMP": "DoS/DDoS",
    "SSDP": "DoS/DDoS", "DNS": "DoS/DDoS", "NTP": "DoS/DDoS",
    "PortScan": "PortScan/Recon",
    "Port Scan": "PortScan/Recon",
    "Reconnaissance": "PortScan/Recon",
    "Analysis": "PortScan/Recon",
    "Fuzzers": "PortScan/Recon",
    "Recon-HostDiscovery": "PortScan/Recon",
    "Recon-OSScan": "PortScan/Recon",
    "Recon-PortScan": "PortScan/Recon",
    "Recon-PingSweep": "PortScan/Recon",
    "VulnerabilityScan": "PortScan/Recon",
    "PortScan/Recon": "PortScan/Recon",
    "Web Attack  Brute Force": "Web/Injection",
    "Web Attack  XSS": "Web/Injection",
    "Web Attack  Sql Injection": "Web/Injection",
    "Brute Force -Web": "Web/Injection",
    "Brute Force -XSS": "Web/Injection",
    "SQL Injection": "Web/Injection",
    "XSS": "Web/Injection",
    "SqlInjection": "Web/Injection",
    "FTP-Patator": "Brute Force",
    "SSH-Patator": "Brute Force",
    "FTP-BruteForce": "Brute Force",
    "SSH-Bruteforce": "Brute Force",
    "DictionaryBruteForce": "Brute Force",
    "Bot": "Botnet/C2",
    "Mirai": "Botnet/C2",
    "Mirai-greeth_flood": "Botnet/C2",
    "Mirai-greip_flood": "Botnet/C2",
    "Mirai-udpplain": "Botnet/C2",
    "BrowserHijacking": "Botnet/C2",
    "DNS_Spoofing": "Botnet/C2",
    "Spoofing": "Botnet/C2",
    "MITM-ArpSpoofing": "Botnet/C2",
    "Exploits": "Malware/Exploit",
    "Backdoor": "Malware/Exploit",
    "Backdoors": "Malware/Exploit",
    "Shellcode": "Malware/Exploit",
    "Worms": "Malware/Exploit",
    "Generic": "Malware/Exploit",
    "Backdoor_Malware": "Malware/Exploit",
    "CommandInjection": "Web/Injection",
    "Uploading_Attack": "Web/Injection",
    "Malware/Botnet/Exploit": "Malware/Exploit",
    "Infiltration": "Infiltration",
    "Infilteration": "Infiltration",
}

V2_NUM_FEATURES = 30
V2_SEQUENCE_LENGTH = 20
V2_BASE_FILTERS = 64
V2_LSTM_HIDDEN = 128
V2_GRU_HIDDEN = 128
V2_NUM_HEADS = 8
V2_DROPOUT = 0.2
V2_NUM_CLASSES = NUM_CLASSES

V2_CNN_BASE_FILTERS = V2_BASE_FILTERS
V2_NUM_ATTENTION_HEADS = V2_NUM_HEADS
V2_DROPOUT_CNN = V2_DROPOUT
V2_DROPOUT_RNN = V2_DROPOUT
V2_DROPOUT_HEAD = V2_DROPOUT + 0.1

V2_NUM_EPOCHS = 30
V2_LEARNING_RATE = 1e-3
V2_WEIGHT_DECAY = 1e-4
V2_BATCH_SIZE = 2048
V2_EARLY_STOPPING_PATIENCE = 10
V2_LABEL_SMOOTHING = 0.05
V2_GRAD_CLIP_NORM = 0.5
V2_SWA_START_EPOCH = 15
V2_SWA_LR = 5e-4
V2_WARMUP_EPOCHS = 5
V2_COSINE_T0 = 10
V2_COSINE_TMULT = 2
V2_NUM_WORKERS = 2
V2_PIN_MEMORY = True

V2_FIGURES_DIR = RESULTS_DIR / "figures_v2"
V2_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

V2_MODEL_SAVE_PATH = MODELS_DIR / "best_model_v2.pth"
V2_SWA_MODEL_PATH = MODELS_DIR / "best_model_v2_swa.pth"
V2_SEQUENCES_PATH = DATASETS_DIR / "combined_sequences_v2.pt"

MODEL_V2_CHECKPOINT_PATH = V2_MODEL_SAVE_PATH
MODEL_V2_SWA_PATH = V2_SWA_MODEL_PATH
