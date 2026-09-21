import sys
import time
import hashlib
import logging
import threading
import statistics
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Callable

import numpy as np
import torch

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src import config

try:
    from scapy.all import (
        AsyncSniffer, IP, TCP, UDP, ICMP,
        conf as scapy_conf
    )
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False

logger = logging.getLogger("sentinel.live_capture")

FLOW_TIMEOUT_SEC = 120
HARVEST_INTERVAL_SEC = 2.0
WINDOW_SIZE = config.SEQUENCE_LENGTH
WINDOW_SLIDE = 10
FEATURE_COLUMNS = config.FEATURE_COLUMNS


@dataclass
class FlowKey:
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int

    def __hash__(self):
        return hash((self.src_ip, self.dst_ip, self.src_port, self.dst_port, self.protocol))

    def __eq__(self, other):
        return (self.src_ip == other.src_ip and self.dst_ip == other.dst_ip and
                self.src_port == other.src_port and self.dst_port == other.dst_port and
                self.protocol == other.protocol)

    @property
    def reverse(self):
        return FlowKey(self.dst_ip, self.src_ip, self.dst_port, self.src_port, self.protocol)


@dataclass
class FlowStats:
    flow_key: FlowKey
    start_time: float = 0.0
    last_time: float = 0.0

    fwd_packets: int = 0
    fwd_bytes: int = 0
    fwd_pkt_lengths: list = field(default_factory=list)
    fwd_psh_flags: int = 0
    fwd_urg_flags: int = 0

    bwd_packets: int = 0
    bwd_bytes: int = 0
    bwd_pkt_lengths: list = field(default_factory=list)
    bwd_psh_flags: int = 0
    bwd_urg_flags: int = 0

    fwd_iats: list = field(default_factory=list)
    bwd_iats: list = field(default_factory=list)
    all_iats: list = field(default_factory=list)

    last_fwd_time: float = 0.0
    last_bwd_time: float = 0.0
    last_pkt_time: float = 0.0

    def add_packet(self, pkt_len: int, timestamp: float, is_forward: bool,
                   psh_flag: bool = False, urg_flag: bool = False):
        if self.start_time == 0:
            self.start_time = timestamp
            self.last_fwd_time = timestamp
            self.last_bwd_time = timestamp
            self.last_pkt_time = timestamp
        else:
            iat = timestamp - self.last_pkt_time
            if iat > 0:
                self.all_iats.append(iat)

        self.last_time = timestamp
        self.last_pkt_time = timestamp

        if is_forward:
            self.fwd_packets += 1
            self.fwd_bytes += pkt_len
            self.fwd_pkt_lengths.append(pkt_len)
            if psh_flag:
                self.fwd_psh_flags += 1
            if urg_flag:
                self.fwd_urg_flags += 1
            if self.fwd_packets > 1:
                iat = timestamp - self.last_fwd_time
                if iat > 0:
                    self.fwd_iats.append(iat)
            self.last_fwd_time = timestamp
        else:
            self.bwd_packets += 1
            self.bwd_bytes += pkt_len
            self.bwd_pkt_lengths.append(pkt_len)
            if psh_flag:
                self.bwd_psh_flags += 1
            if urg_flag:
                self.bwd_urg_flags += 1
            if self.bwd_packets > 1:
                iat = timestamp - self.last_bwd_time
                if iat > 0:
                    self.bwd_iats.append(iat)
            self.last_bwd_time = timestamp

    def extract_features(self) -> np.ndarray:
        duration = max(self.last_time - self.start_time, 1e-6)
        duration_us = duration * 1e6

        total_fwd_pkts = self.fwd_packets
        total_bwd_pkts = self.bwd_packets
        total_fwd_len = self.fwd_bytes
        total_bwd_len = self.bwd_bytes

        if self.fwd_pkt_lengths:
            fwd_max = max(self.fwd_pkt_lengths)
            fwd_min = min(self.fwd_pkt_lengths)
            fwd_mean = statistics.mean(self.fwd_pkt_lengths)
            fwd_std = statistics.pstdev(self.fwd_pkt_lengths) if len(self.fwd_pkt_lengths) > 1 else 0.0
        else:
            fwd_max = fwd_min = fwd_mean = fwd_std = 0.0

        if self.bwd_pkt_lengths:
            bwd_max = max(self.bwd_pkt_lengths)
            bwd_min = min(self.bwd_pkt_lengths)
            bwd_mean = statistics.mean(self.bwd_pkt_lengths)
            bwd_std = statistics.pstdev(self.bwd_pkt_lengths) if len(self.bwd_pkt_lengths) > 1 else 0.0
        else:
            bwd_max = bwd_min = bwd_mean = bwd_std = 0.0

        if self.all_iats:
            iat_vals_us = [x * 1e6 for x in self.all_iats]
            iat_mean = statistics.mean(iat_vals_us)
            iat_std = statistics.pstdev(iat_vals_us) if len(iat_vals_us) > 1 else 0.0
            iat_max = max(iat_vals_us)
            iat_min = min(iat_vals_us)
        else:
            iat_mean = iat_std = iat_max = iat_min = 0.0

        flow_bytes_per_s = (total_fwd_len + total_bwd_len) / duration if duration > 0 else 0.0
        flow_pkts_per_s = (total_fwd_pkts + total_bwd_pkts) / duration if duration > 0 else 0.0

        features = np.array([
            self.flow_key.dst_port,
            self.flow_key.protocol,
            duration_us,
            total_fwd_pkts,
            total_bwd_pkts,
            total_fwd_len,
            total_bwd_len,
            fwd_max,
            fwd_min,
            fwd_mean,
            fwd_std,
            bwd_max,
            bwd_min,
            bwd_mean,
            bwd_std,
            iat_mean,
            iat_std,
            iat_max,
            iat_min,
            self.fwd_psh_flags,
            self.bwd_psh_flags,
            self.fwd_urg_flags,
            self.bwd_urg_flags,
            flow_bytes_per_s,
            flow_pkts_per_s,
        ], dtype=np.float32)

        return features


def _compute_temporal_features(window: np.ndarray) -> np.ndarray:
    seq_len = window.shape[0]
    temporal = np.zeros((seq_len, 5), dtype=np.float32)

    BYTES_IDX = FEATURE_COLUMNS.index("Flow Bytes/s")
    PKTS_IDX = FEATURE_COLUMNS.index("Flow Packets/s")
    FWD_PKTS_IDX = FEATURE_COLUMNS.index("Total Fwd Packets")
    BWD_PKTS_IDX = FEATURE_COLUMNS.index("Total Backward Packets")
    DURATION_IDX = FEATURE_COLUMNS.index("Flow Duration")

    temporal[1:, 0] = np.diff(window[:, BYTES_IDX])

    temporal[1:, 1] = np.diff(window[:, PKTS_IDX])

    for i in range(seq_len):
        start = max(0, i - 4)
        temporal[i, 2] = np.mean(window[start:i + 1, FWD_PKTS_IDX])

    for i in range(seq_len):
        start = max(0, i - 4)
        temporal[i, 3] = np.mean(window[start:i + 1, BWD_PKTS_IDX])

    temporal[:, 4] = np.var(window[:, DURATION_IDX])

    return temporal


class LiveCaptureEngine:

    def __init__(self, model: torch.nn.Module, device: torch.device,
                 zscore_mean: Optional[np.ndarray] = None,
                 zscore_std: Optional[np.ndarray] = None,
                 on_alert: Optional[Callable] = None,
                 on_stats: Optional[Callable] = None):
        if not SCAPY_AVAILABLE:
            raise ImportError("Scapy is required for live capture. Install with: pip install scapy")

        self.model = model
        self.device = device
        self.zscore_mean = zscore_mean
        self.zscore_std = zscore_std
        self.on_alert = on_alert
        self.on_stats = on_stats
        self.class_names = config.get_class_names()

        self._flows = {}
        self._flow_lock = threading.Lock()

        self._completed_flows = []
        self._completed_lock = threading.Lock()

        self._running = False
        self._sniffer = None
        self._harvester_thread = None
        self._classifier_thread = None

        self._stats = {
            "packets_captured": 0,
            "flows_created": 0,
            "flows_expired": 0,
            "windows_classified": 0,
            "attacks_detected": 0,
            "capture_start": None,
            "last_packet_time": None,
        }
        self._stats_lock = threading.Lock()

        logger.info("LiveCaptureEngine initialized (scapy=%s)", SCAPY_AVAILABLE)

    def start(self, interface: Optional[str] = None, bpf_filter: str = "ip"):
        if self._running:
            logger.warning("Capture already running")
            return

        self._running = True
        self._stats["capture_start"] = datetime.now().isoformat()

        sniffer_kwargs = {
            "prn": self._packet_callback,
            "store": 0,
            "filter": bpf_filter,
        }
        if interface:
            sniffer_kwargs["iface"] = interface

        self._sniffer = AsyncSniffer(**sniffer_kwargs)
        self._sniffer.start()
        logger.info("Packet capture started on interface=%s, filter='%s'",
                     interface or "default", bpf_filter)

        self._harvester_thread = threading.Thread(
            target=self._harvester_loop, daemon=True, name="flow-harvester"
        )
        self._harvester_thread.start()

        self._classifier_thread = threading.Thread(
            target=self._classifier_loop, daemon=True, name="flow-classifier"
        )
        self._classifier_thread.start()

    def stop(self):
        if not self._running:
            return

        self._running = False

        if self._sniffer:
            try:
                self._sniffer.stop()
            except Exception:
                pass
            self._sniffer = None

        self._harvest_expired_flows(force_all=True)

        logger.info("Capture stopped. Stats: %s", self.get_stats())

    def get_stats(self) -> dict:
        with self._stats_lock:
            return dict(self._stats)

    def get_active_flows(self) -> int:
        with self._flow_lock:
            return len(self._flows)


    def _packet_callback(self, pkt):
        if not pkt.haslayer(IP):
            return

        ip_layer = pkt[IP]
        src_ip = ip_layer.src
        dst_ip = ip_layer.dst
        proto = ip_layer.proto
        pkt_len = len(pkt)
        timestamp = float(pkt.time)

        src_port = 0
        dst_port = 0
        psh_flag = False
        urg_flag = False

        if pkt.haslayer(TCP):
            tcp = pkt[TCP]
            src_port = tcp.sport
            dst_port = tcp.dport
            psh_flag = bool(tcp.flags & 0x08)
            urg_flag = bool(tcp.flags & 0x20)
        elif pkt.haslayer(UDP):
            udp = pkt[UDP]
            src_port = udp.sport
            dst_port = udp.dport

        fwd_key = FlowKey(src_ip, dst_ip, src_port, dst_port, proto)
        rev_key = fwd_key.reverse

        with self._flow_lock:
            if fwd_key in self._flows:
                self._flows[fwd_key].add_packet(pkt_len, timestamp, True, psh_flag, urg_flag)
            elif rev_key in self._flows:
                self._flows[rev_key].add_packet(pkt_len, timestamp, False, psh_flag, urg_flag)
            else:
                flow = FlowStats(flow_key=fwd_key)
                flow.add_packet(pkt_len, timestamp, True, psh_flag, urg_flag)
                self._flows[fwd_key] = flow
                self._stats["flows_created"] += 1

        with self._stats_lock:
            self._stats["packets_captured"] += 1
            self._stats["last_packet_time"] = datetime.now().isoformat()


    def _harvester_loop(self):
        while self._running:
            self._harvest_expired_flows()
            time.sleep(HARVEST_INTERVAL_SEC)

    def _harvest_expired_flows(self, force_all: bool = False):
        now = time.time()
        expired_keys = []

        with self._flow_lock:
            for key, flow in self._flows.items():
                idle_time = now - flow.last_time
                if force_all or idle_time > FLOW_TIMEOUT_SEC:
                    if (flow.fwd_packets + flow.bwd_packets) >= 2:
                        features = flow.extract_features()
                        with self._completed_lock:
                            self._completed_flows.append((features, flow.flow_key.src_ip))
                    expired_keys.append(key)

            for key in expired_keys:
                del self._flows[key]

        if expired_keys:
            with self._stats_lock:
                self._stats["flows_expired"] += len(expired_keys)


    def _classifier_loop(self):
        while self._running:
            self._try_classify()
            time.sleep(0.5)

    def _try_classify(self):
        with self._completed_lock:
            if len(self._completed_flows) < WINDOW_SIZE:
                return

            window_data = self._completed_flows[:WINDOW_SIZE]
            self._completed_flows = self._completed_flows[WINDOW_SLIDE:]

        features_list = [d[0] for d in window_data]
        source_ips = [d[1] for d in window_data]
        base_features = np.stack(features_list)

        temporal = _compute_temporal_features(base_features)

        full_features = np.concatenate([base_features, temporal], axis=1)

        if self.zscore_mean is not None and self.zscore_std is not None:
            mean = self.zscore_mean
            std = np.where(self.zscore_std < 1e-8, 1.0, self.zscore_std)
            full_features = (full_features - mean) / std
            full_features = np.clip(full_features, -10, 10)

        tensor = torch.tensor(full_features, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            output = self.model(tensor)
            temperature = 0.5
            scaled_logits = output / temperature
            probabilities = torch.nn.functional.softmax(scaled_logits, dim=1)
            pred_class = output.argmax(dim=1).item()
            confidence = probabilities[0][pred_class].item() * 100

        from collections import Counter
        ip_counts = Counter(source_ips)
        dominant_ip = ip_counts.most_common(1)[0][0]

        class_name = self.class_names[pred_class] if pred_class < len(self.class_names) else "Unknown"

        with self._stats_lock:
            self._stats["windows_classified"] += 1
            if pred_class != 0:
                self._stats["attacks_detected"] += 1

        alert = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "meta_class": class_name,
            "ip": dominant_ip,
            "confidence": round(confidence, 1),
            "pred_class_id": pred_class,
            "source": "live_capture",
            "flow_count": WINDOW_SIZE,
            "unique_ips": len(set(source_ips)),
        }

        if self.on_alert:
            try:
                self.on_alert(alert)
            except Exception as e:
                logger.error("Alert callback error: %s", e)


    def list_interfaces(self):
        if not SCAPY_AVAILABLE:
            return []
        try:
            from scapy.all import get_if_list
            return get_if_list()
        except Exception:
            return []
