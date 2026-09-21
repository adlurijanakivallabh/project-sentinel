import sys
import json
import hashlib
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
from collections import defaultdict

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src import config


logger = logging.getLogger("IPS.TLS")


KNOWN_MALWARE_JA3 = {
    "4d7a28d6f2263ed61de88ca66eb011e3": {"family": "Emotet", "severity": "CRITICAL"},
    "51c64c77e60f3980eea90869b68c58a8": {"family": "Emotet", "severity": "CRITICAL"},
    "07e3d944d77311009e6b0261788b0675": {"family": "Emotet", "severity": "CRITICAL"},
    "1b3890e040b20e5b4f60e3e7e547eb11": {"family": "Emotet", "severity": "CRITICAL"},

    "6734f37431670b3ab4292b8f60f29984": {"family": "TrickBot", "severity": "CRITICAL"},
    "72a589da586844d7f0818ce684948eea": {"family": "TrickBot", "severity": "CRITICAL"},

    "51c64c77e60f3980eea90869b68c58a8": {"family": "Dridex", "severity": "CRITICAL"},
    "8e625cf3e07cd498437be68de6cb1b4b": {"family": "Dridex", "severity": "CRITICAL"},

    "72a589da586844d7f0818ce684948eea": {"family": "CobaltStrike", "severity": "CRITICAL"},
    "a0e9f5d64349fb13191bc781f81f42e1": {"family": "CobaltStrike", "severity": "CRITICAL"},
    "b742b407517bac9536a77a7b0fee28e9": {"family": "CobaltStrike", "severity": "CRITICAL"},

    "5d65ea3fb1d4aa7d826733d2f2cbbb1d": {"family": "Metasploit", "severity": "HIGH"},
    "dee9ddcd0c3e7c06a01a34a0efa62b98": {"family": "Metasploit", "severity": "HIGH"},

    "c12f54a3f91dc7bafd92cb59fe8096a4": {"family": "Qakbot", "severity": "CRITICAL"},
    "6e2df4e4422a71af24dac5e07fb9d22c": {"family": "Qakbot", "severity": "CRITICAL"},

    "fc54e0d16d9764783542f0146a98b300": {"family": "AsyncRAT", "severity": "HIGH"},

    "1138de370e523e824bbca3fe02e44eb5": {"family": "IcedID", "severity": "CRITICAL"},

    "e7d705a3286e19ea42f587b344ee6865": {"family": "GenericMalware", "severity": "HIGH"},
}


def compute_ja3_hash(tls_version: int, cipher_suites: list[int],
                     extensions: list[int], elliptic_curves: list[int],
                     ec_point_formats: list[int]) -> str:
    grease_values = {0x0a0a, 0x1a1a, 0x2a2a, 0x3a3a, 0x4a4a, 0x5a5a,
                     0x6a6a, 0x7a7a, 0x8a8a, 0x9a9a, 0xaaaa, 0xbaba,
                     0xcaca, 0xdada, 0xeaea, 0xfafa}

    ciphers = [c for c in cipher_suites if c not in grease_values]
    exts = [e for e in extensions if e not in grease_values]
    curves = [c for c in elliptic_curves if c not in grease_values]
    formats = [f for f in ec_point_formats if f not in grease_values]

    ja3_str = (
        f"{tls_version},"
        f"{'-'.join(str(c) for c in ciphers)},"
        f"{'-'.join(str(e) for e in exts)},"
        f"{'-'.join(str(c) for c in curves)},"
        f"{'-'.join(str(f) for f in formats)}"
    )

    return hashlib.md5(ja3_str.encode()).hexdigest()


def extract_ja3_from_packet(packet) -> Optional[str]:
    try:
        from scapy.layers.tls.handshake import TLSClientHello
        from scapy.layers.tls.extensions import (
            TLS_Ext_SupportedGroups,
            TLS_Ext_SupportedPointFormat,
        )
    except ImportError:
        logger.warning("Scapy TLS layer not available. Install scapy with: pip install scapy")
        return None

    if not packet.haslayer(TLSClientHello):
        return None

    ch = packet[TLSClientHello]

    tls_version = ch.version if hasattr(ch, 'version') else 0
    cipher_suites = list(ch.ciphers) if hasattr(ch, 'ciphers') else []

    extensions = []
    elliptic_curves = []
    ec_point_formats = []

    if hasattr(ch, 'ext') and ch.ext:
        for ext in ch.ext:
            ext_type = ext.type if hasattr(ext, 'type') else 0
            extensions.append(ext_type)

            if isinstance(ext, TLS_Ext_SupportedGroups):
                elliptic_curves = list(ext.groups) if hasattr(ext, 'groups') else []
            elif isinstance(ext, TLS_Ext_SupportedPointFormat):
                ec_point_formats = list(ext.ecpl) if hasattr(ext, 'ecpl') else []

    return compute_ja3_hash(tls_version, cipher_suites, extensions,
                            elliptic_curves, ec_point_formats)


class TLSFingerprintEngine:

    def __init__(self, blacklist_path: Optional[str] = None):
        self.blacklist = dict(KNOWN_MALWARE_JA3)
        self.blacklist_path = (Path(blacklist_path) if blacklist_path
                               else config.RESULTS_DIR / "ja3_blacklist.json")

        self.stats = {
            "total_tls_packets": 0,
            "total_ja3_extracted": 0,
            "total_blacklist_hits": 0,
            "hits_by_family": defaultdict(int),
            "unique_ja3_seen": set(),
            "last_hit_time": None,
        }

        self._recent_alerts: dict[str, datetime] = {}
        self._alert_cooldown = timedelta(minutes=5)

        logger.info("TLS Fingerprint Engine initialized (%d blacklist entries)",
                     len(self.blacklist))

    def load_blacklist(self, path: Optional[str] = None) -> int:
        load_path = Path(path) if path else self.blacklist_path

        if not load_path.exists():
            logger.info("No blacklist file found at %s, using built-in (%d entries)",
                        load_path, len(self.blacklist))
            return len(self.blacklist)

        try:
            with open(load_path) as f:
                loaded = json.load(f)

            self.blacklist.update(loaded)
            logger.info("Loaded %d JA3 blacklist entries from %s (total: %d)",
                        len(loaded), load_path, len(self.blacklist))
            return len(self.blacklist)

        except Exception as e:
            logger.error("Failed to load blacklist from %s: %s", load_path, e)
            return len(self.blacklist)

    def save_blacklist(self, path: Optional[str] = None):
        save_path = Path(path) if path else self.blacklist_path
        with open(save_path, "w") as f:
            json.dump(self.blacklist, f, indent=2)
        logger.info("Saved %d blacklist entries to %s", len(self.blacklist), save_path)

    def check_ja3(self, ja3_hash: str, source_ip: str = "unknown") -> dict:
        self.stats["total_ja3_extracted"] += 1
        self.stats["unique_ja3_seen"].add(ja3_hash)

        result = {
            "is_malicious": False,
            "malware_family": None,
            "severity": None,
            "ja3_hash": ja3_hash,
            "source_ip": source_ip,
        }

        if ja3_hash in self.blacklist:
            entry = self.blacklist[ja3_hash]
            result["is_malicious"] = True
            result["malware_family"] = entry.get("family", "Unknown")
            result["severity"] = entry.get("severity", "HIGH")

            self.stats["total_blacklist_hits"] += 1
            self.stats["hits_by_family"][result["malware_family"]] += 1
            self.stats["last_hit_time"] = datetime.now().isoformat()

            alert_key = f"{ja3_hash}:{source_ip}"
            now = datetime.now()
            if alert_key in self._recent_alerts:
                if now - self._recent_alerts[alert_key] < self._alert_cooldown:
                    result["rate_limited"] = True
                    return result

            self._recent_alerts[alert_key] = now

            logger.critical(
                "MALWARE JA3 DETECTED: %s from %s — Family: %s, Severity: %s",
                ja3_hash, source_ip, result["malware_family"], result["severity"]
            )

        return result

    def process_packet(self, packet, source_ip: str = "unknown") -> Optional[dict]:
        self.stats["total_tls_packets"] += 1

        ja3_hash = extract_ja3_from_packet(packet)
        if ja3_hash is None:
            return None

        return self.check_ja3(ja3_hash, source_ip)

    def get_stats(self) -> dict:
        return {
            "total_tls_packets": self.stats["total_tls_packets"],
            "total_ja3_extracted": self.stats["total_ja3_extracted"],
            "total_blacklist_hits": self.stats["total_blacklist_hits"],
            "hits_by_family": dict(self.stats["hits_by_family"]),
            "unique_ja3_count": len(self.stats["unique_ja3_seen"]),
            "blacklist_size": len(self.blacklist),
            "last_hit_time": self.stats["last_hit_time"],
        }

    def add_to_blacklist(self, ja3_hash: str, family: str, severity: str = "HIGH"):
        self.blacklist[ja3_hash] = {"family": family, "severity": severity}
        logger.info("Added JA3 %s to blacklist (family=%s)", ja3_hash, family)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TLS Fingerprint Engine CLI")
    parser.add_argument("--check", type=str, help="Check a JA3 hash against blacklist")
    parser.add_argument("--load", type=str, help="Load blacklist from JSON file")
    parser.add_argument("--stats", action="store_true", help="Print engine statistics")
    parser.add_argument("--list", action="store_true", help="List all blacklisted JA3 hashes")
    args = parser.parse_args()

    engine = TLSFingerprintEngine()
    engine.load_blacklist()

    if args.check:
        result = engine.check_ja3(args.check)
        print(json.dumps(result, indent=2))
    elif args.list:
        print(f"\nBlacklisted JA3 hashes ({len(engine.blacklist)}):")
        print(f"{'Hash':<35} {'Family':<20} {'Severity'}")
        print("-" * 70)
        for h, info in sorted(engine.blacklist.items(), key=lambda x: x[1]["family"]):
            print(f"{h:<35} {info['family']:<20} {info['severity']}")
    elif args.stats:
        print(json.dumps(engine.get_stats(), indent=2))
    else:
        print("\n  Testing known malware JA3 hashes:")
        print(f"  {'Hash':<35} {'Result':<12} {'Family'}")
        print(f"  {'-' * 60}")

        test_hashes = [
            "4d7a28d6f2263ed61de88ca66eb011e3",
            "6734f37431670b3ab4292b8f60f29984",
            "a0e9f5d64349fb13191bc781f81f42e1",
            "0000000000000000000000000000000",
        ]

        for h in test_hashes:
            result = engine.check_ja3(h)
            status = "MALICIOUS" if result["is_malicious"] else "CLEAN"
            family = result["malware_family"] or "—"
            print(f"  {h:<35} {status:<12} {family}")

        print(f"\n  Stats: {json.dumps(engine.get_stats(), indent=2)}")
