import os
import sys
import time
import json
import logging
import platform
import subprocess
import threading
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Optional

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src import config


RESPONSE_POLICIES = {
    "Normal": {
        "action": "allow",
        "description": "Benign traffic — no action",
    },
    "DoS": {
        "action": "block_ip",
        "block_duration_minutes": 30,
        "threshold_windows": 3,
        "description": "Block source IP for 30 minutes after 3 DoS windows",
    },
    "DDoS": {
        "action": "block_ip",
        "block_duration_minutes": 60,
        "threshold_windows": 2,
        "description": "Block source IP for 60 minutes after 2 DDoS windows",
    },
    "PortScan/Recon": {
        "action": "block_ip",
        "block_duration_minutes": 15,
        "threshold_windows": 5,
        "description": "Block source IP for 15 minutes after 5 scan attempts",
    },
    "Web/SQLi": {
        "action": "block_ip",
        "block_duration_minutes": 1440,
        "threshold_windows": 1,
        "description": "Immediately block — SQL injection is high severity",
    },
    "Web/Injection": {
        "action": "block_ip",
        "block_duration_minutes": 1440,
        "threshold_windows": 1,
        "description": "Immediately block — web injection is high severity",
    },
    "Brute Force": {
        "action": "block_ip",
        "block_duration_minutes": 60,
        "threshold_windows": 3,
        "description": "Block source IP for 60 minutes after 3 brute force attempts",
    },
    "Botnet/C2": {
        "action": "block_ip",
        "block_duration_minutes": 1440,
        "threshold_windows": 1,
        "description": "Immediately block — botnet C2 traffic is critical",
    },
    "Malware/Exploit": {
        "action": "block_ip",
        "block_duration_minutes": 1440,
        "threshold_windows": 1,
        "description": "Immediately block — malware/exploit is critical",
    },
    "Malware/Botnet/Exploit": {
        "action": "block_ip",
        "block_duration_minutes": 1440,
        "threshold_windows": 1,
        "description": "Immediately block — malware/exploit is critical (V1 compat)",
    },
    "Infiltration": {
        "action": "block_ip",
        "block_duration_minutes": 120,
        "threshold_windows": 2,
        "description": "Block source IP for 2 hours after 2 infiltration detections",
    },
}

SEVERITY_MAP = {
    "Normal": "INFO",
    "DoS": "HIGH",
    "DDoS": "HIGH",
    "PortScan/Recon": "MEDIUM",
    "Web/SQLi": "CRITICAL",
    "Web/Injection": "CRITICAL",
    "Brute Force": "HIGH",
    "Botnet/C2": "CRITICAL",
    "Malware/Exploit": "CRITICAL",
    "Malware/Botnet/Exploit": "CRITICAL",
    "Infiltration": "HIGH",
}


@dataclass
class PreventionAction:
    timestamp: str
    source_ip: str
    attack_type: str
    severity: str
    action_taken: str
    rule_name: str
    block_until: Optional[str] = None
    confidence: float = 0.0
    window_id: int = 0
    details: str = ""


@dataclass
class IPState:
    ip: str
    detection_count: dict = field(default_factory=lambda: defaultdict(int))
    blocked: bool = False
    blocked_until: Optional[datetime] = None
    block_rule_name: Optional[str] = None
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    total_windows: int = 0
    total_attacks: int = 0


class PreventionEngine:

    _IPSET_NAME = "sentinel_blocked_ips"

    def __init__(self, log_dir=None, dry_run=True):
        self.dry_run = dry_run
        self.log_dir = Path(log_dir) if log_dir else config.RESULTS_DIR / "ips_logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._platform = platform.system()

        self.ip_states: dict[str, IPState] = {}
        self.action_log: list[PreventionAction] = []
        self.blocked_ips: set = set()
        self.stats = {
            "total_windows_processed": 0,
            "total_attacks_detected": 0,
            "total_ips_blocked": 0,
            "total_ips_unblocked": 0,
            "actions_by_type": defaultdict(int),
            "blocks_by_attack": defaultdict(int),
        }

        self.logger = logging.getLogger("IPS")
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.log_dir / "prevention.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        ))
        self.logger.addHandler(handler)

        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(
            "[IPS %(levelname)s] %(message)s"
        ))
        self.logger.addHandler(console)

        self.logger.info("Prevention Engine initialized (dry_run=%s, platform=%s)",
                         dry_run, self._platform)
        self._lock = threading.Lock()

        if self._platform == "Linux" and not self.dry_run:
            self._init_ipset()

    def _init_ipset(self):
        try:
            subprocess.run(
                ["ipset", "create", self._IPSET_NAME, "hash:net", "-exist"],
                capture_output=True, text=True, timeout=5
            )
            check = subprocess.run(
                ["iptables", "-C", "INPUT", "-m", "set",
                 "--match-set", self._IPSET_NAME, "src", "-j", "DROP"],
                capture_output=True, text=True, timeout=5
            )
            if check.returncode != 0:
                subprocess.run(
                    ["iptables", "-I", "INPUT", "-m", "set",
                     "--match-set", self._IPSET_NAME, "src", "-j", "DROP"],
                    capture_output=True, text=True, timeout=5
                )
            self.logger.info("Linux ipset '%s' initialized with iptables rule",
                             self._IPSET_NAME)
        except FileNotFoundError:
            self.logger.warning(
                "ipset/iptables not found — falling back to individual iptables rules. "
                "Install with: apt-get install ipset iptables"
            )
        except Exception as e:
            self.logger.error("Failed to initialize ipset: %s", e)

    def process_detection(self, source_ip: str, attack_class: str,
                          confidence: float = 0.0, window_id: int = 0):
        with self._lock:
            now = datetime.now()
            self.stats["total_windows_processed"] += 1

            if source_ip not in self.ip_states:
                self.ip_states[source_ip] = IPState(
                    ip=source_ip, first_seen=now
                )
            state = self.ip_states[source_ip]
            state.last_seen = now
            state.total_windows += 1

            if state.blocked and state.blocked_until:
                if now > state.blocked_until:
                    self._unblock_ip(source_ip, state)
                else:
                    self.logger.debug(
                        "IP %s still blocked until %s, ignoring detection",
                        source_ip, state.blocked_until.strftime("%H:%M:%S")
                    )
                    return

            if attack_class == "Normal":
                return

            self.stats["total_attacks_detected"] += 1
            state.total_attacks += 1
            state.detection_count[attack_class] += 1

            severity = SEVERITY_MAP.get(attack_class, "MEDIUM")
            policy = RESPONSE_POLICIES.get(attack_class, {})
            threshold = policy.get("threshold_windows", 3)

            self.logger.warning(
                "ATTACK DETECTED: %s from %s (confidence=%.2f, "
                "count=%d/%d threshold, severity=%s)",
                attack_class, source_ip, confidence,
                state.detection_count[attack_class], threshold, severity
            )

            if state.detection_count[attack_class] >= threshold:
                action = policy.get("action", "alert")
                if action == "block_ip":
                    duration = policy.get("block_duration_minutes", 30)
                    self._block_ip(source_ip, state, attack_class,
                                   duration, confidence, window_id)
            else:
                self._log_action(PreventionAction(
                    timestamp=now.isoformat(),
                    source_ip=source_ip,
                    attack_type=attack_class,
                    severity=severity,
                    action_taken="ALERT",
                    rule_name="",
                    confidence=confidence,
                    window_id=window_id,
                    details=f"Detection {state.detection_count[attack_class]}/{threshold} "
                            f"before auto-block",
                ))

    def _block_ip(self, ip: str, state: IPState, attack_class: str,
                  duration_minutes: int, confidence: float, window_id: int):
        now = datetime.now()
        block_until = now + timedelta(minutes=duration_minutes)
        rule_name = f"IPS_BLOCK_{ip.replace('.', '_')}_{attack_class.replace('/', '_')}"

        success = self._add_firewall_rule(ip, rule_name)

        state.blocked = True
        state.blocked_until = block_until
        state.block_rule_name = rule_name
        self.blocked_ips.add(ip)
        self.stats["total_ips_blocked"] += 1
        self.stats["blocks_by_attack"][attack_class] += 1

        action_str = "BLOCKED" if success else "BLOCK_FAILED"

        self.logger.critical(
            "IP %s %s for %d minutes (until %s) — Attack: %s, Confidence: %.2f",
            ip, action_str, duration_minutes,
            block_until.strftime("%H:%M:%S"), attack_class, confidence
        )

        self._log_action(PreventionAction(
            timestamp=now.isoformat(),
            source_ip=ip,
            attack_type=attack_class,
            severity=SEVERITY_MAP.get(attack_class, "HIGH"),
            action_taken=action_str,
            rule_name=rule_name,
            block_until=block_until.isoformat(),
            confidence=confidence,
            window_id=window_id,
            details=f"Firewall rule added: {rule_name}, duration: {duration_minutes}min",
        ))

    def _unblock_ip(self, ip: str, state: IPState):
        if state.block_rule_name:
            self._remove_firewall_rule(state.block_rule_name, ip=ip)

        state.blocked = False
        state.blocked_until = None
        state.detection_count.clear()
        self.blocked_ips.discard(ip)
        self.stats["total_ips_unblocked"] += 1

        self.logger.info("IP %s UNBLOCKED (block expired)", ip)

        self._log_action(PreventionAction(
            timestamp=datetime.now().isoformat(),
            source_ip=ip,
            attack_type="",
            severity="INFO",
            action_taken="UNBLOCKED",
            rule_name=state.block_rule_name or "",
            details="Block duration expired",
        ))
        state.block_rule_name = None

    def _add_firewall_rule(self, ip: str, rule_name: str) -> bool:
        import re
        if not re.match(r'^\d{1,3}(\.\d{1,3}){3}$', ip):
            self.logger.error("Invalid IP format rejected: %s", ip)
            return False

        if self.dry_run:
            backend = "ipset" if self._platform == "Linux" else "netsh"
            self.logger.info("[DRY RUN] Would block %s via %s", ip, backend)
            return True

        if self._platform == "Linux":
            return self._linux_block_ip(ip)
        else:
            return self._windows_add_rule(ip, rule_name)

    def _remove_firewall_rule(self, rule_name: str, ip: str = None) -> bool:
        if self.dry_run:
            backend = "ipset" if self._platform == "Linux" else "netsh"
            self.logger.info("[DRY RUN] Would unblock via %s", backend)
            return True

        if self._platform == "Linux" and ip:
            return self._linux_unblock_ip(ip)
        else:
            return self._windows_remove_rule(rule_name)

    def _windows_add_rule(self, ip: str, rule_name: str) -> bool:
        cmd = (
            f'netsh advfirewall firewall add rule name="{rule_name}" '
            f'dir=in action=block remoteip={ip} '
            f'protocol=any enable=yes'
        )
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                self.logger.info("Windows firewall rule added: %s", rule_name)
                return True
            else:
                self.logger.error(
                    "Windows firewall rule failed: %s — %s", rule_name, result.stderr
                )
                return False
        except Exception as e:
            self.logger.error("Windows firewall exception: %s", e)
            return False

    def _windows_remove_rule(self, rule_name: str) -> bool:
        cmd = f'netsh advfirewall firewall delete rule name="{rule_name}"'
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=10
            )
            return result.returncode == 0
        except Exception:
            return False

    def _linux_block_ip(self, ip: str) -> bool:
        try:
            result = subprocess.run(
                ["ipset", "add", self._IPSET_NAME, ip, "-exist"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                self.logger.info("ipset: blocked %s", ip)
                return True
        except FileNotFoundError:
            pass

        try:
            result = subprocess.run(
                ["iptables", "-I", "INPUT", "-s", ip, "-j", "DROP"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                self.logger.info("iptables: blocked %s", ip)
                return True
            else:
                self.logger.error("iptables block failed for %s: %s", ip, result.stderr)
                return False
        except Exception as e:
            self.logger.error("Linux firewall exception: %s", e)
            return False

    def _linux_unblock_ip(self, ip: str) -> bool:
        try:
            result = subprocess.run(
                ["ipset", "del", self._IPSET_NAME, ip, "-exist"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                self.logger.info("ipset: unblocked %s", ip)
                return True
        except FileNotFoundError:
            pass

        try:
            result = subprocess.run(
                ["iptables", "-D", "INPUT", "-s", ip, "-j", "DROP"],
                capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    def _log_action(self, action: PreventionAction):
        self.action_log.append(action)
        self.stats["actions_by_type"][action.action_taken] += 1

    def get_status(self) -> dict:
        return {
            "engine_status": "ACTIVE",
            "dry_run": self.dry_run,
            "blocked_ips": list(self.blocked_ips),
            "blocked_count": len(self.blocked_ips),
            "stats": dict(self.stats),
            "recent_actions": [asdict(a) for a in self.action_log[-20:]],
            "response_policies": RESPONSE_POLICIES,
        }

    def export_log(self):
        log_path = self.log_dir / f"actions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_path, "w") as f:
            json.dump({
                "actions": [asdict(a) for a in self.action_log],
                "stats": {k: dict(v) if isinstance(v, defaultdict) else v
                          for k, v in self.stats.items()},
                "blocked_ips": list(self.blocked_ips),
            }, f, indent=2)
        self.logger.info("Action log exported to %s", log_path)
        return log_path

    def cleanup_all_rules(self):
        self.logger.info("Cleaning up all IPS firewall rules...")
        for ip, state in self.ip_states.items():
            if state.blocked and state.block_rule_name:
                self._remove_firewall_rule(state.block_rule_name, ip=ip)
                state.blocked = False
        self.blocked_ips.clear()

        if self._platform == "Linux" and not self.dry_run:
            try:
                subprocess.run(
                    ["ipset", "flush", self._IPSET_NAME],
                    capture_output=True, text=True, timeout=5
                )
                subprocess.run(
                    ["iptables", "-D", "INPUT", "-m", "set",
                     "--match-set", self._IPSET_NAME, "src", "-j", "DROP"],
                    capture_output=True, text=True, timeout=5
                )
                subprocess.run(
                    ["ipset", "destroy", self._IPSET_NAME],
                    capture_output=True, text=True, timeout=5
                )
            except Exception as e:
                self.logger.error("ipset cleanup failed: %s", e)

        self.logger.info("All IPS rules cleaned up (platform=%s)", self._platform)


def demo():
    print("\n" + "=" * 60)
    print("  INTRUSION PREVENTION SYSTEM — DEMO")
    print("=" * 60)

    engine = PreventionEngine(dry_run=True)

    attacks = [
        ("192.168.1.100", "Normal", 0.95),
        ("10.0.0.50", "PortScan/Recon", 0.82),
        ("10.0.0.50", "PortScan/Recon", 0.85),
        ("10.0.0.50", "PortScan/Recon", 0.88),
        ("10.0.0.50", "PortScan/Recon", 0.90),
        ("10.0.0.50", "PortScan/Recon", 0.91),
        ("172.16.0.1", "Web/SQLi", 0.97),
        ("192.168.1.200", "DoS", 0.78),
        ("192.168.1.200", "DoS", 0.81),
        ("192.168.1.200", "DoS", 0.84),
        ("10.0.0.99", "DDoS", 0.92),
        ("10.0.0.99", "DDoS", 0.94),
        ("172.16.0.5", "Malware/Botnet/Exploit", 0.88),
        ("10.0.0.50", "PortScan/Recon", 0.95),
    ]

    for i, (ip, attack, conf) in enumerate(attacks):
        engine.process_detection(ip, attack, confidence=conf, window_id=i)
        time.sleep(0.1)

    status = engine.get_status()
    print(f"\n{'=' * 60}")
    print(f"  IPS SESSION SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Windows processed:  {status['stats']['total_windows_processed']}")
    print(f"  Attacks detected:   {status['stats']['total_attacks_detected']}")
    print(f"  IPs blocked:        {status['blocked_count']}")
    print(f"  Blocked IPs:        {', '.join(status['blocked_ips'])}")

    print(f"\n  Actions taken:")
    for action_type, count in status['stats']['actions_by_type'].items():
        print(f"    {action_type}: {count}")

    print(f"\n  Blocks by attack type:")
    for attack, count in status['stats']['blocks_by_attack'].items():
        print(f"    {attack}: {count}")

    print(f"\n  Response Policies:")
    print(f"  {'Attack Type':<25} {'Action':<12} {'Threshold':>10} {'Duration':>10}")
    print(f"  {'-' * 60}")
    for attack, policy in RESPONSE_POLICIES.items():
        action = policy.get("action", "allow")
        thresh = policy.get("threshold_windows", "-")
        dur = policy.get("block_duration_minutes", "-")
        dur_str = f"{dur}min" if dur != "-" else "-"
        print(f"  {attack:<25} {action:<12} {str(thresh):>10} {dur_str:>10}")

    log_path = engine.export_log()
    print(f"\n  Log exported: {log_path}")

    print(f"\n{'=' * 60}")
    print(f"  IPS DEMO COMPLETE")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    demo()
