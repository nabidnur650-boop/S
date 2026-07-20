from __future__ import annotations

import hashlib
import json
from typing import Any

import yaml

from .common import ROOT, atomic_json, sha256_file

Q1_ROOT = ROOT / "q1"


def ensure_q1_directories() -> None:
    for relative in [
        "artifacts",
        "checkpoints",
        "figures/png",
        "figures/pdf",
        "logs",
        "notebooks",
        "results/tables",
        "manuscript",
        "external_data",
    ]:
        (Q1_ROOT / relative).mkdir(parents=True, exist_ok=True)


def load_q1_config(verify_lock: bool = True) -> dict[str, Any]:
    path = Q1_ROOT / "config_q1.yaml"
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if verify_lock:
        verify_protocol_lock()
    config["_q1_root"] = str(Q1_ROOT)
    return config


def verify_protocol_lock() -> dict[str, Any]:
    lock_path = Q1_ROOT / "LOCKED_PROTOCOL.json"
    if not lock_path.exists():
        raise RuntimeError("Q1 protocol has not been locked")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    for filename, expected in lock["files"].items():
        path = Q1_ROOT / filename
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(f"Locked protocol file changed: {filename}")
    canonical = {key: value for key, value in lock.items() if key != "combined_sha256"}
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    observed_combined = hashlib.sha256(payload).hexdigest()
    if observed_combined != lock["combined_sha256"]:
        raise RuntimeError("Combined protocol hash is invalid")
    return lock


def q1_json(relative: str, payload: Any) -> None:
    atomic_json(Q1_ROOT / relative, payload)

