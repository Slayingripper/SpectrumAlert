#!/usr/bin/env python3
"""
Cleanup legacy/unreferenced files for SpectrumAlert workspace.

- Default: dry-run (prints what would be moved)
- Use --apply to actually move items into ./legacy/
- Use --delete to permanently delete instead of moving (use with caution)

This script targets known legacy paths from earlier versions that are not
used by the current spectrum_alert package CLI.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEGACY_DIR = ROOT / "legacy"

CANDIDATES = [
    # Legacy monolithic source tree
    "src",
    # Legacy interactive/menu + services and docker wrappers
    "Main.py",
    "main.py",
    "autonomous_service.py",
    "deploy.sh",
    "run_docker.sh",
    "DOCKER_README.md",
    # One-off analysis/tuning helpers from old pipeline
    "analyze_anomalies.py",
    "tune_anomaly_detection.py",
    # Old trainer implementation
    "Trainer",
    # Legacy model/data artifacts
    "anomaly_detection_model_lite.pkl",
    "rf_fingerprinting_model_lite.pkl",
    "collected_data_lite.csv",
]

SAFE_KEEP = {
    # Keep the active package, CLI, and app
    "spectrum_alert",
    "README.md",
    "LICENSE",
    "config",
    "data",
    "models",
    "logs",
}

def find_existing(paths: list[str]) -> list[Path]:
    existing: list[Path] = []
    for p in paths:
        path = ROOT / p
        if path.exists():
            existing.append(path)
    return existing


def move_to_legacy(paths: list[Path], delete: bool = False) -> None:
    LEGACY_DIR.mkdir(exist_ok=True)
    for path in paths:
        rel = path.relative_to(ROOT)
        target = LEGACY_DIR / rel
        if delete:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
            print(f"DELETED: {rel}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.move(str(path), str(target))
                print(f"MOVED: {rel} -> legacy/{rel}")
            except Exception as e:
                print(f"SKIP (move failed): {rel} ({e})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean up legacy files")
    parser.add_argument("--apply", action="store_true", help="Apply changes (move to legacy)")
    parser.add_argument("--delete", action="store_true", help="Delete instead of moving (danger)")
    args = parser.parse_args()

    existing = find_existing(CANDIDATES)

    if not existing:
        print("No legacy candidates found. Nothing to do.")
        return

    print("Detected legacy candidates:")
    for p in existing:
        print(f"  - {p.relative_to(ROOT)}")

    if not args.apply:
        print("\nDry run. Use --apply to move into ./legacy. Use --delete to permanently delete.")
        return

    move_to_legacy(existing, delete=bool(args.delete))
    print("\nCleanup complete.")


if __name__ == "__main__":
    main()
