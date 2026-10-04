#!/usr/bin/env python3
"""Run embedding extraction (if needed), then the complete Block 3 pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--tune-epochs", type=int, default=10)
    parser.add_argument("--final-epochs", type=int, default=100)
    parser.add_argument("--force-embeddings", action="store_true")
    args = parser.parse_args()
    embedding = HERE / "artifacts/kronos_embeddings_float16.npy"
    if args.force_embeddings or not embedding.exists():
        subprocess.run([sys.executable, str(HERE / "extract_kronos_embeddings.py")], check=True)
    subprocess.run([
        sys.executable, str(HERE / "block3_pipeline.py"),
        "--trials", str(args.trials), "--tune-epochs", str(args.tune_epochs),
        "--final-epochs", str(args.final_epochs),
    ], check=True)


if __name__ == "__main__":
    main()
