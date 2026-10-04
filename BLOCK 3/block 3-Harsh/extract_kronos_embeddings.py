#!/usr/bin/env python3
"""Extract causal final-layer states from the Block 1 Kronos checkpoint.

The supplied checkpoint is Kronos-base with d_model=832 (not 768 as assumed
in the prose specification). No projection or truncation is applied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
ROOT = PROJECT.parent
DEFAULT_PANEL = ROOT / "tanishq/phase2_data/nifty50_phase2_all_features_full.parquet"
DEFAULT_CHECKPOINT = PROJECT / "best_kronos_weights.pth"
DEFAULT_KRONOS_SOURCE = Path("/home/soq/Kronos")
DEFAULT_BASE = Path("/home/soq/.cache/huggingface/hub/models--NeoQuasar--Kronos-base/snapshots/2b554741eca47781b64468546e77fef3e85130e6")
DEFAULT_TOKENIZER = Path("/home/soq/.cache/huggingface/hub/models--NeoQuasar--Kronos-Tokenizer-base/snapshots/0e0117387f39004a9016484a186a908917e22426")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--kronos-source", type=Path, default=DEFAULT_KRONOS_SOURCE)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output-dir", type=Path, default=HERE / "artifacts")
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--normalization-window", type=int, default=252)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def causal_normalize(values: pd.DataFrame, window: int) -> np.ndarray:
    """Normalize every bar using only values available through that bar."""
    mean = values.rolling(window, min_periods=2).mean()
    std = values.rolling(window, min_periods=2).std(ddof=0)
    normalized = (values - mean) / std.replace(0.0, np.nan)
    return normalized.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5).to_numpy(np.float32)


def load_models(args: argparse.Namespace):
    sys.path.insert(0, str(args.kronos_source))
    from model.kronos import Kronos, KronosTokenizer, calc_time_stamps

    base_config = json.loads((args.base_model / "config.json").read_text())
    tokenizer_config = json.loads((args.tokenizer / "config.json").read_text())
    model = Kronos(**base_config)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    tokenizer = KronosTokenizer(**tokenizer_config)
    tokenizer.load_state_dict(load_file(args.tokenizer / "model.safetensors"), strict=True)
    model.eval().to(args.device)
    tokenizer.eval().to(args.device)
    return model, tokenizer, calc_time_stamps, base_config


def chunk_boundaries(length: int, context: int, stride: int):
    first_end = min(context, length)
    yield 0, first_end, 0, first_end
    output_start = first_end
    while output_start < length:
        output_end = min(output_start + stride, length)
        input_start = max(0, output_end - context)
        yield input_start, output_end, output_start, output_end
        output_start = output_end


def main() -> None:
    args = arguments()
    if args.stride <= 0 or args.stride > args.context:
        raise ValueError("stride must be in [1, context]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, calc_time_stamps, base_config = load_models(args)
    hidden_dim = int(base_config["d_model"])

    columns = ["date", "symbol", "open", "high", "low", "close", "volume"]
    panel = pd.read_parquet(args.panel, columns=columns)
    panel["date"] = pd.to_datetime(panel["date"]).dt.tz_localize(None)
    panel = panel.sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"]).reset_index(drop=True)

    embeddings = np.empty((len(panel), hidden_dim), dtype=np.float16)
    records = []
    offset = 0
    amp_enabled = str(args.device).startswith("cuda")

    for stock_number, (symbol, stock) in enumerate(panel.groupby("symbol", sort=True), start=1):
        stock = stock.sort_values("date").reset_index(drop=True)
        price = stock[["open", "high", "low", "close", "volume"]].astype(float).ffill().bfill()
        price["amount"] = price["volume"] * price[["open", "high", "low", "close"]].mean(axis=1)
        normalized = causal_normalize(price, args.normalization_window)
        stamps = calc_time_stamps(stock["date"]).to_numpy(np.float32)
        stock_embeddings = np.empty((len(stock), hidden_dim), dtype=np.float16)

        for input_start, input_end, output_start, output_end in chunk_boundaries(
            len(stock), args.context, args.stride
        ):
            x = torch.from_numpy(normalized[input_start:input_end].copy()).unsqueeze(0).to(args.device)
            stamp = torch.from_numpy(stamps[input_start:input_end].copy()).unsqueeze(0).to(args.device)
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=amp_enabled
            ):
                s1_ids, s2_ids = tokenizer.encode(x, half=True)
                _, context = model.decode_s1(s1_ids, s2_ids, stamp)
            local_start = output_start - input_start
            local_end = output_end - input_start
            stock_embeddings[output_start:output_end] = (
                context[0, local_start:local_end].float().cpu().numpy().astype(np.float16)
            )

        embeddings[offset:offset + len(stock)] = stock_embeddings
        records.append(stock[["date", "symbol"]].assign(embedding_row=np.arange(offset, offset + len(stock))))
        offset += len(stock)
        print(f"[{stock_number:02d}/{panel.symbol.nunique():02d}] {symbol}: {len(stock)} embeddings", flush=True)

    index = pd.concat(records, ignore_index=True)
    if offset != len(panel) or not np.isfinite(embeddings).all():
        raise RuntimeError("Embedding extraction failed completeness/finite-value checks")
    np.save(args.output_dir / "kronos_embeddings_float16.npy", embeddings)
    index.to_parquet(args.output_dir / "kronos_embedding_index.parquet", index=False)
    metadata = {
        "rows": len(index),
        "stocks": int(index.symbol.nunique()),
        "date_min": str(index.date.min().date()),
        "date_max": str(index.date.max().date()),
        "hidden_dim": hidden_dim,
        "dtype": "float16",
        "nan_count": int(np.isnan(embeddings).sum()),
        "infinite_count": int(np.isinf(embeddings).sum()),
        "context": args.context,
        "stride": args.stride,
        "minimum_context_after_warmup": args.context - args.stride + 1,
        "normalization": f"causal rolling z-score, window={args.normalization_window}",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "specification_note": "Actual Block 1 checkpoint d_model is 832; no 768-d truncation was applied.",
    }
    (args.output_dir / "embedding_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
