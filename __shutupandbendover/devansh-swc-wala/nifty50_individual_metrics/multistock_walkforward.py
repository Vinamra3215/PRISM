import json
import os
import sys
import numpy as np
import pandas as pd
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file as load_safetensors
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEVANSH_DIR = os.path.dirname(CURRENT_DIR)
MY_KRONOS_PROJECT = os.path.join(DEVANSH_DIR, "my_kronos_project")
KRONOS_REPO = "/home/soq/Kronos"

for p in [CURRENT_DIR, DEVANSH_DIR, MY_KRONOS_PROJECT, KRONOS_REPO]:
  if os.path.exists(p) and p not in sys.path:
    sys.path.insert(0, p)

from model.kronos import Kronos, KronosPredictor, KronosTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = os.path.join(CURRENT_DIR, "nifty50_constituents_2022_2026.parquet")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "outputs")
PRED_FILE = os.path.join(
    OUTPUT_DIR, "nifty50_50stocks_walkforward_predictions.csv"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_tokenizer():
  target = "/home/soq/.cache/huggingface/hub/models--NeoQuasar--Kronos-Tokenizer-base/snapshots"
  snap = [
      os.path.join(target, d)
      for d in os.listdir(target)
      if os.path.isdir(os.path.join(target, d))
  ][0]
  with open(os.path.join(snap, "config.json"), "r") as f:
    cfg = json.load(f)
  for k in [
      "model_type",
      "architectures",
      "auto_map",
      "torch_dtype",
      "transformers_version",
  ]:
    cfg.pop(k, None)
  t = KronosTokenizer(**cfg)
  sf = os.path.join(snap, "model.safetensors")
  if os.path.exists(sf):
    t.load_state_dict(load_safetensors(sf))
  else:
    t.load_state_dict(
        torch.load(
            os.path.join(snap, "pytorch_model.bin"), map_location="cpu"
        )
    )
  if hasattr(t, "to"):
    t = t.to(DEVICE)
  return t.eval()


def get_base_model():
  target = (
      "/home/soq/.cache/huggingface/hub/models--NeoQuasar--Kronos-small/snapshots"
  )
  snap = [
      os.path.join(target, d)
      for d in os.listdir(target)
      if os.path.isdir(os.path.join(target, d))
  ][0]
  return Kronos.from_pretrained(snap)


def apply_lora(base_model, r=16, alpha=32):
  for param in base_model.parameters():
    param.requires_grad = False
  targets = []
  for name, mod in base_model.named_modules():
    if isinstance(mod, nn.Linear):
      lower = name.lower()
      if any(k in lower for k in ["q_proj", "v_proj", "attn", "mlp"]):
        targets.append(name.split(".")[-1])
  targets = sorted(list(set(targets)))
  cfg = LoraConfig(
      r=r,
      lora_alpha=alpha,
      target_modules=targets or ["q_proj", "v_proj"],
      task_type=None,
  )
  return get_peft_model(base_model, cfg)


def tokenize_window(df_window, tokenizer, patch_size=16):
  cols = ["open", "high", "low", "close", "volume", "amount"]
  windows = [
      df_window[cols].iloc[i : i + patch_size].values
      for i in range(len(df_window) - patch_size + 1)
  ]
  tokenizer_device = (
      next(tokenizer.parameters()).device
      if list(tokenizer.parameters())
      else torch.device(DEVICE)
  )
  raw = torch.tensor(np.array(windows), dtype=torch.float32).to(
      tokenizer_device
  )
  s1, s2 = [], []
  with torch.no_grad():
    for b in range(0, len(raw), 32):
      enc = tokenizer.encode(raw[b : b + 32])
      if isinstance(enc, (tuple, list)):
        s1.append(enc[0].detach().cpu().flatten())
        s2.append(enc[1].detach().cpu().flatten())
      elif isinstance(enc, dict):
        s1.append(enc["s1_ids"].detach().cpu().flatten())
        s2.append(enc["s2_ids"].detach().cpu().flatten())
      else:
        s1.append(enc[..., 0].detach().cpu().flatten())
        s2.append(enc[..., 1].detach().cpu().flatten())
  return torch.clamp(torch.cat(s1, dim=0), 0, 1023), torch.clamp(
      torch.cat(s2, dim=0), 0, 1023
  )


class RollingDataset(Dataset):

  def __init__(self, s1, s2, seq_len=32):
    self.s1, self.s2, self.seq_len = s1.long(), s2.long(), seq_len

  def __len__(self):
    return max(1, len(self.s1) - self.seq_len)

  def __getitem__(self, idx):
    return {
        "s1_ids": self.s1[idx : idx + self.seq_len],
        "s2_ids": self.s2[idx : idx + self.seq_len],
        "s1_targets": self.s1[idx + 1 : idx + self.seq_len + 1],
        "s2_targets": self.s2[idx + 1 : idx + self.seq_len + 1],
    }


def train_step(model, loader, epochs, lr, recency_decay):
  model.train()
  opt = torch.optim.AdamW(
      filter(lambda p: p.requires_grad, model.parameters()), lr=lr
  )
  crit = nn.CrossEntropyLoss(reduction="none")
  for _ in range(epochs):
    for b in loader:
      s1, s2, tgt = (
          b["s1_ids"].to(DEVICE),
          b["s2_ids"].to(DEVICE),
          b["s1_targets"].to(DEVICE),
      )
      w = torch.linspace(recency_decay, 1.0, steps=s1.size(1), device=DEVICE)
      opt.zero_grad()
      out = model(s1_ids=s1, s2_ids=s2)
      logits = out.logits if hasattr(out, "logits") else out[0]
      loss = (
          crit(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1)).view(
              s1.size(0), s1.size(1)
          )
          * w
      ).mean()
      loss.backward()
      opt.step()


if __name__ == "__main__":
  hp_path = os.path.join(OUTPUT_DIR, "best_cluster_hyperparams.json")
  if os.path.exists(hp_path):
    with open(hp_path, "r") as f:
      hp = json.load(f)
  else:
    print(f"Warning: {hp_path} not found. Using defaults.")
    hp = {
        "rolling_window": 442,
        "adapt_epochs": 10,
        "lr": 8e-5,
        "lora_r": 16,
        "lora_alpha": 32,
        "recency_decay": 0.35,
    }

  df = pd.read_parquet(DATA_PATH)
  tok = get_tokenizer()
  symbols = sorted(df["symbol"].unique())
  rw = int(hp.get("rolling_window", 442))
  epochs = int(hp.get("adapt_epochs", 10))
  lr = float(hp.get("lr", 8e-5))
  decay = float(hp.get("recency_decay", 0.35))

  completed_symbols = set()
  if os.path.exists(PRED_FILE):
    existing_df = pd.read_csv(PRED_FILE)
    completed_symbols = set(existing_df["symbol"].unique())
    print(f"--> Found checkpoint with {len(completed_symbols)} finished stocks.")

  for s_idx, sym in enumerate(symbols):
    if sym in completed_symbols:
      print(f"Skipping [{s_idx+1}/{len(symbols)}] {sym} (Already processed)")
      continue

    stk = df[df["symbol"] == sym].sort_values("date").reset_index(drop=True)
    split = stk[stk["date"] >= "2023-07-01"].index
    if len(split) == 0:
      continue
    start_idx = split[0]

    print(f"\nProcessing [{s_idx+1}/{len(symbols)}] {sym}...")
    base = get_base_model().to(DEVICE)
    lora = apply_lora(
        base, r=hp.get("lora_r", 16), alpha=hp.get("lora_alpha", 32)
    ).to(DEVICE)

    # Initial adaptation using tuned window
    init_slice = stk.iloc[max(0, start_idx - rw) : start_idx].reset_index(
        drop=True
    )
    s1, s2 = tokenize_window(init_slice, tok)
    loader = DataLoader(RollingDataset(s1, s2), batch_size=16, shuffle=True)
    train_step(
        lora, loader, epochs=epochs, lr=lr, recency_decay=decay
    )

    pred = KronosPredictor(lora, tok, max_context=512)
    stock_preds = []

    for step in range(len(stk) - start_idx):
      idx = start_idx + step
      prev_close = stk["close"].iloc[idx - 1]
      actual_close = stk["close"].iloc[idx]
      ctx = stk.iloc[max(0, idx - rw) : idx]

      with torch.no_grad():
        res = pred.predict(
            ctx[["open", "high", "low", "close", "volume", "amount"]],
            pd.to_datetime(ctx["date"]),
            pd.to_datetime(stk["date"].iloc[idx : idx + 1]),
            pred_len=1,
        )

      if isinstance(res, pd.DataFrame):
        po, ph, pl, pc = (
            float(res["open"].iloc[-1]),
            float(res["high"].iloc[-1]),
            float(res["low"].iloc[-1]),
            float(res["close"].iloc[-1]),
        )
      else:
        po, ph, pl, pc = (
            float(res[-1, 0]),
            float(res[-1, 1]),
            float(res[-1, 2]),
            float(res[-1, 3]),
        )

      r_act = (actual_close - prev_close) / prev_close
      r_pred = (pc - prev_close) / prev_close

      stock_preds.append({
          "symbol": sym,
          "date": stk["date"].iloc[idx],
          "open": stk["open"].iloc[idx],
          "high": stk["high"].iloc[idx],
          "low": stk["low"].iloc[idx],
          "close": actual_close,
          "pred_open": po,
          "pred_high": max(ph, po, pc),
          "pred_low": min(pl, po, pc),
          "pred_close": pc,
          "actual_return": r_act,
          "pred_return": r_pred,
          "return_residual_%": (r_act - r_pred) * 100.0,
      })

      # Daily online 1-epoch adaptation using full tuned window
      s1_d, s2_d = tokenize_window(ctx, tok)
      train_step(
          lora,
          DataLoader(
              RollingDataset(s1_d, s2_d), batch_size=8, shuffle=True
          ),
          epochs=1,
          lr=lr * 0.5,
          recency_decay=decay,
      )

    del lora, base
    torch.cuda.empty_cache()

    df_sym = pd.DataFrame(stock_preds)
    df_sym.to_csv(
        PRED_FILE, mode="a", header=not os.path.exists(PRED_FILE), index=False
    )
    print(f"✓ Saved {sym} to {PRED_FILE}")