import json
import os
import sys
import numpy as np
import optuna
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
os.makedirs(OUTPUT_DIR, exist_ok=True)
optuna.logging.set_verbosity(optuna.logging.WARNING)

CLUSTER_STOCKS = ["ADANIENT", "RELIANCE", "ITC", "TCS", "SUNPHARMA", "MARUTI"]


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


def apply_lora(base_model, r=16, alpha=32, dropout=0.05):
  for param in base_model.parameters():
    param.requires_grad = False
  targets = []
  for name, mod in base_model.named_modules():
    if isinstance(mod, nn.Linear):
      lower = name.lower()
      if any(k in lower for k in ["q_proj", "v_proj", "attn", "mlp"]):
        targets.append(name.split(".")[-1])
  targets = sorted(list(set(targets)))
  if not targets:
    targets = ["q_proj", "v_proj"]
  cfg = LoraConfig(
      r=r,
      lora_alpha=alpha,
      target_modules=targets,
      lora_dropout=dropout,
      bias="none",
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


def cluster_objective(trial, df, tokenizer):
  rw = trial.suggest_int("rolling_window", 402, 482, step=10)
  epochs = trial.suggest_int("adapt_epochs", 10, 14)
  lr = trial.suggest_float("lr", 3e-5, 2e-4, log=True)
  r = trial.suggest_categorical("lora_r", [8, 16, 32])
  alpha = trial.suggest_categorical("lora_alpha", [16, 32, 64])
  decay = trial.suggest_float("recency_decay", 0.25, 0.55)

  basket_errors = []
  for sym in CLUSTER_STOCKS:
    stk = df[df["symbol"] == sym].sort_values("date").reset_index(drop=True)
    split = stk[stk["date"] >= "2023-05-01"].index
    if len(split) == 0:
      continue
    start_idx = split[0]

    train_slice = stk.iloc[max(0, start_idx - rw) : start_idx].reset_index(
        drop=True
    )
    s1, s2 = tokenize_window(train_slice, tokenizer)
    loader = DataLoader(RollingDataset(s1, s2), batch_size=16, shuffle=True)

    base = get_base_model().to(DEVICE)
    lora = apply_lora(base, r=r, alpha=alpha).to(DEVICE)
    train_step(lora, loader, epochs=epochs, lr=lr, recency_decay=decay)

    lora.eval()
    pred = KronosPredictor(lora, tokenizer, max_context=512)
    eval_errors = []
    for step in range(15):
      idx = start_idx + step
      ctx = stk.iloc[max(0, idx - rw) : idx]
      with torch.no_grad():
        res = pred.predict(
            ctx[["open", "high", "low", "close", "volume", "amount"]],
            pd.to_datetime(ctx["date"]),
            pd.to_datetime(stk["date"].iloc[idx : idx + 1]),
            pred_len=1,
        )
      pc = (
          float(res["close"].iloc[-1])
          if isinstance(res, pd.DataFrame)
          else float(res[-1, 3])
      )
      eval_errors.append(
          abs((stk["close"].iloc[idx] - pc) / stk["close"].iloc[idx])
      )

    basket_errors.append(np.mean(eval_errors))
    del lora, base
    torch.cuda.empty_cache()

  return float(np.mean(basket_errors)) * 100.0


if __name__ == "__main__":
  print("--> Ingesting dataset:", DATA_PATH)
  df = pd.read_parquet(DATA_PATH)
  tok = get_tokenizer()

  print("--> Running Optuna study across 6 representative cluster stocks...")
  study = optuna.create_study(
      direction="minimize", sampler=optuna.samplers.TPESampler(seed=42)
  )
  study.optimize(lambda t: cluster_objective(t, df, tok), n_trials=8)

  out_params = os.path.join(OUTPUT_DIR, "best_cluster_hyperparams.json")
  with open(out_params, "w") as f:
    json.dump(study.best_params, f, indent=4)
  print("\n✓ Tuning Complete. Hyperparameters saved to:", out_params)
  print(json.dumps(study.best_params, indent=2))