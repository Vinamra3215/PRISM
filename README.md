<div align="center">

# PRISM

### **Pre-trained Indian Stock Model — Foundation Model for Indian Equity Markets**

*Fine-tuning Kronos with market-conditioned gating for cross-sectional stock ranking on NSE*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co)

**Quant × RAID · IIT Jodhpur**

</div>

---

## 📌 Problem Statement

Publicly available financial foundation models like **Kronos** are pre-trained on global exchanges but have **zero Indian market representation**. Indian equities (NSE/BSE) have distinct microstructure — different trading hours, sectoral dynamics, retail-heavy participation, and unique index correlations — that global models fail to capture out of the box.

**PRISM** addresses this gap by:

1. **Fine-tuning Kronos** (a K-line foundation model) on 5 years of NSE OHLCV data using parameter-efficient methods (LoRA / QLoRA)
2. **Adding a market-conditioned gating head** inspired by MASTER (AAAI 2024) that lets individual stock representations attend over market index signals (NIFTY 50, NIFTY 500, sector indices)
3. **Building a cross-sectional ranking strategy** — long top decile, exclude/short bottom decile on Nifty 500 — with rigorous walk-forward backtesting

---

## 🏗️ Architecture

```
                    ┌──────────────────────┐
                    │   NSE OHLCV Data     │
                    │   (Nifty 500 stocks) │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   K-Line Tokenizer   │
                    │   (OHLCV → tokens)   │
                    └──────────┬───────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │     Kronos Foundation Model     │
              │   (frozen backbone + LoRA/QLoRA │
              │         adapter layers)         │
              └────────────────┬───────────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │  Market-Conditioned Gating Head │
              │  (Cross-attention over NIFTY    │
              │   50 / sector index embeddings) │
              └────────────────┬───────────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │   Cross-Sectional Stock Ranking │
              │   Long top decile / Short bottom│
              │   Monthly rebalancing           │
              └────────────────────────────────┘
```

---

## 🔑 Key Components

### 1. Data Pipeline
- 5 years of NSE/BSE OHLCV for **Nifty 500** stocks via `yfinance`
- Market index feature vectors: NIFTY 50, NIFTY 500, sectoral indices (as conditioning signals)
- Derived features: log returns, rolling volatility, beta, turnover

### 2. Fine-Tuned Kronos Encoder
- Base model: [`NeoQuasar/Kronos-base`](https://huggingface.co/NeoQuasar/Kronos-base) from Hugging Face
- Adapter-based fine-tuning using **LoRA** / **QLoRA** (4-bit quantized backbone)
- Evaluation: next-period return **RankIC** on held-out NSE data vs. zero-shot baseline

### 3. Market-Conditioned Gating Head
- Cross-attention module: stock token representations attend over market index embeddings
- Gating mechanism following MASTER's intra-stock / inter-stock aggregation
- Only the head + adapter layers are trained; Kronos backbone stays **frozen**

### 4. Ranking Strategy & Backtest
- Cross-sectional return ranking on Nifty 500
- Walk-forward backtest with expanding window & monthly rebalancing
- Transaction cost model: brokerage + bid-ask + market impact
- Metrics: **Sharpe, Sortino, Calmar, Max Drawdown, RankIC, Annual Return**

### 5. Dashboard
- Rolling portfolio NAV vs. NIFTY 50 benchmark
- Top/bottom holdings with predicted scores
- Attention heatmap showing which market signals the gating head uses
- Rolling Sharpe, volatility, and drawdown panels

---

## 🛠️ Tech Stack

| Area | Tool | Purpose |
|:-----|:-----|:--------|
| Data | yfinance, nsepy, Kaggle | NSE/BSE OHLCV + index data |
| Foundation Model | Kronos, HF Transformers | Pre-trained K-line encoder |
| Fine-Tuning | PEFT / LoRA / QLoRA, bitsandbytes | Parameter-efficient adaptation |
| Gating Module | PyTorch `nn.MultiheadAttention` | Market-conditioned cross-attention |
| Evaluation | Qlib, pandas, numpy | RankIC, Sharpe, factor attribution |
| Dashboard | Plotly Dash / Gradio | Portfolio viz + attention heatmap |
| CI / Infra | GitHub Actions | Scheduled pipeline, version control |

---

## 📅 Timeline

| Weeks | Phase | Focus |
|:-----:|:------|:------|
| 1–2 | Foundation & Learning | Financial domain knowledge, basic ML/DL |
| 3 | Paper Deep Dive | Kronos & MASTER papers, architecture understanding |
| 4 | Data & Feature Engineering | OHLCV pipeline, market-index features, EDA |
| 5–6 | Kronos Fine-Tuning | LoRA/QLoRA adaptation on NSE data |
| 7–8 | Gating Head | Market-conditioned module, ablation study |
| 9–10 | Backtest, Ablations & Dashboard | Walk-forward backtest, metrics, final dashboard |

---

## 📚 References

| Paper | Link |
|:------|:-----|
| **Kronos** — A Foundation Model for the Language of Financial Markets (Shi et al., 2025) | [GitHub](https://github.com/shiyu-coder/Kronos) · [HuggingFace](https://huggingface.co/NeoQuasar/Kronos-base) |
| **MASTER** — Market-Guided Stock Transformer (Li et al., AAAI 2024) | [GitHub](https://github.com/SJTU-DMTai/MASTER) · [arXiv](https://arxiv.org/abs/2312.15235) |
| **LoRA** — Low-Rank Adaptation of Large Language Models (Hu et al., ICLR 2022) | [arXiv](https://arxiv.org/abs/2106.09685) |
| **QLoRA** — Efficient Finetuning of Quantized LLMs (Dettmers et al., NeurIPS 2023) | [arXiv](https://arxiv.org/abs/2305.14314) |

---

## 🚀 Getting Started

> 🚧 **Project is under active development.** Setup instructions and code will be added as each phase is completed.

```bash
git clone https://github.com/Vinamra3215/PRISM.git
cd PRISM
# pip install -r requirements.txt  (coming soon)
```

---

<div align="center">

*Quant × RAID · IIT Jodhpur*

</div>