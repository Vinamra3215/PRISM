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


<div align="center">

*Quant × RAID · IIT Jodhpur*

</div>
