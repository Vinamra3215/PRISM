| Feature | Stationary? | RankIC | ICIR | Dropped Correlated With | Verdict |
|:---|:---:|:---:|:---:|:---|:---:|
| Log Return (1-day) | Yes | -0.035 | -0.18 | Dropped — corr 0.86 with open_close_spread | **DISCARD (Redundant)** |
| Return 5-day | Yes | -0.040 | -0.19 | — | **KEEP** |
| Return 10-day | Yes | -0.041 | -0.21 | — | **KEEP** |
| Return 20-day | Yes | -0.018 | -0.09 | — | **KEEP** |
| Return 60-day | Yes | -0.012 | -0.06 | — | **DISCARD (Low ICIR)** |
| Return 120-day | Yes | -0.018 | -0.09 | Dropped — corr -0.72 with 52w_high_dist | **DISCARD (Redundant)** |
| Intraday Range | Yes | -0.015 | -0.09 | — | **KEEP** |
| Open-Close Spread | Yes | -0.034 | -0.18 | — | **KEEP** |
| Gap (Overnight Return) | Yes | -0.004 | -0.02 | — | **DISCARD (Low ICIR)** |
| Close-to-High Distance | Yes | 0.024 | 0.13 | Dropped — corr -0.72 with open_close_spread | **DISCARD (Redundant)** |
| 52-Week High Distance | Yes | 0.020 | 0.11 | — | **KEEP** |
| 52-Week Low Distance | Yes | -0.015 | -0.07 | — | **DISCARD (Low ICIR)** |
