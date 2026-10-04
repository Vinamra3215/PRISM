# C2-C3 Feature Selection Deliverable

LightGBM was trained on 2022 only and evaluated on 2023 only. No random split or shuffle was used. The last 2022 feature row per stock was excluded because its next-day label falls in 2023.

Candidate pool: 43 features; training rows: 12,103; validation rows: 12,094; stocks: 50.

## Final 19 features

| # | Feature | Category | RankIC | ICIR | LightGBM Rank | Why Selected |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | ret_10d | Price & Returns | -0.0410 | -0.2100 | 9 | Best Price & Returns ICIR and a top-10 LightGBM feature. |
| 2 | 52w_high_dist | Price & Returns | 0.0200 | 0.1100 | 8 | Long-horizon price-position signal with positive ICIR and high gain. |
| 3 | intraday_range | Price & Returns | -0.0150 | -0.0900 | 12 | Adds short-horizon price dispersion not captured by return windows. |
| 4 | rolling_volatility_10d | Volatility | -0.0092 | -0.0486 | 10 | Highest-gain stock-level volatility feature. |
| 5 | volatility_ratio_20d_60d | Volatility | -0.0115 | -0.0712 | 18 | Captures volatility-regime change and ranks well by gain. |
| 6 | atr_14d | Volatility | -0.0079 | -0.0422 | 23 | Range-based volatility measure complementary to return volatility. |
| 7 | ma_ratio_50_200 | Momentum & Trend | 0.0007 | 0.0033 | 7 | Highest-gain stock-level trend feature by a wide margin. |
| 8 | macd_hist | Momentum & Trend | -0.0231 | -0.1335 | 21 | Strong daily ICIR and a faster trend-change signal than the MA ratio. |
| 9 | mfi | Volume | -0.0229 | -0.1247 | 20 | Combines price direction and volume; strong ICIR and useful model gain. |
| 10 | volume_price_trend | Volume | 0.0084 | 0.0406 | 11 | Highest-gain pure volume-category feature. |
| 11 | keltner_channel_position | Candlestick | -0.0405 | -0.2045 | 16 | Best candlestick/channel feature in both ICIR and gain evidence. |
| 12 | upper_shadow_ratio | Candlestick | 0.0201 | 0.1233 | 19 | Strong ICIR and a distinct intraday rejection signal. |
| 13 | nifty_daily_return | Market-Level | 0.1509 | 0.4272 | 1 | Top LightGBM feature; controls for the current market move. |
| 14 | nifty_5d_return | Market-Level | 0.0658 | -0.4056 | 3 | Top-ranked medium-horizon market regime feature. |
| 15 | pharma_return | Market-Level | 0.1114 | 0.2659 | 2 | Strong sector-level gain beyond broad NIFTY returns. |
| 16 | cs_return_z | Cross-Sectional | -0.0336 | -0.1726 | 13 | Best cross-sectional ICIR and highest gain in its category. |
| 17 | sector_relative_return | Cross-Sectional | -0.0120 | -0.0767 | 28 | Adds sector-relative context to the universe-wide z-score. |
| 18 | mean_reversion | Derived | 0.0293 | 0.1635 | 30 | Best supplied Derived ICIR with useful conditional gain. |
| 19 | vol_weight_mom | Derived | -0.0229 | 0.0988 | 17 | Best-gain Derived interaction among the supplied composite signals. |

## Category distribution

| Category | Selected Features |
| --- | --- |
| Candlestick | 2 |
| Cross-Sectional | 2 |
| Derived | 2 |
| Market-Level | 3 |
| Momentum & Trend | 2 |
| Price & Returns | 3 |
| Volatility | 3 |
| Volume | 2 |

## Validation note

The ranking model's 2023 RMSE was 0.016105 versus 0.015756 for a constant training-mean baseline; its mean daily RankIC was 0.0151. This weak out-of-sample result is why gain importance was combined with Phase 1 IC/ICIR evidence and category coverage instead of being used alone.

The complete 43-feature gain ranking is in `c2_lightgbm_feature_importance.csv`; exact metrics and split boundaries are in `c2_validation_metrics.json`.

Phase 1 values were copied from the available category deliverables. Where a category deliverable did not contain a value, the script computes a clearly labeled 2022-2023 fallback. Category 8 provided absolute ICIR values, so their sign must not be inferred from the table.
