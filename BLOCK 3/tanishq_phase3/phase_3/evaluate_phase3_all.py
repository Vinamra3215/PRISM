import numpy as np
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("/home/soq/__shutupandbendover/tanishq/phase3_results")

models = {
    "Architecture A (Concat+MLP)": RESULTS_DIR / "walk_forward_daily_metrics_arch_a.parquet",
    "Architecture B (MASTER Gating)": RESULTS_DIR / "walk_forward_daily_metrics_arch_b.parquet",
    "Architecture C (FiLM Conditioning)": RESULTS_DIR / "walk_forward_daily_metrics_arch_c.parquet"
}

summary = []

for name, path in models.items():
    if not path.exists():
        print(f"Skipping {name}: file not found at {path}")
        continue
    
    df = pd.read_parquet(path)
    mean_ic = df['rank_ic'].mean()
    std_ic = df['rank_ic'].std()
    icir = mean_ic / (std_ic + 1e-8)
    
    mean_ls = df['ls_spread'].mean()
    std_ls = df['ls_spread'].std()
    ann_return = mean_ls * 252.0
    sharpe = (mean_ls / (std_ls + 1e-8)) * np.sqrt(252.0)
    
    downside = df['ls_spread'][df['ls_spread'] < 0]
    sortino = (mean_ls / (downside.std() + 1e-8)) * np.sqrt(252.0)
    
    cum_returns = (1.0 + df['ls_spread']).cumprod()
    max_dd = ((cum_returns - cum_returns.cummax()) / cum_returns.cummax()).min()
    win_rate = (df['ls_spread'] > 0).mean() * 100.0
    
    summary.append({
        "Architecture": name,
        "Test Days": len(df),
        "Mean Rank IC": f"{mean_ic:.4f}",
        "ICIR": f"{icir:.4f}",
        "Win Rate": f"{win_rate:.2f}%",
        "Ann. L/S Return": f"{ann_return * 100:.2f}%",
        "Sharpe": f"{sharpe:.2f}",
        "Sortino": f"{sortino:.2f}",
        "Max Drawdown": f"{max_dd * 100:.2f}%"
    })

summary_df = pd.DataFrame(summary)

# Pure-python Markdown Table Formatter (Zero external dependencies)
headers = list(summary_df.columns)
rows = summary_df.values.tolist()

col_widths = [max(len(str(val)) for val in [h] + [r[i] for r in rows]) for i, h in enumerate(headers)]

header_line = "| " + " | ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers)) + " |"
separator_line = "|-" + "-|-".join("-" * col_widths[i] for i in range(len(headers))) + "-|"
data_lines = ["| " + " | ".join(f"{str(r[i]):<{col_widths[i]}}" for i, r in enumerate(rows)) for r in rows]

print("\n" + "=" * 105)
print("PRISM PHASE 3: COMPREHENSIVE ARCHITECTURE BENCHMARK (NIFTY50 WALK-FORWARD)")
print("=" * 105)
print(header_line)
print(separator_line)
for line in data_lines:
    print(line)
print("=" * 105 + "\n")

summary_df.to_csv(RESULTS_DIR / "final_phase3_comparison.csv", index=False)
print(f"Summary metrics exported cleanly to: {RESULTS_DIR / 'final_phase3_comparison.csv'}")
