import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(CURRENT_DIR, "outputs")
PRED_FILE = os.path.join(
    OUTPUT_DIR, "nifty50_50stocks_walkforward_predictions.csv"
)


def main():
  if not os.path.exists(PRED_FILE):
    print(f"Error: Could not find {PRED_FILE}. Run multistock_walkforward.py first.")
    return

  df = pd.read_csv(PRED_FILE)
  df["date"] = pd.to_datetime(df["date"])
  dates = sorted(df["date"].unique())
  daily_stats = []

  for d in dates:
    slice_d = df[df["date"] == d].dropna(
        subset=["actual_return", "pred_return"]
    )
    if len(slice_d) < 5:
      continue

    ric, _ = spearmanr(slice_d["pred_return"], slice_d["actual_return"])
    slice_d = slice_d.sort_values("pred_return", ascending=False).reset_index(
        drop=True
    )
    q = max(1, int(len(slice_d) * 0.2))

    top_ret = slice_d.iloc[:q]["actual_return"].mean()
    bot_ret = slice_d.iloc[-q:]["actual_return"].mean()

    daily_stats.append({
        "date": d,
        "rank_ic": 0.0 if np.isnan(ric) else ric,
        "top_quintile_ret": top_ret,
        "bottom_quintile_ret": bot_ret,
        "long_short_spread": top_ret - bot_ret,
        "hit": 1.0 if top_ret > 0 else 0.0,
    })

  daily_df = pd.DataFrame(daily_stats)
  daily_df["cumulative_ic"] = daily_df["rank_ic"].cumsum()

  records = []
  for sym, grp in df.groupby("symbol"):
    ic, _ = spearmanr(grp["pred_return"], grp["actual_return"])
    mae = np.mean(np.abs(grp["close"] - grp["pred_close"]))
    rmse = np.sqrt(np.mean((grp["close"] - grp["pred_close"]) ** 2))
    mda = np.mean((grp["actual_return"] * grp["pred_return"]) > 0) * 100.0

    records.append({
        "Symbol": sym,
        "Stock_IC": ic,
        "Price_MAE": mae,
        "Price_RMSE": rmse,
        "MDA_%": mda,
        "Mean_Return_Residual_%": grp["return_residual_%"].mean(),
        "Std_Return_Residual_%": grp["return_residual_%"].std(),
    })

  res_df = pd.DataFrame(records)
  mean_ic = daily_df["rank_ic"].mean()
  icir = (mean_ic / (daily_df["rank_ic"].std() + 1e-9)) * np.sqrt(252)
  daily_ls = daily_df["long_short_spread"].mean() * 100.0

  res_df["Cross_Sectional_Rank_IC"] = mean_ic
  res_df["ICIR"] = icir
  res_df["Daily_LS_Spread_%"] = daily_ls
  res_df["Annualized_LS_%"] = daily_ls * 252.0
  res_df["Top_Quintile_Hit_Rate_%"] = daily_df["hit"].mean() * 100.0
  res_df["Final_Cumulative_IC"] = daily_df["cumulative_ic"].iloc[-1]

  out_csv = os.path.join(OUTPUT_DIR, "cross_sectional_stock_metrics.csv")
  daily_csv = os.path.join(OUTPUT_DIR, "daily_alpha_series.csv")
  res_df.to_csv(out_csv, index=False)
  daily_df.to_csv(daily_csv, index=False)

  print("=" * 65)
  print(f"✓ Cross-Sectional Alpha Table Saved: {out_csv}")
  print(f"• Mean Rank IC         : {mean_ic:.4f}")
  print(f"• Annualized ICIR      : {icir:.4f}")
  print(
      f"• Long-Short Spread    : {daily_ls:.2f}% / day ({daily_ls * 252.0:.2f}%"
      " annualized)"
  )
  print(f"• Top Quintile HitRate : {daily_df['hit'].mean() * 100.0:.2f}%")
  print(f"• Cumulative IC        : {daily_df['cumulative_ic'].iloc[-1]:.2f}")
  print("=" * 65)


if __name__ == "__main__":
  main()