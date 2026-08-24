import pandas as pd
import matplotlib.pyplot as plt

# Read the dataset
df = pd.read_parquet("RELIANCE_5Y_OHLCVdata.parquet")

# Convert index to datetime (only if needed)
df.index = pd.to_datetime(df.index)

# Select date range
start_date = "2024-01-01"
end_date = "2024-06-30"

selected_data = df.loc[start_date:end_date]

# Plot
plt.figure(figsize=(12,6))
plt.plot(selected_data["Close"])

plt.title(f"RELIANCE Close Price ({start_date} to {end_date})")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.grid(True)

plt.savefig("plots/reliance_selected_range.png")
plt.show()