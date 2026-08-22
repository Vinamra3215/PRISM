from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import mean_absolute_error, mean_squared_error

import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =========================================================
# RANDOM SEED
# =========================================================

torch.manual_seed(42)
np.random.seed(42)


# =========================================================
# 1. CONFIGURATION
# =========================================================

@dataclass
class LoRAConfig:
    r: int = 4
    alpha: float = 1.0
    dropout: float = 0.05
    learning_rate: float = 0.005
    epochs: int = 10


DATA_FILE = "NIFTY50_2022_to_today.parquet"

# Sliding window settings
WINDOW_SIZE = 308
LOOKBACK = 30

# Initial base model training
PRETRAIN_EPOCHS = 100
PRETRAIN_LEARNING_RATE = 0.01

# Output files
CSV_FILE = "nifty_308day_lora_predictions.csv"
HTML_FILE = "nifty_308day_lora_finetuned.html"

# LoRA settings
lora_config = LoRAConfig(
    r=4,
    alpha=1.0,
    dropout=0.05,
    learning_rate=0.005,
    epochs=10
)


# =========================================================
# 2. LORA LINEAR LAYER
# =========================================================

class LoRALinear(nn.Module):

    def __init__(self, in_features, out_features, config):
        super().__init__()

        self.scaling = config.alpha / config.r

        # Original base layer
        self.linear = nn.Linear(
            in_features,
            out_features
        )

        # LoRA dropout
        self.dropout = nn.Dropout(config.dropout)

        # LoRA A matrix
        self.lora_A = nn.Parameter(
            torch.randn(config.r, in_features) * 0.01
        )

        # LoRA B matrix
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, config.r)
        )


    def freeze_base_weights(self):

        self.linear.weight.requires_grad = False

        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False


    def forward(self, x):

        # Original base model output
        base_output = self.linear(x)

        # LoRA update
        lora_output = (
            self.dropout(x)
            @ self.lora_A.T
            @ self.lora_B.T
        ) * self.scaling

        return base_output + lora_output


# =========================================================
# 3. NEURAL NETWORK + LORA
# =========================================================

class LoRAModel(nn.Module):

    def __init__(self, lookback, config):
        super().__init__()

        self.layer1 = LoRALinear(
            lookback,
            32,
            config
        )

        self.relu = nn.ReLU()

        self.layer2 = LoRALinear(
            32,
            1,
            config
        )


    def freeze_base(self):

        self.layer1.freeze_base_weights()
        self.layer2.freeze_base_weights()


    def forward(self, x):

        x = self.layer1(x)

        x = self.relu(x)

        x = self.layer2(x)

        return x


# =========================================================
# 4. LOAD DATA
# =========================================================

print("\n" + "=" * 70)
print("LOADING NIFTY50 DATA")
print("=" * 70)

df = pd.read_parquet(DATA_FILE)

print("\nAvailable columns:")
print(df.columns.tolist())


# =========================================================
# 5. FIND DATE COLUMN
# =========================================================

if "Date" in df.columns:

    date_col = "Date"

else:

    date_col = df.columns[0]

    print(
        f"\nDate column not directly named 'Date'. "
        f"Using first column: {date_col}"
    )


# =========================================================
# 6. FIND CLOSE COLUMN
# =========================================================

if "Close" in df.columns:

    close_col = "Close"

elif "Adj Close" in df.columns:

    close_col = "Adj Close"

else:

    raise ValueError(
        f"Close price column not found.\n"
        f"Available columns: {df.columns.tolist()}"
    )


# =========================================================
# 7. CLEAN DATA
# =========================================================

df["Date"] = pd.to_datetime(
    df[date_col],
    errors="coerce"
)

df["Close"] = pd.to_numeric(
    df[close_col],
    errors="coerce"
)

df = df[
    ["Date", "Close"]
].dropna()

df = df.sort_values("Date")

df = df.drop_duplicates(
    subset=["Date"]
)

df = df.reset_index(drop=True)


print("\n" + "=" * 70)
print("CLEANED DATA")
print("=" * 70)

print(f"First Date : {df['Date'].iloc[0].date()}")
print(f"Last Date  : {df['Date'].iloc[-1].date()}")
print(f"Total Rows : {len(df)}")


# =========================================================
# 8. CHECK DATA
# =========================================================

minimum_required = WINDOW_SIZE + 2

if len(df) < minimum_required:

    raise ValueError(
        f"\nNot enough data.\n"
        f"Need at least {minimum_required} rows.\n"
        f"Available: {len(df)}"
    )


# =========================================================
# 9. CONVERT TO NUMPY
# =========================================================

prices = df["Close"].values.astype(float)

dates = pd.to_datetime(
    df["Date"]
).values


# =========================================================
# 10. CALCULATE DAILY LOG RETURNS
# =========================================================

log_returns = np.diff(
    np.log(prices)
)

return_dates = dates[1:]


print("\n" + "=" * 70)
print("SLIDING WINDOW CONFIGURATION")
print("=" * 70)

print(f"Window Size : {WINDOW_SIZE} days")
print(f"Lookback    : {LOOKBACK} days")
print("Target      : Next-day log return")
print("Prediction  : Walk-forward")


# =========================================================
# 11. CREATE LAGGED SAMPLES
# =========================================================

# Each row contains:
#
# [return day 1, return day 2, ..., return day 30, target]
#
# First 30 = features
# Last 1 = target

all_lags = sliding_window_view(
    log_returns,
    window_shape=LOOKBACK + 1
)


# =========================================================
# 12. NUMBER OF WALK-FORWARD PREDICTIONS
# =========================================================

# We need:
#
# 308 returns inside the current window
# Then predict the next return

total_predictions = len(log_returns) - WINDOW_SIZE


if total_predictions <= 0:

    raise ValueError(
        "Not enough return observations for the selected window."
    )


print(f"\nTotal predictions: {total_predictions}")


# =========================================================
# 13. DEVICE
# =========================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Using device: {device}")


# =========================================================
# 14. CREATE MODEL
# =========================================================

model = LoRAModel(
    lookback=LOOKBACK,
    config=lora_config
).to(device)


criterion = nn.MSELoss()


# =========================================================
# 15. PREPARE INITIAL 308-DAY WINDOW
# =========================================================

# Number of training samples inside each 308-day window
#
# Example:
# 308 returns
# lookback = 30
#
# Samples:
# 308 - 30 = 278

samples_per_window = WINDOW_SIZE - LOOKBACK


initial_window = all_lags[
    0:samples_per_window
]


X_init = torch.tensor(
    initial_window[:, :-1],
    dtype=torch.float32
).to(device)


y_init = torch.tensor(
    initial_window[:, -1],
    dtype=torch.float32
).unsqueeze(1).to(device)


# =========================================================
# 16. PRE-TRAIN BASE MODEL
# =========================================================

print("\n" + "=" * 70)
print("PRE-TRAINING BASE MODEL")
print("=" * 70)


pretrain_optimizer = optim.Adam(
    model.parameters(),
    lr=PRETRAIN_LEARNING_RATE
)


model.train()


for epoch in range(PRETRAIN_EPOCHS):

    pretrain_optimizer.zero_grad()

    output = model(X_init)

    loss = criterion(
        output,
        y_init
    )

    loss.backward()

    pretrain_optimizer.step()


    if (
        (epoch + 1) % 20 == 0
        or epoch == PRETRAIN_EPOCHS - 1
    ):

        print(
            f"Pre-training Epoch "
            f"{epoch + 1}/{PRETRAIN_EPOCHS} "
            f"| Loss: {loss.item():.8f}"
        )


# =========================================================
# 17. FREEZE BASE WEIGHTS
# =========================================================

print("\nFreezing base model weights...")

model.freeze_base()


# =========================================================
# 18. CREATE OUTPUT ARRAYS
# =========================================================

predicted_prices = np.zeros(
    total_predictions
)

actual_prices = np.zeros(
    total_predictions
)

prediction_dates = []


# =========================================================
# 19. WALK-FORWARD 308-DAY SLIDING WINDOW
# =========================================================

print("\n" + "=" * 70)
print("STARTING LORA WALK-FORWARD PREDICTION")
print("=" * 70)


for start in range(total_predictions):

    # -----------------------------------------------------
    # CURRENT 308-DAY TRAINING WINDOW
    # -----------------------------------------------------

    window_samples = all_lags[
        start:start + samples_per_window
    ]


    # First 30 values = input features
    X_train = torch.tensor(
        window_samples[:, :-1],
        dtype=torch.float32
    ).to(device)


    # Next return = target
    y_train = torch.tensor(
        window_samples[:, -1],
        dtype=torch.float32
    ).unsqueeze(1).to(device)


    # -----------------------------------------------------
    # TEST INPUT
    # -----------------------------------------------------
    #
    # Use the LAST 30 returns of the 308-day window
    # to predict the NEXT day's return.
    #
    # No future information is used.

    test_start = start + WINDOW_SIZE - LOOKBACK

    test_end = start + WINDOW_SIZE


    X_test_values = log_returns[
        test_start:test_end
    ]


    X_test = torch.tensor(
        X_test_values,
        dtype=torch.float32
    ).unsqueeze(0).to(device)


    # -----------------------------------------------------
    # FINE-TUNE ONLY LORA PARAMETERS
    # -----------------------------------------------------

    finetune_optimizer = optim.Adam(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=lora_config.learning_rate
    )


    model.train()


    for epoch in range(lora_config.epochs):

        finetune_optimizer.zero_grad()

        predictions = model(X_train)

        loss = criterion(
            predictions,
            y_train
        )

        loss.backward()

        finetune_optimizer.step()


    # -----------------------------------------------------
    # PREDICT NEXT DAY LOG RETURN
    # -----------------------------------------------------

    model.eval()


    with torch.no_grad():

        predicted_log_return = model(
            X_test
        ).item()


    # -----------------------------------------------------
    # CONVERT RETURN TO PRICE
    # -----------------------------------------------------
    #
    # Previous actual price:
    # P(t)
    #
    # Predicted:
    # P(t+1) = P(t) * exp(predicted return)
    #

    previous_actual_price = prices[
        start + WINDOW_SIZE
    ]


    predicted_price = (
        previous_actual_price
        * np.exp(predicted_log_return)
    )


    actual_price = prices[
        start + WINDOW_SIZE + 1
    ]


    predicted_date = dates[
        start + WINDOW_SIZE + 1
    ]


    # -----------------------------------------------------
    # SAVE RESULT
    # -----------------------------------------------------

    predicted_prices[start] = predicted_price

    actual_prices[start] = actual_price

    prediction_dates.append(
        predicted_date
    )


    # -----------------------------------------------------
    # PROGRESS
    # -----------------------------------------------------

    if (
        (start + 1) % 100 == 0
        or start == total_predictions - 1
    ):

        print(
            f"Completed "
            f"{start + 1}/{total_predictions} "
            f"| Date: "
            f"{pd.Timestamp(predicted_date).date()}"
        )


# =========================================================
# 20. CONVERT DATES
# =========================================================

prediction_dates = pd.to_datetime(
    prediction_dates
)


# =========================================================
# 21. CALCULATE METRICS
# =========================================================

mae = mean_absolute_error(
    actual_prices,
    predicted_prices
)


rmse = np.sqrt(
    mean_squared_error(
        actual_prices,
        predicted_prices
    )
)


mape = np.mean(
    np.abs(
        (actual_prices - predicted_prices)
        / actual_prices
    )
) * 100


absolute_errors = np.abs(
    actual_prices - predicted_prices
)


print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(f"MAE   : {mae:.2f}")
print(f"RMSE  : {rmse:.2f}")
print(f"MAPE  : {mape:.4f}%")


# =========================================================
# 22. SAVE PREDICTIONS
# =========================================================

results_df = pd.DataFrame({

    "Date": prediction_dates,

    "Actual_Price": actual_prices,

    "Predicted_Price": predicted_prices,

    "Absolute_Error": absolute_errors

})


results_df.to_csv(
    CSV_FILE,
    index=False
)


print(f"\nPredictions saved: {CSV_FILE}")


# =========================================================
# 23. CREATE INTERACTIVE GRAPH
# =========================================================

fig = make_subplots(

    rows=2,
    cols=1,

    shared_xaxes=True,

    vertical_spacing=0.10,

    row_heights=[
        0.70,
        0.30
    ],

    subplot_titles=(

        "Actual vs LoRA Fine-Tuned Predicted NIFTY50 Price",

        "Absolute Prediction Error (Index Points)"
    )
)


# =========================================================
# ACTUAL PRICE
# =========================================================

fig.add_trace(

    go.Scattergl(

        x=prediction_dates,

        y=actual_prices,

        mode="lines",

        name="Actual Price",

        line=dict(
            width=1.5,
            color="#1f77b4"
        ),

        hovertemplate=(
            "<b>Date:</b> %{x|%d-%b-%Y}"
            "<br>"
            "<b>Actual Price:</b> %{y:.2f}"
            "<extra></extra>"
        )

    ),

    row=1,
    col=1
)


# =========================================================
# PREDICTED PRICE
# =========================================================

fig.add_trace(

    go.Scattergl(

        x=prediction_dates,

        y=predicted_prices,

        mode="lines",

        name="LoRA Predicted Price",

        line=dict(
            width=1.5,
            dash="dash",
            color="#ff7f0e"
        ),

        hovertemplate=(
            "<b>Date:</b> %{x|%d-%b-%Y}"
            "<br>"
            "<b>Predicted Price:</b> %{y:.2f}"
            "<extra></extra>"
        )

    ),

    row=1,
    col=1
)


# =========================================================
# ABSOLUTE ERROR
# =========================================================

fig.add_trace(

    go.Scattergl(

        x=prediction_dates,

        y=absolute_errors,

        mode="lines",

        name="Absolute Error",

        line=dict(
            width=1,
            color="#d62728"
        ),

        fill="tozeroy",

        hovertemplate=(
            "<b>Date:</b> %{x|%d-%b-%Y}"
            "<br>"
            "<b>Absolute Error:</b> %{y:.2f} points"
            "<extra></extra>"
        )

    ),

    row=2,
    col=1
)


# =========================================================
# 24. GRAPH LAYOUT
# =========================================================

fig.update_layout(

    title=dict(

        text=(

            "<b>NIFTY50 LoRA Fine-Tuned "
            "308-Day Sliding Window Analysis</b><br>"

            f"<sup>"
            f"Window: {WINDOW_SIZE} Days | "
            f"Lookback: {LOOKBACK} Days | "
            f"LoRA Rank: {lora_config.r} | "
            f"Alpha: {lora_config.alpha} | "
            f"MAE: {mae:.2f} | "
            f"RMSE: {rmse:.2f} | "
            f"MAPE: {mape:.2f}%"
            f"</sup>"
        ),

        x=0.02,

        xanchor="left"
    ),

    template="plotly_white",

    height=850,

    hovermode="x unified"
)


fig.update_yaxes(

    title_text="NIFTY50 Close Price",

    row=1,

    col=1
)


fig.update_yaxes(

    title_text="Absolute Error",

    row=2,

    col=1
)


fig.update_xaxes(

    title_text="Date",

    row=2,

    col=1
)


# =========================================================
# 25. SAVE HTML GRAPH
# =========================================================

fig.write_html(

    HTML_FILE,

    include_plotlyjs="cdn",

    full_html=True
)


print("\n" + "=" * 70)
print("DONE SUCCESSFULLY")
print("=" * 70)

print(f"Graph HTML     : {HTML_FILE}")
print(f"Predictions CSV: {CSV_FILE}")


# =========================================================
# 26. SHOW GRAPH
# =========================================================

try:

    fig.show()

except Exception:

    print(
        "\nGraph preview could not open automatically."
    )

    print(
        f"Open this file in your browser:\n{HTML_FILE}"
    )
