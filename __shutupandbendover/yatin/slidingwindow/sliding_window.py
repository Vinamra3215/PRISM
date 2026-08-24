import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =========================================================
# CONFIGURATION
# =========================================================

DATA_FILE = "NIFTY50_2022_to_today.parquet"

WINDOW_SIZE = 308       # Total sliding training window
LOOKBACK = 30           # Previous 30 days used to predict next day
RIDGE_ALPHA = 1.0

CSV_FILE = "nifty_308day_predictions_normalized.csv"
HTML_FILE = "nifty_308day_sliding_window_normalized.html"


# =========================================================
# 1. LOAD DATA
# =========================================================

print("=" * 70)
print("LOADING NIFTY50 DATA")
print("=" * 70)

df = pd.read_parquet(DATA_FILE)

print("\nOriginal columns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


# =========================================================
# 2. CLEAN DATA
# Handles both:
# A. Proper Date column
# B. Your earlier format where first column contains Date
# =========================================================

if "Date" in df.columns:

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

else:

    print("\nNo direct Date column found.")
    print("Trying to recover Date from first column...")

    first_col = df.columns[0]

    temp_dates = pd.to_datetime(
        df[first_col],
        errors="coerce"
    )

    # If first column contains dates, use it as Date
    if temp_dates.notna().sum() > 0:
        df["Date"] = temp_dates
    else:
        raise ValueError(
            f"Could not find Date information. "
            f"Available columns: {df.columns.tolist()}"
        )


# =========================================================
# 3. FIND CLOSE PRICE COLUMN
# =========================================================

if "Close" in df.columns:
    close_col = "Close"

elif "Adj Close" in df.columns:
    close_col = "Adj Close"

else:
    raise ValueError(
        "Close price column not found. "
        f"Available columns: {df.columns.tolist()}"
    )


# Convert Close to numeric
df[close_col] = pd.to_numeric(
    df[close_col],
    errors="coerce"
)


# Keep valid rows only
df = df[["Date", close_col]].copy()

df = df.dropna()

df = df.sort_values("Date")

df = df.drop_duplicates(subset=["Date"])

df = df.reset_index(drop=True)


# Rename for easier use
df.rename(
    columns={close_col: "Close"},
    inplace=True
)


print("\n" + "=" * 70)
print("CLEANED DATA")
print("=" * 70)

print(f"First date : {df['Date'].iloc[0].date()}")
print(f"Last date  : {df['Date'].iloc[-1].date()}")
print(f"Total rows : {len(df)}")

print("\nFirst rows:")
print(df.head())

print("\nLast rows:")
print(df.tail())


# =========================================================
# 4. CHECK ENOUGH DATA
# =========================================================

if len(df) <= WINDOW_SIZE:
    raise ValueError(
        f"Not enough data. Need more than {WINDOW_SIZE} rows, "
        f"but only found {len(df)} rows."
    )


# =========================================================
# 5. CONVERT PRICE TO NUMPY
# =========================================================

prices = df["Close"].values.astype(float)
dates = pd.to_datetime(df["Date"]).values


# =========================================================
# 6. SLIDING WINDOW PREDICTION
# =========================================================

print("\n" + "=" * 70)
print("STARTING 308-DAY SLIDING WINDOW PREDICTION")
print("=" * 70)

print(f"Window size : {WINDOW_SIZE} trading days")
print(f"Lookback    : {LOOKBACK} trading days")
print(f"Model       : Ridge Regression")
print(f"Alpha       : {RIDGE_ALPHA}")

# First prediction happens after first 308 data points
total_predictions = len(prices) - WINDOW_SIZE

print(f"Total predictions to make: {total_predictions}")
print()


actual_prices = []
predicted_prices = []
prediction_dates = []


for start in range(total_predictions):

    # -----------------------------------------------------
    # Example:
    #
    # start = 0
    # Train = Day 1 to Day 308
    # Predict = Day 309
    #
    # start = 1
    # Train = Day 2 to Day 309
    # Predict = Day 310
    #
    # start = 2
    # Train = Day 3 to Day 310
    # Predict = Day 311
    # -----------------------------------------------------

    end = start + WINDOW_SIZE

    window_prices = prices[start:end]

    target_price = prices[end]
    target_date = dates[end]


    # =====================================================
    # NORMALIZE ONLY THE CURRENT TRAINING WINDOW
    #
    # Important:
    # The model does NOT see the target day's actual price.
    # The scaler is fitted only on the 308-day training window.
    # =====================================================

    window_scaler = MinMaxScaler(
        feature_range=(0, 1)
    )

    window_scaled = window_scaler.fit_transform(
        window_prices.reshape(-1, 1)
    ).flatten()


    # =====================================================
    # CREATE TRAINING SAMPLES
    #
    # Previous 30 normalized prices -> next normalized price
    #
    # Example:
    # Days 1-30  -> Day 31
    # Days 2-31  -> Day 32
    # ...
    # =====================================================

    X_train = []
    y_train = []

    for i in range(WINDOW_SIZE - LOOKBACK):

        X_train.append(
            window_scaled[i:i + LOOKBACK]
        )

        y_train.append(
            window_scaled[i + LOOKBACK]
        )


    X_train = np.array(X_train)
    y_train = np.array(y_train)


    # =====================================================
    # CREATE A FRESH MODEL FOR THIS WINDOW
    #
    # This avoids carrying learned weights from the previous
    # sliding window.
    # =====================================================

    model = Ridge(
        alpha=RIDGE_ALPHA
    )

    model.fit(
        X_train,
        y_train
    )


    # =====================================================
    # PREDICT NEXT DAY
    #
    # Input:
    # Last 30 days from the current 308-day window
    # =====================================================

    last_30_days = window_scaled[-LOOKBACK:].reshape(1, -1)

    predicted_scaled = model.predict(
        last_30_days
    )[0]


    # =====================================================
    # CONVERT PREDICTION BACK TO ORIGINAL PRICE
    # =====================================================

    predicted_price = window_scaler.inverse_transform(
        [[predicted_scaled]]
    )[0][0]


    # =====================================================
    # SAVE RESULT
    # =====================================================

    actual_prices.append(target_price)

    predicted_prices.append(predicted_price)

    prediction_dates.append(target_date)


    # Progress
    if (start + 1) % 100 == 0 or start == total_predictions - 1:

        print(
            f"Completed {start + 1}/{total_predictions} predictions "
            f"| Date: {pd.Timestamp(target_date).date()}"
        )


# =========================================================
# 7. CONVERT RESULTS TO NUMPY
# =========================================================

actual_prices = np.array(actual_prices)
predicted_prices = np.array(predicted_prices)

prediction_dates = pd.to_datetime(prediction_dates)


# =========================================================
# 8. CALCULATE METRICS ON ORIGINAL PRICE
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


print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(f"MAE  : {mae:.2f} points")
print(f"RMSE : {rmse:.2f} points")
print(f"MAPE : {mape:.2f}%")

print(f"\nPrediction start date: {prediction_dates[0].date()}")
print(f"Prediction end date  : {prediction_dates[-1].date()}")


# =========================================================
# 9. NORMALIZE PRICES FOR GRAPH
# =========================================================
#
# IMPORTANT:
#
# We already normalized each individual 308-day training
# window for the model.
#
# But each window has a DIFFERENT scaler.
# Therefore those normalized values cannot directly be
# compared on one complete graph.
#
# So here we create ONE normalization only for visualization.
#
# Original:
# Actual:    17000, 22000, 26000
#
# Graph:
# Actual:    0.00, 0.56, 1.00
#
# The model evaluation metrics above remain in ORIGINAL INR.
# =========================================================

plot_scaler = MinMaxScaler(
    feature_range=(0, 1)
)

# Fit using actual evaluation-period prices
plot_scaler.fit(
    actual_prices.reshape(-1, 1)
)


actual_normalized = plot_scaler.transform(
    actual_prices.reshape(-1, 1)
).flatten()


predicted_normalized = plot_scaler.transform(
    predicted_prices.reshape(-1, 1)
).flatten()


# =========================================================
# 10. ABSOLUTE ERROR
#
# Keep this in ORIGINAL PRICE POINTS so we can understand
# the real prediction error.
# =========================================================

absolute_error = np.abs(
    actual_prices - predicted_prices
)


# =========================================================
# 11. SAVE PREDICTIONS
# =========================================================

results_df = pd.DataFrame({

    "Date": prediction_dates,

    # Original prices
    "Actual_Price": actual_prices,
    "Predicted_Price": predicted_prices,

    # Normalized prices used in graph
    "Actual_Normalized": actual_normalized,
    "Predicted_Normalized": predicted_normalized,

    # Error in original NIFTY points
    "Absolute_Error": absolute_error
})


results_df.to_csv(
    CSV_FILE,
    index=False
)


print(f"\nPredictions saved to: {CSV_FILE}")


# =========================================================
# 12. CREATE PLOTLY GRAPH
# =========================================================

fig = make_subplots(

    rows=2,
    cols=1,

    shared_xaxes=True,

    vertical_spacing=0.10,

    row_heights=[0.65, 0.35],

    subplot_titles=(
        "Actual vs Predicted NIFTY50 Normalized Close Price",
        "Absolute Prediction Error (Original NIFTY Points)"
    )
)


# =========================================================
# TOP GRAPH: NORMALIZED ACTUAL PRICE
# =========================================================

fig.add_trace(

    go.Scatter(

        x=prediction_dates,

        y=actual_normalized,

        mode="lines",

        name="Actual Price (Normalized)",

        line=dict(
            width=2
        ),

        hovertemplate=
        "<b>Date:</b> %{x|%d-%b-%Y}<br>"
        "<b>Actual Normalized:</b> %{y:.4f}<br>"
        "<extra></extra>"
    ),

    row=1,
    col=1
)


# =========================================================
# TOP GRAPH: NORMALIZED PREDICTED PRICE
# =========================================================

fig.add_trace(

    go.Scatter(

        x=prediction_dates,

        y=predicted_normalized,

        mode="lines",

        name="Predicted Price (Normalized)",

        line=dict(
            width=2,
            dash="dash"
        ),

        hovertemplate=
        "<b>Date:</b> %{x|%d-%b-%Y}<br>"
        "<b>Predicted Normalized:</b> %{y:.4f}<br>"
        "<extra></extra>"
    ),

    row=1,
    col=1
)


# =========================================================
# BOTTOM GRAPH: ORIGINAL ABSOLUTE ERROR
# =========================================================

fig.add_trace(

    go.Scatter(

        x=prediction_dates,

        y=absolute_error,

        mode="lines",

        name="Absolute Error",

        line=dict(
            width=1
        ),

        fill="tozeroy",

        hovertemplate=
        "<b>Date:</b> %{x|%d-%b-%Y}<br>"
        "<b>Absolute Error:</b> %{y:.2f} points<br>"
        "<extra></extra>"
    ),

    row=2,
    col=1
)


# =========================================================
# 13. GRAPH LAYOUT
# =========================================================

fig.update_layout(

    title=dict(

        text=(
            "<b>NIFTY50 308-Day Sliding Window Baseline "
            "(Normalized Price)</b><br>"

            f"<sup>"
            f"Model: Ridge Regression | "
            f"Window: {WINDOW_SIZE} Days | "
            f"Lookback: {LOOKBACK} Days | "
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

    hovermode="x unified",

    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.03,
        xanchor="right",
        x=1
    ),

    margin=dict(
        l=70,
        r=40,
        t=100,
        b=60
    )
)


# =========================================================
# 14. AXIS SETTINGS
# =========================================================

fig.update_yaxes(

    title_text="Normalized Price (0 to 1)",

    range=[0, 1.05],

    tickformat=".1f",

    row=1,
    col=1
)


fig.update_yaxes(

    title_text="Absolute Error (Points)",

    row=2,
    col=1
)


fig.update_xaxes(

    title_text="Date",

    rangeslider_visible=True,

    rangeselector=dict(

        buttons=[

            dict(
                count=1,
                label="1M",
                step="month",
                stepmode="backward"
            ),

            dict(
                count=3,
                label="3M",
                step="month",
                stepmode="backward"
            ),

            dict(
                count=6,
                label="6M",
                step="month",
                stepmode="backward"
            ),

            dict(
                count=1,
                label="1Y",
                step="year",
                stepmode="backward"
            ),

            dict(
                step="all",
                label="All"
            )
        ]
    ),

    row=2,
    col=1
)


# =========================================================
# 15. SAVE INTERACTIVE HTML
# =========================================================

fig.write_html(
    HTML_FILE,
    include_plotlyjs=True
)


print(f"Interactive graph saved to: {HTML_FILE}")


# =========================================================
# 16. SHOW GRAPH
# =========================================================

fig.show()


print("\n" + "=" * 70)
print("FILES SAVED SUCCESSFULLY")
print("=" * 70)

print(f"Graph       : {HTML_FILE}")
print(f"Predictions : {CSV_FILE}")