import os
import time
import requests
import pandas as pd
import numpy as np

# ============================================================
# SETTINGS
# ============================================================

TICKER = "HCLTECH.NS"

START_DATE = "2018-01-01"
END_DATE = "2026-08-16"

OUTPUT_DIR = "/home/soq/yatin/kronos.run/hcltech_daily"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "HCLTECH_daily_API_2018_2026.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# DATE -> UNIX TIMESTAMP
# ============================================================

def date_to_unix(date_string):

    timestamp = pd.Timestamp(
        date_string,
        tz="Asia/Kolkata"
    )

    return int(timestamp.timestamp())


period1 = date_to_unix(START_DATE)
period2 = date_to_unix(END_DATE)


# ============================================================
# YAHOO FINANCE CHART API
# ============================================================

url = (
    f"https://query1.finance.yahoo.com/v8/finance/chart/"
    f"{TICKER}"
)

params = {

    "period1": period1,

    "period2": period2,

    "interval": "1d",

    "events": "history",

    "includeAdjustedClose": "true"
}


headers = {

    "User-Agent":
        "Mozilla/5.0 "
        "(X11; Linux x86_64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/120 Safari/537.36"
}


# ============================================================
# DOWNLOAD WITH RETRIES
# ============================================================

print("=" * 80)
print("HCLTECH DAILY DATA - YAHOO FINANCE API")
print("=" * 80)

print()
print("Ticker :", TICKER)
print("Start  :", START_DATE)
print("End    :", END_DATE)
print("Output :", OUTPUT_FILE)

print()
print("Connecting to Yahoo Finance...")


data = None

for attempt in range(1, 6):

    try:

        print(
            f"Attempt {attempt}/5 ..."
        )

        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=60
        )

        response.raise_for_status()

        data = response.json()

        print(
            "Yahoo Finance response received."
        )

        break

    except Exception as e:

        print(
            f"Attempt {attempt} failed:"
        )

        print(e)

        if attempt < 5:

            print(
                "Waiting 5 seconds..."
            )

            time.sleep(5)


if data is None:

    raise RuntimeError(
        """
Yahoo Finance could not be reached after
5 attempts.

This is a network/connectivity problem,
not a data-processing problem.
"""
    )


# ============================================================
# CHECK RESPONSE
# ============================================================

chart = data.get(
    "chart",
    {}
)

error = chart.get(
    "error"
)

if error:

    raise RuntimeError(
        f"Yahoo Finance API error: {error}"
    )


results = chart.get(
    "result"
)

if not results:

    raise RuntimeError(
        "Yahoo Finance returned no result."
    )


result = results[0]


timestamps = result.get(
    "timestamp"
)

indicators = result.get(
    "indicators",
    {}
)


if timestamps is None:

    raise RuntimeError(
        "Yahoo Finance returned no timestamps."
    )


quote = indicators.get(
    "quote",
    [{}]
)[0]


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame({

    "timestamp": pd.to_datetime(
        timestamps,
        unit="s",
        utc=True
    ).tz_convert(
        "Asia/Kolkata"
    ).tz_localize(None),

    "open": quote.get("open"),

    "high": quote.get("high"),

    "low": quote.get("low"),

    "close": quote.get("close"),

    "volume": quote.get("volume")
})


# ============================================================
# CLEAN
# ============================================================

df = df.dropna(
    subset=[
        "open",
        "high",
        "low",
        "close"
    ]
)


df = df.sort_values(
    "timestamp"
).reset_index(drop=True)


# ============================================================
# DATE COLUMN
# ============================================================

df["date"] = (
    df["timestamp"]
    .dt.normalize()
)


# ============================================================
# REMOVE DUPLICATES
# ============================================================

df = df.drop_duplicates(
    subset=["date"]
)


# ============================================================
# FINAL COLUMNS
# ============================================================

df = df[
    [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]
]


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 80)
print("DOWNLOAD SUCCESSFUL")
print("=" * 80)

print()

print(
    "Rows:",
    len(df)
)

print(
    "Start:",
    df["date"].min()
)

print(
    "End:",
    df["date"].max()
)

print()

print(
    "First 5 rows:"
)

print(
    df.head()
)

print()

print(
    "Last 5 rows:"
)

print(
    df.tail()
)


# ============================================================
# CHECK 400 DAYS BEFORE 2022
# ============================================================

before_2022 = df[
    df["date"]
    < pd.Timestamp("2022-01-01")
]


print()
print("=" * 80)
print("400-DAY TRAINING CHECK")
print("=" * 80)

print(
    "Trading days before 2022-01-01:",
    len(before_2022)
)


if len(before_2022) >= 400:

    training_400 = before_2022.tail(
        400
    )

    print()
    print(
        "400-day training start:",
        training_400["date"].min()
    )

    print(
        "400-day training end:",
        training_400["date"].max()
    )

    training_400.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "HCLTECH_training_400_days.csv"
        ),
        index=False
    )

    print()
    print(
        "Saved:"
    )

    print(
        os.path.join(
            OUTPUT_DIR,
            "HCLTECH_training_400_days.csv"
        )
    )

else:

    print()
    print(
        "ERROR: Yahoo data still does not contain "
        "400 trading days before 2022."
    )


# ============================================================
# TEST DATA
# ============================================================

test = df[
    df["date"]
    >= pd.Timestamp("2022-01-01")
].copy()


print()
print("=" * 80)
print("PREDICTION / TEST PERIOD")
print("=" * 80)

print(
    "Rows:",
    len(test)
)

print(
    "Start:",
    test["date"].min()
)

print(
    "End:",
    test["date"].max()
)


test.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "HCLTECH_test_2022_to_latest.csv"
    ),
    index=False
)


print()
print("=" * 80)
print("FILES CREATED")
print("=" * 80)

print()

print(
    "1.",
    OUTPUT_FILE
)

print(
    "2.",
    os.path.join(
        OUTPUT_DIR,
        "HCLTECH_training_400_days.csv"
    )
)

print(
    "3.",
    os.path.join(
        OUTPUT_DIR,
        "HCLTECH_test_2022_to_latest.csv"
    )
)

print()
print("=" * 80)

