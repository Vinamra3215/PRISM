import os
import pandas as pd

# ==========================
# Folder Paths
# ==========================

INPUT_FOLDER = "data/raw_data"
OUTPUT_FOLDER = "data/parquet_data"

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================
# Convert Every CSV
# ==========================

files = os.listdir(INPUT_FOLDER)

for file in files:

    if file.endswith(".csv"):

        input_path = os.path.join(INPUT_FOLDER, file)

        output_name = file.replace(".csv", ".parquet")
        output_path = os.path.join(OUTPUT_FOLDER, output_name)

        print("----------------------------------")
        print("Reading :", file)

        df = pd.read_csv(input_path)

        print("Rows :", len(df))
        print("Columns :", len(df.columns))

        df.to_parquet(output_path, index=False)

        print("Saved :", output_name)

print("----------------------------------")
print("All files converted successfully!")
