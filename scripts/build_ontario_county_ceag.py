import pandas as pd
import numpy as np
from pathlib import Path

# Path to the source Excel file
EXCEL_PATH = Path(
    r"C:\Users\ben.lefort\OneDrive - Ontario Federation of Agriculture\Desktop\Ben Desktop Files\Economic Analyst Position\County level datasets\_AgCensus_Wide_Formatted.xlsx"
)

# Output path
OUTPUT_PATH = Path("data/latest/ontario_county_ceag.csv")


def main():
    if not EXCEL_PATH.exists():
        raise FileNotFoundError(
            f"Excel file not found at: {EXCEL_PATH}\n"
            "Update EXCEL_PATH in build_ontario_county_ceag.py if the file has moved."
        )

    # Read Excel
    df = pd.read_excel(EXCEL_PATH)

    # Standardize column names for YEAR and GEO
    rename_map = {}
    if "Municipality" in df.columns:
        rename_map["Municipality"] = "GEO"
    if "Year" in df.columns:
        rename_map["Year"] = "YEAR"

    df = df.rename(columns=rename_map)

    if "GEO" not in df.columns or "YEAR" not in df.columns:
        raise ValueError(
            "The Excel file must contain 'Municipality' and 'Year' columns. "
            f"Columns found: {list(df.columns)}"
        )

    # Clean whitespace
    obj_cols = df.select_dtypes(include="object").columns
    for col in obj_cols:
        df[col] = df[col].astype(str).str.strip()

    # Replace placeholder values
    df.replace({"": np.nan, "..": np.nan, "-": 0}, inplace=True)

    # Convert numerics
    for col in df.columns:
        if col not in ["YEAR", "GEO"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clean year column
    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")
    df = df.dropna(subset=["YEAR"]).copy()
    df["YEAR"] = df["YEAR"].astype(int)

    # Ensure GEO is string
    df["GEO"] = df["GEO"].astype(str)

    # Make sure output directory exists
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save
    df.to_csv(OUTPUT_PATH, index=False)
    print(
        f"Saved cleaned Ontario county CEAG dataset to:\n  {OUTPUT_PATH.resolve()}\n"
        f"Rows: {len(df):,}\nColumns: {len(df.columns)}"
    )


if __name__ == "__main__":
    main()
