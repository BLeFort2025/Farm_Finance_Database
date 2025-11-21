import pandas as pd
from pathlib import Path

DATA_LATEST = Path("data/latest")

def standardize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    if "REF_DATE" in df.columns:
        df["YEAR"] = df["REF_DATE"].astype(str).str.slice(0,4).astype(int)
    if "GEO" in df.columns:
        df["GEO"] = df["GEO"].astype(str).str.strip()
    if "VALUE" in df.columns:
        df["VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce")
    return df

def load(table_id: str) -> pd.DataFrame:
    return pd.read_csv(DATA_LATEST / f"{table_id}.csv", low_memory=False)

def compute_ceag_profitability():
    # CEAG Revenues + Expenses => NOI, Margin
    rev = standardize(load("32-10-0240-01"))
    exp = standardize(load("32-10-0241-01"))
    keys = ["GEO","YEAR"]
    merged = (rev.assign(REV=rev["VALUE"])[keys+["REV"]]
                .merge(exp.assign(EXP=exp["VALUE"])[keys+["EXP"]], on=keys, how="inner"))
    merged["NET_OP_INCOME"] = merged["REV"] - merged["EXP"]
    merged["PROFIT_MARGIN"] = merged["NET_OP_INCOME"] / merged["REV"]
    out = merged[keys + ["REV","EXP","NET_OP_INCOME","PROFIT_MARGIN"]]
    out.to_csv(DATA_LATEST / "ceag_profitability_cd.csv", index=False)

def compute_ceag_direct_sales_share():
    # Direct sales as share of operating revenues (same GEO,YEAR)
    try:
        ds = standardize(load("32-10-0242-01"))
        rev = standardize(load("32-10-0240-01"))
    except FileNotFoundError:
        return
    keys = ["GEO","YEAR"]
    merged = (ds.assign(DIRECT_SALES=ds["VALUE"])[keys+["DIRECT_SALES"]]
                .merge(rev.assign(REV=rev["VALUE"])[keys+["REV"]], on=keys, how="inner"))
    merged["DIRECT_SALES_SHARE"] = merged["DIRECT_SALES"] / merged["REV"]
    merged[keys + ["DIRECT_SALES","REV","DIRECT_SALES_SHARE"]].to_csv(
        DATA_LATEST / "ceag_direct_sales_share_cd.csv", index=False
    )

def run():
    # Run all transformations that depend on fetched sources
    try:
        compute_ceag_profitability()
        compute_ceag_direct_sales_share()
        print("Derived CEAG metrics written.")
    except FileNotFoundError:
        print("CEAG source files not found yet; run fetch first.")

if __name__ == "__main__":
    run()
