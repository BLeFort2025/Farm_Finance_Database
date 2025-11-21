import re
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import yaml

# -------------------------
# Page + constants
# -------------------------
st.set_page_config(page_title="OFA Canada Farm Statistics Dashboard", layout="wide")
st.title("OFA Canada Farm Statistics Dashboard")
st.caption("Source: Statistics Canada and custom OFA datasets. Data may be subject to revision.")

DATA = Path("data/latest")
CONFIG_PATH = Path("config/tables.yml")

# Load table metadata once
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    raw_cfg = yaml.safe_load(f)
TABLE_CONFIG = raw_cfg.get("tables", raw_cfg)

# Consistent colours for Canada + provinces
GEO_COLOURS = {
    "Canada": "#D00000",
    "British Columbia": "#FF9800",
    "Alberta": "#0D3692",
    "Saskatchewan": "#8BB800",
    "Manitoba": "#A07F37",
    "Ontario": "#007A30",
    "Quebec": "#001F97",
    "New Brunswick": "#F4C600",
    "Nova Scotia": "#00AFEF",
    "Prince Edward Island": "#00968C",
    "Newfoundland and Labrador": "#C2185B",
    "Yukon": "#7E57C2",
    "Northwest Territories": "#006064",
    "Nunavut": "#FF7043",
}


@st.cache_data(ttl=7 * 24 * 3600)
def load_csv(name: str) -> pd.DataFrame:
    """Load a CSV from data/latest, or return empty DataFrame if missing."""
    fp = DATA / name
    if not fp.exists():
        return pd.DataFrame()

    df = pd.read_csv(fp, low_memory=False)

    # Ensure YEAR exists
    if "YEAR" not in df.columns:
        if "REF_DATE" in df.columns:
            df["YEAR"] = df["REF_DATE"].astype(str).str.slice(0, 4)
        else:
            df["YEAR"] = ""

    if "VALUE" in df.columns:
        df["VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce")

    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")
    return df


def get_unit_label(df: pd.DataFrame) -> str:
    """
    Build a compact unit label from UOM and SCALAR_FACTOR.

    Examples:
      - UOM='Dollars', SCALAR_FACTOR='thousands' -> 'Dollars (× 1,000)'
      - UOM='Dollars', SCALAR_FACTOR='millions'  -> 'Dollars (× 1,000,000)'
      - UOM='Index, 2016=100', SCALAR_FACTOR='units' -> 'Index, 2016=100'
    """
    uom = None
    scalar = None

    if "UOM" in df.columns and df["UOM"].notna().any():
        uom = df["UOM"].dropna().astype(str).mode().iloc[0]

    if "SCALAR_FACTOR" in df.columns and df["SCALAR_FACTOR"].notna().any():
        scalar = df["SCALAR_FACTOR"].dropna().astype(str).mode().iloc[0]

    if not uom and not scalar:
        return ""

    if scalar:
        scalar_lower = scalar.lower()
        scale_map = {
            "thousands": "× 1,000",
            "millions": "× 1,000,000",
            "billions": "× 1,000,000,000",
        }
        if scalar_lower == "units":
            return uom or ""

        if uom and scalar_lower in scale_map:
            return f"{uom} ({scale_map[scalar_lower]})"

        if uom:
            return f"{uom}, {scalar}"

        return scalar

    return uom or ""


def build_download_name(
    dataset_label: str,
    series_label: str | None,
    yr_min: int | None,
    yr_max: int | None,
    geos: list[str],
) -> str:
    """Create a friendly download filename for chart export."""
    parts: list[str] = [dataset_label]
    if series_label:
        parts.append(series_label)
    if yr_min is not None and yr_max is not None:
        parts.append(f"{yr_min}-{yr_max}")
    if geos:
        parts.append(", ".join(geos[:3]) + ("" if len(geos) <= 3 else ", …"))

    name = " — ".join(parts)
    name = re.sub(r"[\\/:*?\"<>|]+", "-", name)  # illegal path chars
    name = re.sub(r"\s+", " ", name).strip()
    return name


def altair_colour_scale(present_geos: list[str]) -> alt.Scale:
    """Colour scale mapping geos to fixed colours."""
    domain: list[str] = []
    rng: list[str] = []

    for g in present_geos:
        if g in GEO_COLOURS:
            domain.append(g)
            rng.append(GEO_COLOURS[g])

    for g in present_geos:
        if g not in domain:
            domain.append(g)

    if rng:
        return alt.Scale(domain=domain, range=rng)
    return alt.Scale(domain=domain)


def make_line_chart(long_df: pd.DataFrame, y_title: str, present_geos: list[str]) -> alt.Chart:
    """Altair line chart (YEAR, GEO, VALUE) for annual data."""
    chart_df = long_df.copy()
    chart_df["YEAR"] = chart_df["YEAR"].astype(int).astype(str)
    scale = altair_colour_scale(present_geos)

    chart = (
        alt.Chart(chart_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("YEAR:O", title="Year", axis=alt.Axis(format="d")),
            y=alt.Y("VALUE:Q", title=y_title, axis=alt.Axis(format="~s")),
            color=alt.Color("GEO:N", title="Geography", scale=scale),
            tooltip=[
                alt.Tooltip("YEAR:O", title="Year", format="d"),
                alt.Tooltip("GEO:N", title="Geography"),
                alt.Tooltip("VALUE:Q", title="Value", format=","),
            ],
        )
        .properties(height=420)
        .interactive()
    )
    return chart


def find_column_fuzzy(df: pd.DataFrame, search_terms: list[str]) -> str | None:
    """Return first column whose name contains any of the given search terms (case-insensitive)."""
    cols = list(df.columns)
    lower_cols = [c.lower() for c in cols]
    for term in search_terms:
        t = term.lower()
        for col, col_lower in zip(cols, lower_cols):
            if t in col_lower:
                return col
    return None


def find_series_column(table_id: str, df: pd.DataFrame) -> str | None:
    """Figure out which column should be treated as the 'series' selector."""

    # ---------- Farm finance core tables ----------

    # Farm cash receipts
    if table_id == "32-10-0045-01" and "Type of cash receipts" in df.columns:
        return "Type of cash receipts"

    # Farm operating revenues & expenses (ATDP)
    if table_id == "32-10-0136-01":
        if "Estimates" in df.columns:
            return "Estimates"
        if "Revenues and expenses" in df.columns:
            return "Revenues and expenses"

    # Farm operating expenses & depreciation charges
    if table_id == "32-10-0049-01" and "Expenses and rebates" in df.columns:
        return "Expenses and rebates"

    # Farm capital (value of assets)
    if table_id == "32-10-0050-01" and "Farm items" in df.columns:
        return "Farm items"

    # Farm debt outstanding
    if table_id == "32-10-0051-01" and "Type of lender" in df.columns:
        return "Type of lender"

    # Balance sheet of the agricultural sector
    if table_id == "32-10-0056-01":
        if "Balance sheet item" in df.columns:
            return "Balance sheet item"
        if "Commodities" in df.columns:
            return "Commodities"

    # Net farm income, by components
    if table_id == "32-10-0052-01":
        for c in ["Net farm income components", "Component"]:
            if c in df.columns:
                return c

    # Direct program payments to producers
    if table_id == "32-10-0106-01":
        for c in ["Program", "Program type"]:
            if c in df.columns:
                return c

    # Agriculture value added account
    if table_id == "32-10-0048-01":
        for c in ["Value added account", "Value added component"]:
            if c in df.columns:
                return c

    # Farm income in kind
    if table_id == "32-10-0055-01":
        for c in [
            "Farm income in kind items",
            "Farm income in kind, by item",
            "Farm income in kind",
        ]:
            if c in df.columns:
                return c

    # ---------- Costs & Inflation tables ----------

    # Farm input price index (FIPI)
    if table_id == "18-10-0258-01":
        for c in ["Price index", "Farm input category", "Inputs"]:
            if c in df.columns:
                return c

    # Machinery & equipment price index (MEPI)
    if table_id == "18-10-0270-01" and "Industry of purchase" in df.columns:
        return "Industry of purchase"

    # Food CPI and related CPI tables
    if table_id.startswith("18-10-0004") or table_id in {"18-10-0001-01", "18-10-0002-01"}:
        if "Products and product groups" in df.columns:
            return "Products and product groups"

    # For-hire motor carrier freight SPI
    if table_id == "18-10-0281-01" and "North American Industry Classification System (NAICS)" in df.columns:
        return "North American Industry Classification System (NAICS)"

    # Freight Rail Services Price Index
    if table_id == "18-10-0212-01" and "Commodity group" in df.columns:
        return "Commodity group"

    # ---------- Transport / trade tables ----------

    # Trucking financial statistics
    if table_id == "23-10-0291-01" and "Financial statistics" in df.columns:
        return "Financial statistics"

    # CIMT by mode of transport
    if table_id == "12-10-0177-01" and "Mode of transport" in df.columns:
        return "Mode of transport"

    # Railway carloadings
    if table_id == "23-10-0216-02":
        for c in ["Estimates", "Railway carloadings components"]:
            if c in df.columns:
                return c

    # CFAF origin–destination
    if table_id == "23-10-0142-01" and "Mode of transportation" in df.columns:
        return "Mode of transportation"

    # Transport activity / supply chain indicators
    if table_id == "23-10-0269-01" and "Activity indicators" in df.columns:
        return "Activity indicators"
    if table_id == "23-10-0271-01" and "Performance indicators" in df.columns:
        return "Performance indicators"

    # ---------- Generic fallback ----------

    candidates = [
        "Type of cash receipts",
        "Farm items",
        "Expenses and rebates",
        "Commodities",
        "Balance sheet item",
        "Component",
        "Program",
        "Program type",
        "Farm type",
        "Revenues and expenses",
        "Estimates",
        "Price index",
        "Products and product groups",
        "Industry of purchase",
        "Value added account",
        "Farm income in kind items",
        "Mode of transport",
        "Mode of transportation",
        "North American Product Classification System (NAPCS)",
        "North American Industry Classification System (NAICS)",
        "Commodity group",
        "Financial statistics",
        "Activity indicators",
        "Performance indicators",
    ]
    for c in candidates:
        if c in df.columns:
            return c

    return None


# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Filters")
level = st.sidebar.radio(
    "Geography / View",
    ["Farm finance statistics", "County Level Census Stats (ON)", "Costs & Inflation", "Transportation & Exports"],
    horizontal=False,
)

# ============================================================
# ONTARIO COUNTY / SUB-PROVINCIAL VIEW
# ============================================================
if level == "County Level Census Stats (ON)":
    st.subheader("County Level Census Stats (Ontario only)")
    st.caption(
        "Ontario-only county / census division data from custom Census of Agriculture dataset."
    )

    df = load_csv("ontario_county_ceag.csv")
    if df.empty:
        st.info(
            "Prepare the Ontario county dataset first: "
            "`python -m scripts.build_ontario_county_ceag`"
        )
        st.stop()

    if "YEAR" not in df.columns or "GEO" not in df.columns:
        st.error("Ontario county dataset must contain YEAR and GEO columns.")
        st.stop()

    df = df.dropna(subset=["YEAR"]).copy()
    df["YEAR"] = df["YEAR"].astype(int)
    df["GEO"] = df["GEO"].astype(str)

    counties = sorted(df["GEO"].unique().tolist())
    default_counties = counties[:3] if len(counties) > 3 else counties
    county_pick = st.sidebar.multiselect(
        "County / Census division (Ontario only)", counties, default=default_counties
    )
    if not county_pick:
        st.info("Select at least one county.")
        st.stop()

    yrs = df[df["GEO"].isin(county_pick)]["YEAR"]
    yr_min, yr_max = int(yrs.min()), int(yrs.max())
    if yr_min < yr_max:
        yr_from, yr_to = st.sidebar.slider(
            "Year range", min_value=yr_min, max_value=yr_max, value=(yr_min, yr_max)
        )
    else:
        yr_from, yr_to = yr_min, yr_max

    f = df[(df["GEO"].isin(county_pick)) & (df["YEAR"].between(yr_from, yr_to))].copy()
    if f.empty:
        st.info("No data for selected filters.")
        st.stop()

    exclude_cols = {
        "GEO",
        "YEAR",
        "REF_DATE",
        "DGUID",
        "UOM",
        "UOM_ID",
        "SCALAR_FACTOR",
        "SCALAR_ID",
        "VECTOR",
        "COORDINATE",
        "STATUS",
        "SYMBOL",
        "TERMINATED",
        "DECIMALS",
    }
    numeric_cols = [
        c
        for c in f.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(f[c])
    ]
    if not numeric_cols:
        st.error("No numeric metric columns found in Ontario county dataset.")
        st.stop()

    metric = st.sidebar.selectbox("Metric", numeric_cols, index=0)

    latest_year = f["YEAR"].max()
    latest_slice = f[f["YEAR"] == latest_year]
    prev_slice = f[f["YEAR"] < latest_year]

    total_latest = latest_slice[metric].sum()
    if not prev_slice.empty:
        prev_year = prev_slice["YEAR"].max()
        prev_total = prev_slice[prev_slice["YEAR"] == prev_year][metric].sum()
        change = (total_latest - prev_total) / prev_total * 100 if prev_total else 0.0
        st.metric(
            f"{metric} in {latest_year}",
            f"{total_latest:,.0f}",
            f"{change:+.1f}% vs {int(prev_year)}",
        )
    else:
        st.metric(f"{metric} in {latest_year}", f"{total_latest:,.0f}", "")
        st.caption("No earlier census year in selection for comparison.")

    st.caption(f"Ontario counties — {metric}")

    chart_df = f[["YEAR", "GEO", metric]].rename(columns={metric: "VALUE"})
    present_geos = sorted(chart_df["GEO"].unique().tolist())
    y_title = metric

    dl_name = build_download_name(
        "Ontario county CEAG", metric, yr_from, yr_to, present_geos
    )
    alt.renderers.set_embed_options(
        actions={"export": True, "source": False, "compiled": False, "editor": False},
        downloadFileName=dl_name,
    )

    chart = make_line_chart(chart_df, y_title, present_geos)
    st.altair_chart(chart, use_container_width=True)

    st.caption("Source: Custom Ontario county Census of Agriculture dataset")
    st.dataframe(f)

# ============================================================
# COSTS & INFLATION VIEW
# ============================================================
elif level == "Costs & Inflation":
    st.subheader("Costs & Inflation (Price Indices & CPI)")

    dataset_options = {
        # Farm input & machinery price indices
        "Farm input price index (FIPI, quarterly)": "18-10-0258-01.csv",
        "Machinery & equipment price index (MEPI, quarterly)": "18-10-0270-01.csv",

        # Consumer fuel prices
        "Gasoline CPI (retail price index)": "18-10-0001-01.csv",
        "Diesel fuel CPI (retail price index)": "18-10-0002-01.csv",

        # Freight service price indices
        "For-hire motor carrier freight services price index": "18-10-0281-01.csv",
        "Freight Rail Services Price Index": "18-10-0212-01.csv",

        # Food CPI (downstream price signal)
        "Food CPI (index – 2002=100)": "18-10-0004-03.csv",
    }

    dataset_label = st.sidebar.selectbox("Dataset", list(dataset_options.keys()))
    filename = dataset_options[dataset_label]
    table_id = filename.replace(".csv", "")

    df = load_csv(filename)
    if df.empty:
        st.info("Run the pipeline first to fetch data: `python -m scripts.run_pipeline`")
        st.stop()

    # For MEPI, keep only "Total domestic and imported" and clarify Canada-only
    if table_id == "18-10-0270-01":
        if "Machinery and equipment, domestic and imported" in df.columns:
            df = df[
                df["Machinery and equipment, domestic and imported"]
                == "Total domestic and imported"
            ].copy()
        st.info(
            "MEPI is available at the **Canada** level only "
            "in Statistics Canada Table 18-10-0270-01."
        )

    # Clean YEAR + VALUE and drop rows without YEAR
    df = df.dropna(subset=["YEAR"]).copy()
    df["YEAR"] = df["YEAR"].astype(int)
    if "VALUE" in df.columns:
        df["VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce")

    # Compare mode: geographies vs series
    compare_mode = st.sidebar.radio(
        "Compare by",
        ["Geographies", "Series"],
        index=0,
        help="Geographies: one series, multiple geographies. Series: one geography, multiple series.",
    )

    # SERIES SELECTOR(S)
    dim_col = find_series_column(table_id, df)
    selected_series: str | None = None
    selected_series_list: list[str] | None = None

    if dim_col:
        all_series = sorted(df[dim_col].dropna().unique().tolist())
        series_filter = st.sidebar.text_input("Filter series (optional)", "").strip().lower()
        series_options = (
            [s for s in all_series if series_filter in str(s).lower()]
            if series_filter
            else all_series
        )
        if not series_options:
            st.info("No series match your filter.")
            st.stop()

        if compare_mode == "Geographies":
            default_series = series_options[0]

            if table_id == "18-10-0258-01":  # FIPI
                default_series = next(
                    (s for s in series_options if str(s).startswith("Farm input total")),
                    series_options[0],
                )
            elif table_id == "18-10-0270-01":  # MEPI
                default_series = next(
                    (s for s in series_options if "All manufacturing" in str(s)),
                    series_options[0],
                )
            elif table_id in {"18-10-0001-01", "18-10-0002-01"}:  # fuel CPI
                default_series = next(
                    (s for s in series_options if "gasoline" in str(s).lower()
                     or "diesel" in str(s).lower()),
                    series_options[0],
                )
            elif table_id == "18-10-0281-01":  # trucking SPI
                default_series = next(
                    (s for s in series_options if "Truck transportation" in str(s)),
                    series_options[0],
                )
            elif table_id == "18-10-0212-01":  # freight rail SPI
                default_series = next(
                    (s for s in series_options if "Total freight rail services" in str(s)),
                    series_options[0],
                )
            elif table_id == "18-10-0004-03":  # Food CPI
                default_series = next(
                    (
                        s
                        for s in series_options
                        if str(s).startswith("Food")
                        or "Food purchased from stores" in str(s)
                    ),
                    series_options[0],
                )

            selected_series = st.sidebar.selectbox(
                "Series", series_options, index=series_options.index(default_series)
            )
            df = df[df[dim_col] == selected_series].copy()
        else:
            # Series comparison: one geography, multiple series
            preferred: list[str] = []
            if table_id == "18-10-0258-01":
                preferred = [
                    s
                    for s in series_options
                    if str(s).startswith("Farm input total")
                    or "crop inputs" in str(s).lower()
                    or "livestock inputs" in str(s).lower()
                ]
            elif table_id == "18-10-0270-01":
                preferred = [
                    s
                    for s in series_options
                    if "All manufacturing" in str(s)
                    or "Crop and animal production" in str(s)
                    or "Animal production" in str(s)
                ]
            elif table_id in {"18-10-0001-01", "18-10-0002-01"}:
                preferred = [
                    s
                    for s in series_options
                    if "gasoline" in str(s).lower()
                    or "diesel" in str(s).lower()
                ]
            elif table_id == "18-10-0004-03":
                preferred = [
                    s
                    for s in series_options
                    if str(s).startswith("Food")
                    or "Food purchased from stores" in str(s)
                ]
            elif table_id == "18-10-0281-01":
                preferred = [
                    s
                    for s in series_options
                    if "Truck transportation" in str(s)
                ]
            elif table_id == "18-10-0212-01":
                preferred = [
                    s
                    for s in series_options
                    if "Total freight rail services" in str(s)
                ]

            default_series_list = preferred or (series_options[:3] if len(series_options) > 3 else series_options)
            selected_series_list = st.sidebar.multiselect(
                "Series (multiple allowed)", series_options, default=default_series_list
            )
            if not selected_series_list:
                st.info("Select at least one series.")
                st.stop()
            df = df[df[dim_col].isin(selected_series_list)].copy()
    else:
        if compare_mode == "Series":
            st.info("This dataset has no separate series dimension; using geography comparison instead.")
            compare_mode = "Geographies"

    # GEOGRAPHY SELECTORS
    if "GEO" not in df.columns or "VALUE" not in df.columns:
        st.error("Expected columns 'GEO' and 'VALUE' not found in this dataset.")
        st.stop()

    geos_all = sorted(df["GEO"].dropna().unique().tolist())

    if compare_mode == "Geographies":
        if table_id in {"18-10-0270-01", "18-10-0281-01", "18-10-0212-01"} and geos_all == ["Canada"]:
            pick = ["Canada"]
        else:
            default_geos: list[str] = []
            if "Canada" in geos_all:
                default_geos.append("Canada")
            if "Ontario" in geos_all:
                default_geos.append("Ontario")
            if not default_geos and geos_all:
                default_geos = [geos_all[0]]
            pick = st.sidebar.multiselect("Geography", geos_all, default=default_geos)
        geo_sel = None
    else:
        default_geo = "Canada" if "Canada" in geos_all else geos_all[0]
        geo_sel = st.sidebar.selectbox(
            "Geography", geos_all, index=geos_all.index(default_geo)
        )
        pick = [geo_sel]

    # YEAR RANGE
    yrs = df[df["GEO"].isin(pick)]["YEAR"]
    if yrs.empty:
        st.info("No data for selected filters.")
        st.stop()

    yr_min, yr_max = int(yrs.min()), int(yrs.max())
    if yr_min < yr_max:
        yr_from, yr_to = st.sidebar.slider(
            "Year range", min_value=yr_min, max_value=yr_max, value=(yr_min, yr_max)
        )
    else:
        yr_from, yr_to = yr_min, yr_max

    if compare_mode == "Geographies":
        f = df[(df["GEO"].isin(pick)) & (df["YEAR"].between(yr_from, yr_to))].copy()
    else:
        f = df[(df["GEO"] == geo_sel) & (df["YEAR"].between(yr_from, yr_to))].copy()

    if f.empty:
        st.info("No data for selected filters.")
        st.stop()

    # Build DATE column for quarterly/monthly time series
    if "REF_DATE" in f.columns:
        if any("Q" in str(x) for x in f["REF_DATE"].unique()):
            def quarter_to_date(qstr: str) -> str:
                text = str(qstr)
                year = int(text[:4])
                q = text[-1]
                qm = {"1": "03-31", "2": "06-30", "3": "09-30", "4": "12-31"}.get(q, "12-31")
                return f"{year}-{qm}"

            f["DATE"] = pd.to_datetime(
                f["REF_DATE"].astype(str).apply(quarter_to_date), errors="coerce"
            )
        else:
            f["DATE"] = pd.to_datetime(
                f["REF_DATE"].astype(str).str.slice(0, 7) + "-01", errors="coerce"
            )

    # KPI construction
    kpi_rows: list[tuple[str, object, float, float | None]] = []
    if compare_mode == "Geographies":
        if "DATE" in f.columns and not f["DATE"].isna().all():
            f2 = f.dropna(subset=["DATE"]).copy()
            f2 = f2.sort_values(["GEO", "DATE"])
            f2["PREV_VALUE"] = f2.groupby("GEO")["VALUE"].shift(1)
            latest_date = f2["DATE"].max()
            last_rows = f2[f2["DATE"] == latest_date].copy()
            for _, row in last_rows.iterrows():
                geo = row["GEO"]
                val = row["VALUE"]
                prev = row["PREV_VALUE"]
                change = (val - prev) / prev * 100 if pd.notnull(prev) and prev else None
                kpi_rows.append((geo, latest_date.date(), float(val), change))
            metric_title_suffix = f"on {latest_date.date()}"
        else:
            f2 = f.sort_values(["GEO", "YEAR"])
            f2["PREV_VALUE"] = f2.groupby("GEO")["VALUE"].shift(1)
            latest_year = int(f2["YEAR"].max())
            last_rows = f2[f2["YEAR"] == latest_year].copy()
            for _, row in last_rows.iterrows():
                geo = row["GEO"]
                val = row["VALUE"]
                prev = row["PREV_VALUE"]
                change = (val - prev) / prev * 100 if pd.notnull(prev) and prev else None
                kpi_rows.append((geo, latest_year, float(val), change))
            metric_title_suffix = f"in {latest_year}"
    else:
        if dim_col is None:
            st.error("Series comparison requires a series dimension.")
            st.stop()

        if "DATE" in f.columns and not f["DATE"].isna().all():
            f2 = f.dropna(subset=["DATE"]).copy()
            f2 = f2.sort_values([dim_col, "DATE"])
            f2["PREV_VALUE"] = f2.groupby(dim_col)["VALUE"].shift(1)
            latest_date = f2["DATE"].max()
            last_rows = f2[f2["DATE"] == latest_date].copy()
            for _, row in last_rows.iterrows():
                s_name = row[dim_col]
                val = row["VALUE"]
                prev = row["PREV_VALUE"]
                change = (val - prev) / prev * 100 if pd.notnull(prev) and prev else None
                kpi_rows.append((str(s_name), latest_date.date(), float(val), change))
            metric_title_suffix = f"on {latest_date.date()}"
        else:
            f2 = f.sort_values([dim_col, "YEAR"])
            f2["PREV_VALUE"] = f2.groupby(dim_col)["VALUE"].shift(1)
            latest_year = int(f2["YEAR"].max())
            last_rows = f2[f2["YEAR"] == latest_year].copy()
            for _, row in last_rows.iterrows():
                s_name = row[dim_col]
                val = row["VALUE"]
                prev = row["PREV_VALUE"]
                change = (val - prev) / prev * 100 if pd.notnull(prev) and prev else None
                kpi_rows.append((str(s_name), latest_year, float(val), change))
            metric_title_suffix = f"in {latest_year}"

    if kpi_rows:
        n = min(len(kpi_rows), 6)
        cols = st.columns(n)
        for i, (label_item, period, val, change) in enumerate(kpi_rows[:n]):
            with cols[i]:
                label = f"{dataset_label} — {label_item} {metric_title_suffix}"
                delta_txt = (
                    f"{change:+.1f}% vs previous period" if change is not None else ""
                )
                st.metric(label, f"{val:,.1f}", delta_txt)

    unit_lbl = get_unit_label(f)
    if compare_mode == "Geographies":
        if "DATE" in f.columns and not f["DATE"].isna().all():
            long_df = f[["DATE", "GEO", "VALUE"]].copy()
            present_geos = sorted(long_df["GEO"].unique().tolist())
            y_title = f"{dataset_label}{(' (' + unit_lbl + ')') if unit_lbl else ''}"

            dl_series_label = selected_series
            dl_name = build_download_name(
                dataset_label, dl_series_label, yr_from, yr_to, present_geos
            )
            alt.renderers.set_embed_options(
                actions={"export": True, "source": False, "compiled": False, "editor": False},
                downloadFileName=dl_name,
            )

            chart = (
                alt.Chart(long_df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("DATE:T", title="Date"),
                    y=alt.Y("VALUE:Q", title=y_title, axis=alt.Axis(format="~s")),
                    color=alt.Color(
                        "GEO:N",
                        title="Geography",
                        scale=altair_colour_scale(present_geos),
                    ),
                    tooltip=[
                        alt.Tooltip("DATE:T", title="Date"),
                        alt.Tooltip("GEO:N", title="Geography"),
                        alt.Tooltip("VALUE:Q", title="Value", format=","),
                    ],
                )
                .properties(height=420)
                .interactive()
            )
        else:
            long_df = f[["YEAR", "GEO", "VALUE"]].copy()
            present_geos = sorted(long_df["GEO"].unique().tolist())
            y_title = f"{dataset_label}{(' (' + unit_lbl + ')') if unit_lbl else ''}"

            dl_series_label = selected_series
            dl_name = build_download_name(
                dataset_label, dl_series_label, yr_from, yr_to, present_geos
            )
            alt.renderers.set_embed_options(
                actions={"export": True, "source": False, "compiled": False, "editor": False},
                downloadFileName=dl_name,
            )

            chart = make_line_chart(long_df, y_title, present_geos)
    else:
        if dim_col is None:
            st.error("Series comparison requires a series dimension.")
            st.stop()

        if "DATE" in f.columns and not f["DATE"].isna().all():
            long_df = f[["DATE", dim_col, "VALUE"]].copy()
            present_geos = [geo_sel] if geo_sel else []
            y_title = f"{dataset_label}{(' (' + unit_lbl + ')') if unit_lbl else ''}"

            if selected_series_list:
                if len(selected_series_list) == 1:
                    dl_series_label = selected_series_list[0]
                else:
                    dl_series_label = f"{len(selected_series_list)} series"
            else:
                dl_series_label = None

            dl_name = build_download_name(
                dataset_label, dl_series_label, yr_from, yr_to, present_geos
            )
            alt.renderers.set_embed_options(
                actions={"export": True, "source": False, "compiled": False, "editor": False},
                downloadFileName=dl_name,
            )

            chart = (
                alt.Chart(long_df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("DATE:T", title="Date"),
                    y=alt.Y("VALUE:Q", title=y_title, axis=alt.Axis(format="~s")),
                    color=alt.Color(f"{dim_col}:N", title="Series"),
                    tooltip=[
                        alt.Tooltip("DATE:T", title="Date"),
                        alt.Tooltip(f"{dim_col}:N", title="Series"),
                        alt.Tooltip("VALUE:Q", title="Value", format=","),
                    ],
                )
                .properties(height=420)
                .interactive()
            )
        else:
            long_df = f[["YEAR", dim_col, "VALUE"]].copy()
            long_df["YEAR"] = long_df["YEAR"].astype(int).astype(str)
            present_geos = [geo_sel] if geo_sel else []
            y_title = f"{dataset_label}{(' (' + unit_lbl + ')') if unit_lbl else ''}"

            if selected_series_list:
                if len(selected_series_list) == 1:
                    dl_series_label = selected_series_list[0]
                else:
                    dl_series_label = f"{len(selected_series_list)} series"
            else:
                dl_series_label = None

            dl_name = build_download_name(
                dataset_label, dl_series_label, yr_from, yr_to, present_geos
            )
            alt.renderers.set_embed_options(
                actions={"export": True, "source": False, "compiled": False, "editor": False},
                downloadFileName=dl_name,
            )

            chart = (
                alt.Chart(long_df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("YEAR:O", title="Year", axis=alt.Axis(format="d")),
                    y=alt.Y("VALUE:Q", title=y_title, axis=alt.Axis(format="~s")),
                    color=alt.Color(f"{dim_col}:N", title="Series"),
                    tooltip=[
                        alt.Tooltip("YEAR:O", title="Year", format="d"),
                        alt.Tooltip(f"{dim_col}:N", title="Series"),
                        alt.Tooltip("VALUE:Q", title="Value", format=","),
                    ],
                )
                .properties(height=420)
                .interactive()
            )

    st.altair_chart(chart, use_container_width=True)

    st.caption(f"Unit: {unit_lbl if unit_lbl else '—'}")
    st.caption(f"Source: Statistics Canada Table {table_id}")
    st.dataframe(f)

# ============================================================
# TRANSPORTATION & EXPORTS VIEW
# ============================================================
elif level == "Transportation & Exports":
    st.subheader("Transportation & Exports (Farm logistics & trade flows)")

    # All active Transportation & Exports tables from config,
    # excluding 32-10-0049-01 (farm operating expenses).
    transport_tables = [
        t
        for t in TABLE_CONFIG
        if t.get("theme") == "Transportation & Exports"
        and t.get("active", True)
        and t.get("id") != "32-10-0049-01"
    ]
    if not transport_tables:
        st.info("No transportation & exports tables are configured in tables.yml yet.")
        st.stop()

    dataset_label = st.sidebar.selectbox(
        "Dataset", [t["name"] for t in transport_tables]
    )
    tbl = next(t for t in transport_tables if t["name"] == dataset_label)
    table_id = tbl["id"]
    filename = f"{table_id}.csv"

    df = load_csv(filename)
    if df.empty:
        st.info("Run the pipeline first to fetch data: `python -m scripts.run_pipeline`")
        st.stop()

    if "VALUE" not in df.columns:
        st.error("Expected column 'VALUE' not found in this dataset.")
        st.stop()
    df["VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce")

    # Optional GEO filter
    geo_choice = None
    if "GEO" in df.columns and df["GEO"].notna().any():
        geos = sorted(df["GEO"].dropna().unique().tolist())
        default_geo = "Canada" if "Canada" in geos else geos[0]
        geo_choice = st.sidebar.selectbox(
            "Geography", geos, index=geos.index(default_geo)
        )
        df = df[df["GEO"] == geo_choice].copy()

    # Table-specific extra filters

    # Trucking financial statistics – NAICS industry group
    if table_id == "23-10-0291-01":
        naics_col = find_column_fuzzy(df, ["North American Industry Classification System"])
        if naics_col:
            naics_vals = sorted(df[naics_col].dropna().unique().tolist())
            default_naics = next(
                (v for v in naics_vals if "Total, truck transportation" in str(v)),
                naics_vals[0],
            )
            naics_choice = st.sidebar.selectbox(
                "Industry group (NAICS)", naics_vals, index=naics_vals.index(default_naics)
            )
            df = df[df[naics_col] == naics_choice].copy()

    # Railway carloadings – measure (Estimates)
    if table_id == "23-10-0216-02":
        est_col = find_column_fuzzy(df, ["Estimates"])
        if est_col:
            measures = sorted(df[est_col].dropna().unique().tolist())
            default_meas = next(
                (m for m in measures if "tonnes" in str(m).lower()),
                measures[0],
            )
            meas_choice = st.sidebar.selectbox(
                "Measure", measures, index=measures.index(default_meas)
            )
            df = df[df[est_col] == meas_choice].copy()

    # CFAF – characteristics and commodity group
    if table_id == "23-10-0142-01":
        char_col = find_column_fuzzy(df, ["Characteristics"])
        if char_col:
            chars = sorted(df[char_col].dropna().unique().tolist())
            default_char = next(
                (c for c in chars if "value" in str(c).lower()),
                chars[0],
            )
            char_choice = st.sidebar.selectbox(
                "Characteristic", chars, index=chars.index(default_char)
            )
            df = df[df[char_col] == char_choice].copy()

        comm_col = find_column_fuzzy(df, ["Commodity group"])
        if comm_col:
            comms = sorted(df[comm_col].dropna().unique().tolist())
            default_comm = next(
                (c for c in comms if "agricultural products" in str(c).lower()),
                None,
            )
            if default_comm is None:
                default_comm = next(
                    (c for c in comms if "total" in str(c).lower()),
                    comms[0],
                )
            comm_choice = st.sidebar.selectbox(
                "Commodity group", comms, index=comms.index(default_comm)
            )
            df = df[df[comm_col] == comm_choice].copy()

    # CIMT by mode – trading partner, NAPCS, and trade-flow selector
    if table_id == "12-10-0177-01":
        partner_col = find_column_fuzzy(df, ["Principal trading partner"])
        if partner_col:
            partners = sorted(df[partner_col].dropna().unique().tolist())
            default_partner = next(
                (p for p in partners if "all countries" in str(p).lower()),
                partners[0],
            )
            partner_choice = st.sidebar.selectbox(
                "Principal trading partner", partners, index=partners.index(default_partner)
            )
            df = df[df[partner_col] == partner_choice].copy()

        napcs_col = find_column_fuzzy(df, ["North American Product Classification System", "NAPCS"])
        if napcs_col:
            napcs_vals = sorted(df[napcs_col].dropna().unique().tolist())
            default_napcs = next(
                (v for v in napcs_vals if "fresh fruit, nuts and vegetables" in str(v).lower()),
                None,
            )
            if default_napcs is None:
                default_napcs = next(
                    (v for v in napcs_vals if "farm, fishing and intermediate food products" in str(v).lower()),
                    None,
                )
            if default_napcs is None:
                default_napcs = next(
                    (v for v in napcs_vals if "total" in str(v).lower()),
                    napcs_vals[0],
                )
            napcs_choice = st.sidebar.selectbox(
                "Commodity (NAPCS)", napcs_vals, index=napcs_vals.index(default_napcs)
            )
            df = df[df[napcs_col] == napcs_choice].copy()

        trade_col = find_column_fuzzy(df, ["Trade"])
        if trade_col:
            trade_vals = sorted(df[trade_col].dropna().unique().tolist())
            default_flows = [v for v in trade_vals if "export" in str(v).lower()]
            if not default_flows:
                default_flows = trade_vals
            flow_pick = st.sidebar.multiselect(
                "Trade flow (Exports / Imports)", trade_vals, default=default_flows
            )
            if not flow_pick:
                st.info("Select at least one trade flow.")
                st.stop()
            df = df[df[trade_col].isin(flow_pick)].copy()

    # Determine primary series dimension
    dim_col = find_series_column(table_id, df)

    # For CIMT, treat each Mode × Trade-flow combo as its own series
    if table_id == "12-10-0177-01":
        trade_col = find_column_fuzzy(df, ["Trade"])
        if trade_col and dim_col:
            df["Series"] = df[trade_col].astype(str) + " – " + df[dim_col].astype(str)
            dim_col = "Series"

    selected_series_list: list[str] | None = None
    series_label_for_filename: str | None = None

    if dim_col:
        all_series = sorted(df[dim_col].dropna().unique().tolist())
        series_filter = st.sidebar.text_input("Filter series (optional)", "").strip().lower()
        series_options = (
            [s for s in all_series if series_filter in str(s).lower()]
            if series_filter
            else all_series
        )
        if not series_options:
            st.info("No series match your filter.")
            st.stop()

        default_series_list = series_options[:5] if len(series_options) > 5 else series_options

        if table_id == "12-10-0177-01":
            pref = [
                s
                for s in series_options
                if any(
                    kw in str(s).lower()
                    for kw in ["road", "rail", "water", "air", "other"]
                )
            ]
            if pref:
                default_series_list = pref
        elif table_id == "23-10-0216-02":
            pref = [
                s
                for s in series_options
                if any(
                    kw in str(s).lower()
                    for kw in ["total traffic carried", "total non-intermodal traffic loaded", "wheat", "canola"]
                )
            ]
            if pref:
                default_series_list = pref

        selected_series_list = st.sidebar.multiselect(
            "Series (multiple allowed)", series_options, default=default_series_list
        )
        if not selected_series_list:
            st.info("Select at least one series.")
            st.stop()
        df = df[df[dim_col].isin(selected_series_list)].copy()

        if len(selected_series_list) == 1:
            series_label_for_filename = str(selected_series_list[0])
        else:
            series_label_for_filename = f"{len(selected_series_list)} series"

    # Build DATE column from REF_DATE or YEAR
    if "REF_DATE" in df.columns:
        if any("Q" in str(x) for x in df["REF_DATE"].unique()):
            def quarter_to_date(qstr: str) -> str:
                text = str(qstr)
                year = int(text[:4])
                q = text[-1]
                qm = {"1": "03-31", "2": "06-30", "3": "09-30", "4": "12-31"}.get(q, "12-31")
                return f"{year}-{qm}"

            df["DATE"] = pd.to_datetime(
                df["REF_DATE"].astype(str).apply(quarter_to_date), errors="coerce"
            )
        else:
            df["DATE"] = pd.to_datetime(
                df["REF_DATE"].astype(str).str.slice(0, 7) + "-01", errors="coerce"
            )
    elif "YEAR" in df.columns:
        df["DATE"] = pd.to_datetime(
            df["YEAR"].astype(int).astype(str) + "-01-01", errors="coerce"
        )
    else:
        df["DATE"] = pd.NaT

    if df["DATE"].notna().any():
        df = df.dropna(subset=["DATE"]).copy()
        df["YEAR"] = df["DATE"].dt.year
    elif "YEAR" in df.columns:
        df = df.dropna(subset=["YEAR"]).copy()
        df["YEAR"] = df["YEAR"].astype(int)
    else:
        st.error("Could not determine a time dimension for this dataset.")
        st.stop()

    yrs = df["YEAR"]
    yr_min, yr_max = int(yrs.min()), int(yrs.max())
    if yr_min < yr_max:
        yr_from, yr_to = st.sidebar.slider(
            "Year range", min_value=yr_min, max_value=yr_max, value=(yr_min, yr_max)
        )
    else:
        yr_from, yr_to = yr_min, yr_max
    df = df[df["YEAR"].between(yr_from, yr_to)].copy()
    if df.empty:
        st.info("No data for selected filters.")
        st.stop()

    # Capture units BEFORE aggregation
    unit_lbl = get_unit_label(df)

    # Aggregate across remaining dimensions to avoid vertical spikes
    has_date = "DATE" in df.columns and df["DATE"].notna().any()
    group_cols = ["DATE" if has_date else "YEAR"]
    if dim_col:
        group_cols.append(dim_col)
    if geo_choice and "GEO" in df.columns:
        group_cols.append("GEO")

    df = df.groupby(group_cols, as_index=False)["VALUE"].sum()
    if has_date:
        df["YEAR"] = df["DATE"].dt.year

    # KPI metrics
    kpi_rows: list[tuple[str, object, float, float | None]] = []
    time_col = "DATE" if has_date else "YEAR"
    latest_period = df[time_col].max()
    if dim_col:
        grouped = df.sort_values(time_col).groupby(dim_col)
        for series_name, sdf in grouped:
            sdf = sdf.dropna(subset=["VALUE"])
            if sdf.empty:
                continue
            last = sdf.iloc[-1]
            val = float(last["VALUE"])
            if len(sdf) > 1:
                prev = sdf.iloc[-2]["VALUE"]
                change = (val - prev) / prev * 100 if prev else None
            else:
                change = None
            period = last[time_col]
            kpi_rows.append((str(series_name), period, val, change))
    else:
        sdf = df.sort_values(time_col).dropna(subset=["VALUE"])
        if not sdf.empty:
            last = sdf.iloc[-1]
            val = float(last["VALUE"])
            if len(sdf) > 1:
                prev = sdf.iloc[-2]["VALUE"]
                change = (val - prev) / prev * 100 if prev else None
            else:
                change = None
            label = dataset_label if geo_choice is None else f"{dataset_label} — {geo_choice}"
            period = last[time_col]
            kpi_rows.append((label, period, val, change))

    def _period_label(p):
        if isinstance(p, pd.Timestamp):
            return p.date().isoformat()
        try:
            return str(int(p))
        except Exception:
            return str(p)

    if kpi_rows:
        n = min(len(kpi_rows), 6)
        cols = st.columns(n)
        for i, (label, period, val, change) in enumerate(kpi_rows[:n]):
            with cols[i]:
                delta_txt = (
                    f"{change:+.1f}% vs previous period" if change is not None else ""
                )
                cols[i].metric(f"{label} on {_period_label(period)}", f"{val:,.1f}", delta_txt)

    # Chart
    y_title = f"{dataset_label}{(' (' + unit_lbl + ')') if unit_lbl else ''}"
    present_labels: list[str] = []
    if dim_col:
        present_labels = sorted(df[dim_col].dropna().unique().tolist())
    elif geo_choice:
        present_labels = [geo_choice]

    dl_name = build_download_name(
        dataset_label, series_label_for_filename, yr_from, yr_to, present_labels
    )
    alt.renderers.set_embed_options(
        actions={"export": True, "source": False, "compiled": False, "editor": False},
        downloadFileName=dl_name,
    )

    n_periods = df[time_col].nunique()
    one_period_cross_section = dim_col is not None and n_periods == 1

    if one_period_cross_section and dim_col:
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X(f"{dim_col}:N", title="Series"),
                y=alt.Y("VALUE:Q", title=y_title, axis=alt.Axis(format="~s")),
                tooltip=[
                    alt.Tooltip(f"{dim_col}:N", title="Series"),
                    alt.Tooltip("VALUE:Q", title="Value", format=","),
                ],
            )
            .properties(height=420)
        )
    else:
        x_enc = (
            alt.X("DATE:T", title="Date")
            if has_date
            else alt.X("YEAR:O", title="Year", axis=alt.Axis(format="d"))
        )
        if dim_col:
            chart = (
                alt.Chart(df)
                .mark_line(point=True)
                .encode(
                    x=x_enc,
                    y=alt.Y("VALUE:Q", title=y_title, axis=alt.Axis(format="~s")),
                    color=alt.Color(f"{dim_col}:N", title="Series"),
                    tooltip=[
                        alt.Tooltip("DATE:T", title="Date"),
                        alt.Tooltip(f"{dim_col}:N", title="Series"),
                        alt.Tooltip("VALUE:Q", title="Value", format=","),
                    ],
                )
                .properties(height=420)
                .interactive()
            )
        else:
            chart = (
                alt.Chart(df)
                .mark_line(point=True)
                .encode(
                    x=x_enc,
                    y=alt.Y("VALUE:Q", title=y_title, axis=alt.Axis(format="~s")),
                    tooltip=[
                        alt.Tooltip("DATE:T", title="Date"),
                        alt.Tooltip("VALUE:Q", title="Value", format=","),
                    ],
                )
                .properties(height=420)
                .interactive()
            )

    st.altair_chart(chart, use_container_width=True)
    st.caption(f"Unit: {unit_lbl if unit_lbl else '—'}")
    st.caption(f"Source: Statistics Canada Table {table_id}")
    st.dataframe(df)

# ============================================================
# FARM FINANCE (CANADA / PROVINCE) VIEW
# ============================================================
else:
    st.subheader("Farm finance statistics – Canada & Provinces")

    dataset_options = {
        # Core revenue / expense accounts
        "Farm cash receipts (annual)": "32-10-0045-01.csv",
        "Operating revenues & expenses (ATDP)": "32-10-0136-01.csv",
        "Farm operating expenses & depreciation charges": "32-10-0049-01.csv",
        "Net farm income": "32-10-0052-01.csv",
        "Direct program payments to producers": "32-10-0106-01.csv",
        "Agriculture value added account": "32-10-0048-01.csv",
        "Farm income in kind in Canada": "32-10-0055-01.csv",

        # Balance sheet / stock accounts
        "Farm capital (value of assets)": "32-10-0050-01.csv",
        "Farm debt outstanding": "32-10-0051-01.csv",
        "Balance sheet of farm sector": "32-10-0056-01.csv",
    }

    dataset_label = st.sidebar.selectbox("Dataset", list(dataset_options.keys()))
    filename = dataset_options[dataset_label]
    table_id = filename.replace(".csv", "")

    df = load_csv(filename)
    if df.empty:
        st.info("Run the pipeline first: `python -m scripts.run_pipeline`")
        st.stop()

    # ATDP-specific filtering
    if table_id == "32-10-0136-01":
        if "Farm type" in df.columns:
            df = df[df["Farm type"] == "All farm types"]
        if "Revenue class" in df.columns:
            df = df[df["Revenue class"] == "All revenue classes"]
        if "Estimate type" in df.columns:
            df = df[df["Estimate type"] == "Total estimate"]

    df = df.dropna(subset=["YEAR"]).copy()
    df["YEAR"] = df["YEAR"].astype(int)
    if "VALUE" in df.columns:
        df["VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce")

    dim_col = find_series_column(table_id, df)
    selected_series = None
    if dim_col:
        all_series = sorted(df[dim_col].dropna().unique().tolist())
        series_filter = st.sidebar.text_input("Filter series (optional)", "").strip().lower()
        series_options = (
            [s for s in all_series if series_filter in str(s).lower()]
            if series_filter
            else all_series
        )
        if not series_options:
            st.info("No series match your filter.")
            st.stop()

        # Prefer "Total" where it exists; otherwise first series
        default_series = next(
            (s for s in series_options if str(s).startswith("Total ")),
            series_options[0],
        )

        selected_series = st.sidebar.selectbox(
            "Series", series_options, index=series_options.index(default_series)
        )
        df = df[df[dim_col] == selected_series].copy()

    if "GEO" not in df.columns or "VALUE" not in df.columns:
        st.error("Expected 'GEO' and 'VALUE' columns not found.")
        st.stop()

    geos = sorted(df["GEO"].dropna().unique().tolist())
    default_geos: list[str] = []
    if "Canada" in geos:
        default_geos.append("Canada")
    if "Ontario" in geos:
        default_geos.append("Ontario")
    if not default_geos and geos:
        default_geos = [geos[0]]

    pick = st.sidebar.multiselect("Geography", geos, default=default_geos)

    yrs = df[df["GEO"].isin(pick)]["YEAR"]
    if yrs.empty:
        st.info("No data for selected filters.")
        st.stop()

    yr_min, yr_max = int(yrs.min()), int(yrs.max())
    yr_from, yr_to = st.sidebar.slider(
        "Year range", min_value=yr_min, max_value=yr_max, value=(yr_min, yr_max), step=1
    )

    f = df[(df["GEO"].isin(pick)) & (df["YEAR"].between(yr_from, yr_to))].copy()
    if f.empty:
        st.info("No data for selected filters.")
        st.stop()

    latest_slice = f[f["YEAR"] == yr_to]
    prev_slice = f[f["YEAR"] < yr_to]
    total_latest = latest_slice["VALUE"].sum()

    if not prev_slice.empty:
        prev_year = int(prev_slice["YEAR"].max())
        prev_total = prev_slice[prev_slice["YEAR"] == prev_year]["VALUE"].sum()
        change = (total_latest - prev_total) / prev_total * 100 if prev_total else 0.0
        st.metric(
            f"{dataset_label} in {yr_to}",
            f"{total_latest:,.0f}",
            f"{change:+.1f}% vs {prev_year}",
        )
    else:
        st.metric(f"{dataset_label} in {yr_to}", f"{total_latest:,.0f}", "")

    if selected_series:
        st.caption(f"{dataset_label} — {selected_series}")
    else:
        st.caption(dataset_label)

    long_df = f[["YEAR", "GEO", "VALUE"]].copy()
    present_geos = sorted(long_df["GEO"].unique().tolist())
    unit_lbl = get_unit_label(f)
    y_title = f"{dataset_label}{(' (' + unit_lbl + ')') if unit_lbl else ''}"

    dl_name = build_download_name(
        dataset_label, selected_series, yr_from, yr_to, present_geos
    )
    alt.renderers.set_embed_options(
        actions={"export": True, "source": False, "compiled": False, "editor": False},
        downloadFileName=dl_name,
    )

    chart = make_line_chart(long_df, y_title, present_geos)
    st.altair_chart(chart, use_container_width=True)

    st.caption(f"Unit: {unit_lbl if unit_lbl else '—'}")
    st.caption(f"Source: Statistics Canada Table {table_id}")
    st.dataframe(f)
