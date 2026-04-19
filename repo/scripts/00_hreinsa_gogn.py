"""
00_hreinsa_gogn.py
------------------
Hleður hrágögnum LamaH-Ice fyrir Ölfusá við Selfoss (ID 98), síar í
verkefnistímabilið 1993-10-01 – 2023-09-30 og flytur út tvær hreinar CSV skrár:

    data/olfus_flaedi_93_23.csv      – daglegt rennsli  [m³/s]
    data/olfus_vedur_93_23.csv       – daglegt veðurfar: úrkoma [mm/dag], hiti [°C]
    data/olfus_eiginleikar.csv       – eiginleikar vatnasviðs (aðeins ID 98)

Allar niðurstreymis skriptur hlaða þessar tvær skrár í stað hrágagnanna.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[1]
DATA      = ROOT / "data"
FLOW_RAW  = DATA / "ID_98_OLF_RENNSLI.csv"
MET_RAW   = DATA / "ID_98_OLF_.VEDURcsv.csv"
ATTR_RAW  = DATA / "Catchment_attributes.csv"
FLOW_OUT  = DATA / "olfus_flaedi_93_23.csv"
MET_OUT   = DATA / "olfus_vedur_93_23.csv"
ATTR_OUT  = DATA / "olfus_eiginleikar.csv"

# Project period (hydrological years, Oct–Sep)
START = pd.Timestamp("1993-10-01")
END   = pd.Timestamp("2023-09-30")


# ── helpers ────────────────────────────────────────────────────────────────
def make_date(df: pd.DataFrame) -> pd.Series:
    """Build a DatetimeIndex from YYYY / MM / DD columns."""
    return pd.to_datetime(
        df[["YYYY", "MM", "DD"]].rename(columns={"YYYY": "year", "MM": "month", "DD": "day"})
    )


def filter_period(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[(df.index >= START) & (df.index <= END)].copy()


# ── 1. Streamflow ──────────────────────────────────────────────────────────
print("Loading flow data …")
flow_raw = pd.read_csv(FLOW_RAW, sep=";", dtype={"YYYY": int, "MM": int, "DD": int})
flow_raw.index = make_date(flow_raw)
flow_raw.index.name = "date"

flow = filter_period(flow_raw)[["qobs", "qc_flag"]].copy()
flow.rename(columns={"qobs": "Q_m3s"}, inplace=True)

# Quality control:
#   qc_flag == 40 means "good / unchecked"  (all values in this file are 40)
#   Replace any physically impossible values (negative) with NaN
n_neg = (flow["Q_m3s"] < 0).sum()
if n_neg:
    print(f"  → {n_neg} negative Q values set to NaN")
    flow.loc[flow["Q_m3s"] < 0, "Q_m3s"] = np.nan

n_missing = flow["Q_m3s"].isna().sum()
print(f"  Period: {flow.index[0].date()} → {flow.index[-1].date()}")
print(f"  Rows  : {len(flow)}  |  Missing Q: {n_missing}")

flow[["Q_m3s"]].to_csv(FLOW_OUT)
print(f"  Saved → {FLOW_OUT.relative_to(ROOT)}")


# ── 2. Meteorological data ─────────────────────────────────────────────────
print("\nLoading meteorological data …")
met_raw = pd.read_csv(MET_RAW, sep=";", dtype={"YYYY": int, "MM": int, "DD": int})
met_raw.index = make_date(met_raw)
met_raw.index.name = "date"

met = filter_period(met_raw).copy()

# Preferred columns (CARRA reanalysis – available from 1991-01-01)
# Fallback to ERA5-Land columns if CARRA is somehow missing
carra_prec = "prec_carra"          # mm/day
carra_temp = "2m_temp_carra"       # °C
era5_prec  = "prec"                # mm/day
era5_temp  = "2m_temp_mean"        # °C

# Convert to numeric (stray whitespace → NaN)
for col in [carra_prec, carra_temp, era5_prec, era5_temp]:
    met[col] = pd.to_numeric(met[col], errors="coerce")

# Use CARRA as primary, fill gaps with ERA5-Land
met["P_mm"]  = met[carra_prec].fillna(met[era5_prec])
met["T_degC"] = met[carra_temp].fillna(met[era5_temp])

# Sanity checks
n_neg_p = (met["P_mm"] < 0).sum()
if n_neg_p:
    print(f"  → {n_neg_p} negative P values set to NaN")
    met.loc[met["P_mm"] < 0, "P_mm"] = np.nan

n_missing_p = met["P_mm"].isna().sum()
n_missing_t = met["T_degC"].isna().sum()
print(f"  Period: {met.index[0].date()} → {met.index[-1].date()}")
print(f"  Rows  : {len(met)}  |  Missing P: {n_missing_p}  |  Missing T: {n_missing_t}")

# Also keep solid precipitation fraction for snow/melt context
if "solid_prec_carra" in met.columns:
    met["solid_P_mm"] = pd.to_numeric(met["solid_prec_carra"], errors="coerce")
    out_cols = ["P_mm", "T_degC", "solid_P_mm"]
else:
    out_cols = ["P_mm", "T_degC"]

met[out_cols].to_csv(MET_OUT)
print(f"  Saved → {MET_OUT.relative_to(ROOT)}")


# ── 3. Quick alignment check ───────────────────────────────────────────────
print("\nAlignment check …")
combined = flow[["Q_m3s"]].join(met[["P_mm", "T_degC"]], how="outer")

missing_any = combined.isna().any(axis=1).sum()
expected_days = (END - START).days + 1
print(f"  Expected days : {expected_days}")
print(f"  Combined rows : {len(combined)}")
print(f"  Rows with any NaN: {missing_any}")

if len(combined) != expected_days:
    missing_dates = pd.date_range(START, END).difference(combined.index)
    print(f"  !! {len(missing_dates)} dates missing from combined series:")
    print("    ", missing_dates.tolist()[:10], "…" if len(missing_dates) > 10 else "")
else:
    print("  ✓ All dates present, no gaps in date index")


# ── 4. Catchment attributes (aðeins ID 98) ────────────────────────────────
print("\nHleður eiginleikum vatnasviðs …")
attrs = pd.read_csv(ATTR_RAW, sep=";", index_col="id")
olfus = attrs.loc[[98]]  # halda DataFrame formi (ekki Series)
olfus.to_csv(ATTR_OUT)
print(f"  Raðir : {len(olfus)}  |  Dálkar: {len(olfus.columns)}")
print(f"  Vistað → {ATTR_OUT.relative_to(ROOT)}")

print("\nLokið.")
