"""
Batch-join daily meteorology (Open-Meteo ERA5 archive) to many CSVs in a folder.

- Scans a folder for files whose name ends with PATTERN_SUFFIX.
- Fetches DAILY variables: Tmax, Tmin, Precipitation.
- Fetches HOURLY variables: air temperature (2 m), relative humidity (2 m).
- Computes hourly VPD from T & RH, then aggregates to daily mean VPD; also daily mean RH.
- Computes GDD, EGDD, SDI from daily Tmax/Tmin.
- Writes "<original>_with_meteo.csv" next to each input.

This script has no CLI arguments. Edit the CONFIG block below to change paths/columns/parameters.
"""

import sys
import time
from pathlib import Path
from urllib.parse import urlencode
import pandas as pd
import numpy as np
import requests

# =====================
# CONFIG (edit here)
# =====================
INPUT_DIR = fr"C:\Users\Vlasis\Repos\WheatProteinRemoteSensing\outputs"
PATTERN_SUFFIX = "indices_20m_mean_bySample_interp3d.csv"
LAT_COL = "Latitude"     # latitude column in the input CSVs
LON_COL = "Longitude"    # longitude column in the input CSVs
DATE_COL = "Date"        # date column in the input CSVs (parseable to YYYY-MM-DD)

# Degree-day / stress parameters
GDD_BASE_C  = 10.0       # Base temperature (°C) for GDD
GDD_UPPER_C = 30.0       # Upper cap temperature (°C) for EGDD
SDI_TOPT_C  = 30.0       # Optimal temperature (°C) for SDI = Tmax - Topt

# Networking / batching
BATCH_SIZE = 40                 # number of unique (lat,lon) per batch
SLEEP_BETWEEN_BATCHES_S = 0.3   # polite pause between batches (seconds)
REQUEST_TIMEOUT_S = 60          # request timeout (seconds)

# Open-Meteo historical ERA5 endpoint
HIST_BASE = "https://archive-api.open-meteo.com/v1/era5"

# =====================
# Helpers
# =====================
def daterange_bounds(series: pd.Series) -> tuple[str, str]:
    """Return (min_date, max_date) as YYYY-MM-DD strings for the provided date series."""
    d = pd.to_datetime(series, errors="coerce").dt.normalize()
    if d.isna().all():
        raise ValueError(f"Date column '{DATE_COL}' could not be parsed")
    return d.min().strftime("%Y-%m-%d"), d.max().strftime("%Y-%m-%d")

def chunked(iterable, n):
    """Yield successive chunks of size n from an iterable."""
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf

def http_get(url: str):
    """GET helper with error body included on HTTP errors."""
    r = requests.get(url, timeout=REQUEST_TIMEOUT_S)
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        raise RuntimeError(f"{e}; body={r.text[:300]}") from None
    return r.json()

def fetch_daily(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """
    Fetch daily Tmax, Tmin, and precipitation for [start, end].
    Returns DataFrame with columns: date, tmax, tmin, precip_sum, lat, lon.
    """
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "daily": ",".join([
            "temperature_2m_max", "temperature_2m_min", "precipitation_sum"
        ]),
        "timezone": "auto",
    }
    url = f"{HIST_BASE}?{urlencode(params)}"
    js = http_get(url)
    daily = js.get("daily", {})
    if not daily or not daily.get("time"):
        return pd.DataFrame()
    df = pd.DataFrame(daily).rename(columns={
        "time": "date",
        "temperature_2m_max": "tmax",
        "temperature_2m_min": "tmin",
        "precipitation_sum": "precip_sum",
    })
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["lat"], df["lon"] = float(lat), float(lon)
    return df

def fetch_hourly(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """
    Fetch hourly T (2 m) and RH (2 m), compute hourly VPD, then aggregate to daily means.
    Returns DataFrame with columns: lat, lon, date, rh_mean, VPD
    """
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "hourly": "temperature_2m,relative_humidity_2m",
        "timezone": "auto",
    }
    url = f"{HIST_BASE}?{urlencode(params)}"
    js = http_get(url)
    hourly = js.get("hourly", {})
    if not hourly or not hourly.get("time"):
        return pd.DataFrame()
    df = pd.DataFrame(hourly).rename(columns={
        "time": "time",
        "temperature_2m": "t2m",
        "relative_humidity_2m": "rh",
    })
    # Hourly VPD (kPa) using Tetens equation for saturation vapor pressure (es):
    # es(T) = 0.6108 * exp(17.27 * T / (T + 237.3)); VPD = es * (1 - RH/100)
    T = df["t2m"].astype(float)
    es = 0.6108 * np.exp((17.27 * T) / (T + 237.3))
    df["vpd"] = es * (1.0 - df["rh"].astype(float) / 100.0)
    df["date"] = pd.to_datetime(df["time"]).dt.date
    df["lat"], df["lon"] = float(lat), float(lon)

    # Aggregate hourly → daily means
    daily = df.groupby(["lat", "lon", "date"], as_index=False).agg(
        rh_mean=("rh", "mean"),
        VPD=("vpd", "mean"),
    )
    return daily

def compute_indices(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute GDD, EGDD, and SDI from daily Tmax/Tmin.
      - Tmean = (Tmax + Tmin)/2
      - GDD  = max(Tmean - base, 0)
      - EGDD = min(GDD, (upper - base))
      - SDI  = Tmax - Topt
    """
    tmean = (daily_df["tmax"] + daily_df["tmin"]) / 2.0
    daily_df["GDD"] = (tmean - GDD_BASE_C).clip(lower=0.0)
    daily_df["EGDD"] = daily_df["GDD"].clip(upper=max(GDD_UPPER_C - GDD_BASE_C, 0.0))
    daily_df["SDI"] = daily_df["tmax"] - SDI_TOPT_C
    return daily_df

def attach_metrics(input_df: pd.DataFrame, met_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join the meteorological metrics back to the original rows by (lat, lon, date).
    Coordinates are rounded to 4 decimals on both sides to avoid floating-point mismatches.
    """
    inp = input_df.copy()
    inp["_date_"] = pd.to_datetime(inp[DATE_COL], errors="coerce").dt.date
    if inp["_date_"].isna().any():
        bad = inp[inp["_date_"].isna()].head(5)
        raise ValueError(
            f"Unparseable dates in '{DATE_COL}', examples:\n{bad[[DATE_COL]].to_string(index=False)}"
        )
    inp["_lat_"] = inp[LAT_COL].astype(float).round(4)
    inp["_lon_"] = inp[LON_COL].astype(float).round(4)

    met = met_df.copy()
    met["_date_"] = pd.to_datetime(met["date"]).dt.date
    met["_lat_"] = met["lat"].astype(float).round(4)
    met["_lon_"] = met["lon"].astype(float).round(4)

    cols = ["tmax", "tmin", "precip_sum", "VPD", "rh_mean", "GDD", "EGDD", "SDI"]
    merged = inp.merge(
        met[["_date_", "_lat_", "_lon_"] + cols],
        how="left",
        left_on=["_date_", "_lat_", "_lon_"],
        right_on=["_date_", "_lat_", "_lon_"],
    )
    merged.drop(columns=["_date_", "_lat_", "_lon_"], inplace=True)
    return merged

def process_file(csv_path: Path) -> Path:
    """Process a single CSV: fetch met data, compute indices, join, and write output CSV."""
    print(f"\n▶ Processing: {csv_path}")
    df = pd.read_csv(csv_path)
    missing = [c for c in (LAT_COL, LON_COL, DATE_COL) if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in {csv_path.name}: {missing}")

    start, end = daterange_bounds(df[DATE_COL])
    locs = (
        df[[LAT_COL, LON_COL]].astype(float).round(4).drop_duplicates()
        .rename(columns={LAT_COL: "lat", LON_COL: "lon"})
        .reset_index(drop=True)
    )
    print(f"   · Date range: {start} → {end}")
    print(f"   · Unique locations: {len(locs)}")

    all_daily = []
    for i, batch in enumerate(chunked(locs.itertuples(index=False), BATCH_SIZE), start=1):
        print(f"   · Fetching batch {i} ({len(batch)} locations)...")
        for rec in batch:
            lat, lon = float(rec.lat), float(rec.lon)
            dly = fetch_daily(lat, lon, start, end)
            hr = fetch_hourly(lat, lon, start, end)
            if dly.empty:
                continue
            # Join daily + hourly (daily means) on (lat, lon, date)
            joined = dly.merge(hr, how="left", on=["lat", "lon", "date"])
            all_daily.append(joined)
        time.sleep(SLEEP_BETWEEN_BATCHES_S)

    if not all_daily:
        raise RuntimeError("No meteorological data returned (check dates/coords).")

    met_df = pd.concat(all_daily, ignore_index=True)
    met_df = compute_indices(met_df)

    out_df = attach_metrics(df, met_df)
    out_path = csv_path.with_name(csv_path.stem + "_with_meteo.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    return out_path

def main():
    folder = Path(INPUT_DIR)
    if not folder.exists():
        raise FileNotFoundError(f"Input folder not found: {folder}")
    files = sorted([p for p in folder.glob("*.csv") if p.name.endswith(PATTERN_SUFFIX)])
    if not files:
        print(f"No files ending with '{PATTERN_SUFFIX}' in {folder}")
        return 1

    failed = []
    for p in files:
        try:
            process_file(p)
        except Exception as e:
            failed.append((p.name, str(e)))
            print(f"Failed {p.name}: {e}")

    if failed:
        print("Some files failed:")
        for name, err in failed:
            print(f"   - {name}: {err}")
        return 2

    print("All done.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
