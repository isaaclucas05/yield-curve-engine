# Data ingestion pipeline for government bond par yield curves.
# Pulls US Constant Maturity Treasury (CMT) series directly from St. Louis Fed (FRED).

import os
import numpy as np
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor

# Canonical maturity mapping (Tenor Label -> Maturity in Years)
STANDARD_TENOR_MAP = {
    "1M": 1 / 12,
    "3M": 3 / 12,
    "6M": 6 / 12,
    "1Y": 1.0,
    "2Y": 2.0,
    "3Y": 3.0,
    "5Y": 5.0,
    "7Y": 7.0,
    "10Y": 10.0,
    "20Y": 20.0,
    "30Y": 30.0,
}

# Mapping St. Louis Fed FRED tickers to canonical tenors
FRED_TREASURY_SERIES = {
    "DGS1MO": "1M",
    "DGS3MO": "3M",
    "DGS6MO": "6M",
    "DGS1": "1Y",
    "DGS2": "2Y",
    "DGS3": "3Y",
    "DGS5": "5Y",
    "DGS7": "7Y",
    "DGS10": "10Y",
    "DGS20": "20Y",
    "DGS30": "30Y",
}


# Clean container holding aligned yield curves and maturities
class CurveDataContainer:
    def __init__(self, par_yields, tenors, currency="USD", source="FRED"):
        self.par_yields = par_yields
        self.tenors = np.asarray(tenors, dtype=float)
        self.currency = currency
        self.source = source


# Base loader contract
class BaseYieldCurveDataLoader:
    def fetch_raw_data(self, start_date, end_date):
        raise NotImplementedError

    def clean_and_align(self, raw_df, max_ffill_days=3):
        raise NotImplementedError

    def run_pipeline(self, start_date, end_date):
        raw = self.fetch_raw_data(start_date, end_date)
        return self.clean_and_align(raw)


# Main FRED REST loader
class FREDYieldCurveLoader(BaseYieldCurveDataLoader):
    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(self, api_key):
        if not api_key:
            raise ValueError("FRED_API_KEY is missing!")
        self.api_key = api_key

    def _fetch_series(self, series_id, start_date, end_date):
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start_date,
            "observation_end": end_date,
        }
        resp = requests.get(self.BASE_URL, params=params, timeout=12)
        if resp.status_code != 200:
            raise RuntimeError(
                f"FRED API request failed for {series_id} with status {resp.status_code}: {resp.text}"
            )

        data = resp.json().get("observations", [])
        if not data:
            raise ValueError(f"No observations returned for {series_id} between {start_date} and {end_date}.")

        dates = [obs["date"] for obs in data]
        vals = [obs["value"] for obs in data]

        # FRED represents market holidays with '.' which coerce cleanly to NaN
        s = pd.Series(vals, index=pd.to_datetime(dates), name=series_id)
        return pd.to_numeric(s, errors="coerce")

    def fetch_raw_data(self, start_date, end_date):
        tickers = list(FRED_TREASURY_SERIES.keys())
        
        # Pull series concurrently across worker threads to avoid serial latency bottlenecks
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = [
                pool.submit(self._fetch_series, tick, start_date, end_date)
                for tick in tickers
            ]
            results = [f.result() for f in futures]

        raw_df = pd.concat(results, axis=1)
        raw_df.index.name = "date"
        return raw_df.sort_index()

    def clean_and_align(self, raw_df, max_ffill_days=3):
        # Scale par yields to decimal rates (e.g., 4.5% -> 0.045)
        df_clean = raw_df.copy() / 100.0

        # Rename tickers to fractional year tenors and sort left to right
        ticker_to_tenor = {
            ticker: STANDARD_TENOR_MAP[tenor_label]
            for ticker, tenor_label in FRED_TREASURY_SERIES.items()
        }
        df_clean = df_clean.rename(columns=ticker_to_tenor)
        df_clean = df_clean.reindex(sorted(df_clean.columns), axis=1)

        # Align against standard business days calendar
        b_days = pd.bdate_range(start=df_clean.index.min(), end=df_clean.index.max(), name="date")
        df_clean = df_clean.reindex(b_days)

        # Forward-fill short holiday gaps, then drop rows where full curve isn't populated
        df_clean = df_clean.ffill(limit=max_ffill_days)
        df_clean = df_clean.dropna(how="any")

        return CurveDataContainer(
            par_yields=df_clean,
            tenors=np.array(df_clean.columns, dtype=float),
            currency="USD",
            source="FRED",
        )

    # Pulls and business-day-aligns arbitrary economic or inflation series
    def fetch_custom_series(self, series_id, start_date, end_date):
        cleaned = self._fetch_series(series_id, start_date, end_date).dropna()
        all_b_days = pd.bdate_range(start=cleaned.index.min(), end=cleaned.index.max())
        cleaned = cleaned.reindex(all_b_days).ffill(limit=5)
        return cleaned / 100.0


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("FRED_API_KEY")

    if not api_key:
        raise ValueError("FRED_API_KEY is missing from your .env file!")

    loader = FREDYieldCurveLoader(api_key=api_key)

    # Execute the complete template pipeline
    print("Running end-to-end data pipeline (fetch + clean + align)...")
    curve_data = loader.run_pipeline(start_date="2024-01-01", end_date="2025-10-01")

    # Verification block: check sample records and curve boundaries
    print("\n" + "=" * 60)
    print("PROCESSED CURVE DATA CONTAINER VERIFICATION")
    print("=" * 60)
    print(f"Currency: {curve_data.currency} | Source: {curve_data.source}")
    print(f"Total Trading Days Ingested: {len(curve_data.par_yields)}")
    print(f"Date Range: {curve_data.par_yields.index.min().date()} -> {curve_data.par_yields.index.max().date()}")

    print("\n--- FIRST 3 ROWS (.head) ---")
    print(curve_data.par_yields.head(3))

    print("\n--- LAST 3 ROWS (.tail) ---")
    print(curve_data.par_yields.tail(3))