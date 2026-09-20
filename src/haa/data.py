"""Yahoo Finance acquisition and transparent daily/month-end data preparation."""
from __future__ import annotations

from io import BytesIO
from typing import Mapping

import pandas as pd

from .constants import ASSETS


def _normalise_frame(frame: pd.DataFrame, asset: str) -> pd.Series:
    """Extract an adjusted close series, falling back to Close only if necessary."""
    if isinstance(frame.columns, pd.MultiIndex):
        # yfinance can return either (field, ticker) or (ticker, field).
        for field in ("Adj Close", "Close"):
            if field in frame.columns.get_level_values(0):
                candidate = frame[field]
                return candidate[asset] if isinstance(candidate, pd.DataFrame) else candidate
            if field in frame.columns.get_level_values(1):
                return frame.xs(field, axis=1, level=1)[asset]
    for field in ("Adj Close", "Close"):
        if field in frame.columns:
            return frame[field]
    raise ValueError(f"{asset}: CSV/data must include 'Adj Close' or 'Close'.")


def download_yahoo_prices(assets: tuple[str, ...] = ASSETS) -> pd.DataFrame:
    """Download all canonical assets; Yahoo's adjusted close is preferred."""
    import yfinance as yf

    raw = yf.download(list(assets), period="max", auto_adjust=False, progress=False)
    if raw.empty:
        raise RuntimeError("Yahoo Finance returned no data. Try again or upload CSV files.")
    result = pd.DataFrame({asset: _normalise_frame(raw, asset) for asset in assets})
    return _clean_prices(result)


def read_uploaded_csv(content: bytes, asset: str) -> pd.Series:
    """Read a validation replacement CSV with Date plus Adj Close or Close."""
    frame = pd.read_csv(BytesIO(content))
    date_col = next((c for c in frame.columns if c.lower() in {"date", "datetime"}), None)
    if date_col is None:
        raise ValueError(f"{asset}: CSV requires a Date column.")
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame = frame.dropna(subset=[date_col]).set_index(date_col).sort_index()
    series = _normalise_frame(frame, asset).rename(asset)
    return _clean_prices(series.to_frame())[asset]


def combine_replacements(downloaded: pd.DataFrame, replacements: Mapping[str, pd.Series]) -> pd.DataFrame:
    """Replace whole asset histories supplied for validation; never fill invented data."""
    combined = downloaded.copy()
    for asset, series in replacements.items():
        combined = combined.drop(columns=asset, errors="ignore").join(series.rename(asset), how="outer")
    return _clean_prices(combined.reindex(columns=ASSETS))


def _clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    result = prices.copy()
    result.index = pd.to_datetime(result.index).tz_localize(None)
    result = result[~result.index.duplicated(keep="last")].sort_index()
    return result.apply(pd.to_numeric, errors="coerce")


def to_month_end(daily_prices: pd.DataFrame) -> pd.DataFrame:
    """Use the final available trading observation in each calendar month."""
    return daily_prices.resample("ME").last()


def date_ranges(prices: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for asset in prices.columns:
        valid = prices[asset].dropna()
        rows.append({"asset": asset, "first_available": valid.index.min(), "last_available": valid.index.max(), "observations": len(valid)})
    return pd.DataFrame(rows).set_index("asset")


def common_monthly_period(monthly_prices: pd.DataFrame) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    complete = monthly_prices.dropna(how="any")
    if complete.empty:
        return None, None
    return complete.index.min(), complete.index.max()
