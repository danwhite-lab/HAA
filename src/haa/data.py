"""Yahoo Finance acquisition and transparent daily/month-end data preparation."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import re
from typing import Iterable, Mapping

import pandas as pd

from .constants import ASSETS

DEFAULT_TICKER_MAP = {asset: asset for asset in ASSETS}


def default_ticker_map(assets: Iterable[str]) -> dict[str, str]:
    return {asset: asset for asset in assets}


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


def parse_ticker_map(text: str, required_assets: Iterable[str] = ASSETS) -> dict[str, str]:
    """Parse canonical-role-to-Yahoo-ticker entries such as ``SPY=SPY``.

    Roles stay fixed for HAA-Simple. The editable ticker is only a transparent
    data-source replacement for that role, useful for validation.
    """
    required = tuple(required_assets)
    mapping: dict[str, str] = {}
    entries = [entry.strip() for entry in re.split(r"[,\n]+", text) if entry.strip()]
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Use ROLE=TICKER entries, for example SPY=SPY; received '{entry}'.")
        role, ticker = (part.strip().upper() for part in entry.split("=", 1))
        if role not in required:
            raise ValueError(f"'{role}' is not valid for this model. Use only: {', '.join(required)}.")
        if not ticker:
            raise ValueError(f"{role}: Yahoo ticker cannot be empty.")
        if role in mapping:
            raise ValueError(f"{role} appears more than once.")
        mapping[role] = ticker
    missing = set(required) - set(mapping)
    if missing:
        raise ValueError(f"Missing Yahoo ticker mapping for: {', '.join(sorted(missing))}.")
    return mapping


def upload_asset_from_filename(filename: str, allowed_assets: Iterable[str] = ASSETS) -> str:
    """Resolve the canonical target role from a CSV filename, e.g. ``SPY.csv``."""
    tokens = set(filter(None, re.split(r"[^A-Z0-9]+", Path(filename).stem.upper())))
    allowed = tuple(allowed_assets)
    matches = tokens.intersection(allowed)
    if len(matches) != 1:
        raise ValueError(f"{filename}: name the file with exactly one valid asset ({', '.join(allowed)}), e.g. SPY.csv.")
    return matches.pop()


def download_yahoo_prices(ticker_map: Mapping[str, str] | None = None) -> pd.DataFrame:
    """Download each canonical role from its selected Yahoo ticker source."""
    import yfinance as yf

    sources = dict(ticker_map or DEFAULT_TICKER_MAP)
    missing = set(ASSETS) - set(sources)
    if missing:
        raise ValueError(f"Missing Yahoo ticker mapping for: {', '.join(sorted(missing))}.")
    raw = yf.download(list(sources.values()), period="max", auto_adjust=False, progress=False)
    if raw.empty:
        raise RuntimeError("Yahoo Finance returned no data. Try again or upload CSV files.")
    result = pd.DataFrame({asset: _normalise_frame(raw, ticker) for asset, ticker in sources.items()})
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


def combine_replacements(downloaded: pd.DataFrame, replacements: Mapping[str, pd.Series], assets: Iterable[str] = ASSETS) -> pd.DataFrame:
    """Replace whole asset histories supplied for validation; never fill invented data."""
    combined = downloaded.copy()
    for asset, series in replacements.items():
        combined = combined.drop(columns=asset, errors="ignore").join(series.rename(asset), how="outer")
    return _clean_prices(combined.reindex(columns=tuple(assets)))


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
