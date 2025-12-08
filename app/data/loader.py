import datetime as dt
from typing import Optional

import pandas as pd
import yfinance as yf


def load_price_data(
    ticker: str,
    start: Optional[dt.date] = None,
    end: Optional[dt.date] = None,
) -> pd.DataFrame:
    """
    Load historical OHLCV data for a ticker using yfinance.
    Default: maximum available history (period="max", auto-adjusted).
    Falls back to start/end when explicitly provided.
    """
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("Ticker must not be empty.")

    df: pd.DataFrame
    if start is None and end is None:
        t = yf.Ticker(ticker)
        df = t.history(period="max", auto_adjust=True)
    else:
        if end is None:
            end = dt.date.today()
        df = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )

    if df is None or df.empty:
        raise ValueError(f"No data found for ticker '{ticker}'.")

    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    df.index = pd.to_datetime(df.index)

    cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    if not cols:
        raise ValueError("Downloaded data does not contain OHLCV columns.")
    df = df[cols]

    return df


def basic_metrics(df: pd.DataFrame) -> dict:
    """
    Compute a few basic metrics. Requires a 'Close' column.
    """
    if "Close" not in df.columns:
        raise ValueError("DataFrame does not contain a 'Close' column.")

    close = df["Close"].dropna()
    if len(close) < 2:
        raise ValueError("Not enough datapoints for metrics.")

    start_price = float(close.iloc[0].item())
    end_price = float(close.iloc[-1].item())

    total_return = (end_price / start_price - 1.0) * 100.0

    daily_returns = close.pct_change().dropna()
    volatility = float(daily_returns.std().item() * (252.0**0.5) * 100.0)

    metrics = {
        "start_date": df.index[0].date(),
        "end_date": df.index[-1].date(),
        "start_price": start_price,
        "end_price": end_price,
        "total_return": total_return,
        "volatility": volatility,
        "num_points": int(len(df)),
    }

    return metrics


def load_dividends(ticker: str) -> pd.DataFrame:
    """
    Load dividend history for the ticker.
    Returns a DataFrame with index=dividend dates and column 'Dividend'.
    """
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("Ticker must not be empty.")

    t = yf.Ticker(ticker)
    div = t.dividends

    if div is None or len(div) == 0:
        return pd.DataFrame(columns=["Dividend"])

    df = div.to_frame(name="Dividend")

    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    df.index = pd.to_datetime(df.index)

    return df


def load_splits(ticker: str) -> pd.DataFrame:
    """
    Load stock split history for the ticker.
    Returns a DataFrame with index=dates and column 'Split'.
    """
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("Ticker must not be empty.")

    t = yf.Ticker(ticker)
    splits = t.splits

    if splits is None or len(splits) == 0:
        return pd.DataFrame(columns=["Split"])

    df = splits.to_frame(name="Split")

    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    df.index = pd.to_datetime(df.index)

    return df


def load_earnings_dates(ticker: str, limit: int = 20) -> pd.DataFrame:
    """
    Load upcoming/past earnings dates for the ticker.
    Returns DataFrame indexed by earnings date with a column 'Earnings' (reported EPS when available).
    """
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("Ticker must not be empty.")

    t = yf.Ticker(ticker)
    try:
        ed = t.get_earnings_dates(limit=limit)
    except Exception:
        return pd.DataFrame(columns=["Earnings"])

    if ed is None or ed.empty:
        return pd.DataFrame(columns=["Earnings"])

    df = ed.copy()
    # Prefer reported EPS; fall back to first numeric column
    if "Reported EPS" in df.columns:
        df["Earnings"] = df["Reported EPS"]
    elif "EPS Estimate" in df.columns:
        df["Earnings"] = df["EPS Estimate"]
    else:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        df["Earnings"] = df[numeric_cols[0]] if numeric_cols else pd.Series(dtype=float)

    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    df.index = pd.to_datetime(df.index)

    return df[["Earnings"]]


def load_actions(ticker: str) -> pd.DataFrame:
    """
    Load combined corporate actions (dividends, splits) from yfinance actions endpoint.
    Returns DataFrame with columns such as 'Dividends' and 'Stock Splits'.
    """
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("Ticker must not be empty.")

    t = yf.Ticker(ticker)
    try:
        actions = t.actions
    except Exception:
        return pd.DataFrame()

    if actions is None or actions.empty:
        return pd.DataFrame()

    df = actions.copy()
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    df.index = pd.to_datetime(df.index)
    return df
