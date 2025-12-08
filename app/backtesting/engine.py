import math
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    equity_curve: pd.DataFrame
    trades: List[dict]
    metrics: Dict[str, float | int]


def run_buy_the_dip_backtest(
    df: pd.DataFrame,
    lookback_days: int,
    drop_pct: float,
    max_trades_per_day: int,
    amount_per_trade: float,
    fee_per_trade: float = 0.0,
) -> BacktestResult:
    """
    Buy-the-Dip: kauft, wenn der Kurs heute mindestens drop_pct % niedriger ist
    als vor lookback_days Tagen. Nur Käufe, keine Verkäufe.
    """
    if "Close" not in df.columns:
        raise ValueError("DataFrame benötigt eine 'Close'-Spalte.")

    df = df.sort_index().copy()
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.astype(float)
    dates = pd.to_datetime(close.index)

    n = len(close)
    if n <= lookback_days:
        raise ValueError("Zu wenig Daten für den gewählten Lookback.")

    trades: List[dict] = []
    total_shares = 0.0
    total_invested_cash = 0.0
    total_fees = 0.0

    shares_over_time = np.zeros(n, dtype=float)

    for i in range(lookback_days, n):
        price_today = close.iloc[i]
        price_past = close.iloc[i - lookback_days]

        if price_past <= 0:
            shares_over_time[i] = total_shares
            continue

        drop = (price_today / price_past - 1.0) * 100.0
        trades_today = 0

        if drop <= -drop_pct:
            while trades_today < max_trades_per_day:
                shares = math.floor(amount_per_trade / price_today)
                if shares <= 0:
                    break

                amount_effective = shares * price_today
                total_shares += shares
                total_invested_cash += amount_effective
                total_fees += fee_per_trade

                trades.append(
                    {
                        "date": dates[i],
                        "price": price_today,
                        "shares": shares,
                        "amount": amount_effective,
                        "fee": fee_per_trade,
                    }
                )
                trades_today += 1

        shares_over_time[i] = total_shares

    if len(trades) == 0:
        metrics = {
            "n_trades": 0,
            "total_fees": 0.0,
            "total_invested": 0.0,
            "final_value": 0.0,
            "strategy_return_pct": 0.0,
            "strategy_annual_return_pct": 0.0,
            "bh_return_pct": 0.0,
            "bh_annual_return_pct": 0.0,
            "outperformance_total_pct": 0.0,
            "outperformance_annual_pct": 0.0,
        }
        equity_curve = pd.DataFrame(
            {
                "strategy": np.zeros(n, dtype=float),
                "buy_and_hold": np.zeros(n, dtype=float),
            },
            index=dates,
        )
        return BacktestResult(equity_curve=equity_curve, trades=trades, metrics=metrics)

    strategy_values = shares_over_time * close.to_numpy()

    total_invested = total_invested_cash + total_fees
    last_price = float(close.iloc[-1])
    final_value = float(shares_over_time[-1] * last_price)

    total_return = (final_value / total_invested - 1.0) * 100.0
    duration_days = max(1, (dates[-1] - dates[0]).days)
    strategy_annual = (1.0 + total_return / 100.0) ** (365.0 / duration_days) - 1.0

    first_price = float(close.iloc[0])
    bh_shares = total_invested / first_price
    bh_values = bh_shares * close.to_numpy()

    bh_total_return = (bh_values[-1] / bh_values[0] - 1.0) * 100.0
    bh_annual = (1.0 + bh_total_return / 100.0) ** (365.0 / duration_days) - 1.0

    equity_curve = pd.DataFrame(
        {"strategy": strategy_values, "buy_and_hold": bh_values},
        index=dates,
    )

    metrics: Dict[str, float | int] = {
        "n_trades": len(trades),
        "total_fees": float(total_fees),
        "total_invested": float(total_invested),
        "final_value": float(final_value),
        "strategy_return_pct": float(total_return),
        "strategy_annual_return_pct": float(strategy_annual * 100.0),
        "bh_return_pct": float(bh_total_return),
        "bh_annual_return_pct": float(bh_annual * 100.0),
        "outperformance_total_pct": float(total_return - bh_total_return),
        "outperformance_annual_pct": float(strategy_annual * 100.0 - bh_annual * 100.0),
    }

    return BacktestResult(equity_curve=equity_curve, trades=trades, metrics=metrics)


def run_sma_crossover_backtest(
    df: pd.DataFrame,
    short_window: int,
    long_window: int,
    initial_cash: float = 10_000.0,
    fee_per_trade: float = 0.0,
) -> BacktestResult:
    """Simple all-in/out SMA crossover (buy when short > long, sell when short < long)."""
    if "Close" not in df.columns:
        raise ValueError("DataFrame benötigt eine 'Close'-Spalte.")
    if short_window <= 0 or long_window <= 0 or short_window >= long_window:
        raise ValueError("Kurzfristiges Fenster muss kleiner als langfristiges Fenster sein.")

    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.astype(float)
    prices = close.to_numpy()
    dates = pd.to_datetime(close.index)

    n = len(close)
    if n < long_window:
        raise ValueError("Zu wenig Daten für die gewählten Fenster.")

    short_ma = close.rolling(short_window).mean()
    long_ma = close.rolling(long_window).mean()

    position = 0  # shares
    cash = initial_cash
    total_fees = 0.0
    trades: List[dict] = []
    strategy_values = np.zeros(n, dtype=float)

    for i in range(n):
        price = prices[i]
        if np.isnan(short_ma.iloc[i]) or np.isnan(long_ma.iloc[i]):
            strategy_values[i] = cash + position * price
            continue

        # Buy signal
        if short_ma.iloc[i] > long_ma.iloc[i] and position == 0:
            shares = math.floor(cash / price)
            if shares > 0:
                cost = shares * price
                fee = fee_per_trade
                cash -= cost + fee
                position += shares
                total_fees += fee
                trades.append(
                    {"date": dates[i], "price": price, "shares": shares, "amount": cost, "fee": fee, "side": "buy"}
                )
        # Sell signal
        elif short_ma.iloc[i] < long_ma.iloc[i] and position > 0:
            proceeds = position * price
            fee = fee_per_trade
            cash += proceeds - fee
            total_fees += fee
            trades.append(
                {"date": dates[i], "price": price, "shares": position, "amount": proceeds, "fee": fee, "side": "sell"}
            )
            position = 0

        strategy_values[i] = cash + position * price

    # close any open position at end for metrics (equity already includes unrealized)
    final_value = strategy_values[-1]
    total_invested = initial_cash + total_fees  # fees reduce effective return

    total_return = (final_value / total_invested - 1.0) * 100.0 if total_invested > 0 else 0.0
    duration_days = max(1, (dates[-1] - dates[0]).days)
    strategy_annual = (1.0 + total_return / 100.0) ** (365.0 / duration_days) - 1.0

    # buy & hold benchmark: invest initial cash on day 0
    first_price = float(prices[0])
    bh_shares = initial_cash / first_price if first_price > 0 else 0.0
    bh_values = bh_shares * prices
    bh_total_return = (bh_values[-1] / bh_values[0] - 1.0) * 100.0 if bh_values[0] > 0 else 0.0
    bh_annual = (1.0 + bh_total_return / 100.0) ** (365.0 / duration_days) - 1.0

    equity_curve = pd.DataFrame(
        {"strategy": strategy_values, "buy_and_hold": bh_values},
        index=dates,
    )

    metrics: Dict[str, float | int] = {
        "n_trades": len(trades),
        "total_fees": float(total_fees),
        "total_invested": float(total_invested),
        "final_value": float(final_value),
        "strategy_return_pct": float(total_return),
        "strategy_annual_return_pct": float(strategy_annual * 100.0),
        "bh_return_pct": float(bh_total_return),
        "bh_annual_return_pct": float(bh_annual * 100.0),
        "outperformance_total_pct": float(total_return - bh_total_return),
        "outperformance_annual_pct": float(strategy_annual * 100.0 - bh_annual * 100.0),
    }

    return BacktestResult(equity_curve=equity_curve, trades=trades, metrics=metrics)


def run_dca_backtest(
    df: pd.DataFrame,
    interval_days: int,
    amount_per_trade: float,
    fee_per_trade: float = 0.0,
) -> BacktestResult:
    """Dollar/Euro-cost averaging: invest fixed amount every interval_days."""
    if "Close" not in df.columns:
        raise ValueError("DataFrame benötigt eine 'Close'-Spalte.")
    if interval_days <= 0:
        raise ValueError("Interval_days muss > 0 sein.")

    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.astype(float)
    prices = close.to_numpy()
    dates = pd.to_datetime(close.index)
    n = len(close)

    total_shares = 0.0
    total_fees = 0.0
    trades: List[dict] = []
    strategy_values = np.zeros(n, dtype=float)

    last_invest_idx = 0
    for i in range(n):
        if i == 0 or (i - last_invest_idx) >= interval_days:
            price = prices[i]
            shares = math.floor(amount_per_trade / price)
            if shares > 0:
                cost = shares * price
                fee = fee_per_trade
                total_shares += shares
                total_fees += fee
                trades.append(
                    {"date": dates[i], "price": price, "shares": shares, "amount": cost, "fee": fee, "side": "buy"}
                )
                last_invest_idx = i
        strategy_values[i] = total_shares * prices[i]

    total_invested = len(trades) * amount_per_trade + total_fees
    final_value = strategy_values[-1]
    total_return = (final_value / total_invested - 1.0) * 100.0 if total_invested > 0 else 0.0
    duration_days = max(1, (dates[-1] - dates[0]).days)
    strategy_annual = (1.0 + total_return / 100.0) ** (365.0 / duration_days) - 1.0 if total_invested > 0 else 0.0

    first_price = float(prices[0])
    bh_shares = total_invested / first_price if first_price > 0 else 0.0
    bh_values = bh_shares * prices
    bh_total_return = (bh_values[-1] / bh_values[0] - 1.0) * 100.0 if bh_values[0] > 0 else 0.0
    bh_annual = (1.0 + bh_total_return / 100.0) ** (365.0 / duration_days) - 1.0 if bh_values[0] > 0 else 0.0

    equity_curve = pd.DataFrame(
        {"strategy": strategy_values, "buy_and_hold": bh_values},
        index=dates,
    )

    metrics: Dict[str, float | int] = {
        "n_trades": len(trades),
        "total_fees": float(total_fees),
        "total_invested": float(total_invested),
        "final_value": float(final_value),
        "strategy_return_pct": float(total_return),
        "strategy_annual_return_pct": float(strategy_annual * 100.0),
        "bh_return_pct": float(bh_total_return),
        "bh_annual_return_pct": float(bh_annual * 100.0),
        "outperformance_total_pct": float(total_return - bh_total_return),
        "outperformance_annual_pct": float(strategy_annual * 100.0 - bh_annual * 100.0),
    }

    return BacktestResult(equity_curve=equity_curve, trades=trades, metrics=metrics)


def run_ema_crossover_backtest(
    df: pd.DataFrame,
    short_span: int,
    long_span: int,
    initial_cash: float = 10_000.0,
    fee_per_trade: float = 0.0,
) -> BacktestResult:
    """EMA crossover (buy when short EMA > long EMA, sell when it flips)."""
    if "Close" not in df.columns:
        raise ValueError("DataFrame benötigt eine 'Close'-Spalte.")
    if short_span <= 0 or long_span <= 0 or short_span >= long_span:
        raise ValueError("Short EMA must be >0 and smaller than long EMA.")

    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.astype(float)
    prices = close.to_numpy()
    dates = pd.to_datetime(close.index)
    n = len(close)
    if n < long_span:
        raise ValueError("Zu wenig Daten für die gewählten Spannen.")

    short_ema = close.ewm(span=short_span, adjust=False).mean()
    long_ema = close.ewm(span=long_span, adjust=False).mean()

    position = 0
    cash = initial_cash
    total_fees = 0.0
    trades: List[dict] = []
    strategy_values = np.zeros(n, dtype=float)

    for i in range(n):
        price = prices[i]
        if np.isnan(short_ema.iloc[i]) or np.isnan(long_ema.iloc[i]):
            strategy_values[i] = cash + position * price
            continue
        if short_ema.iloc[i] > long_ema.iloc[i] and position == 0:
            shares = math.floor(cash / price)
            if shares > 0:
                cost = shares * price
                fee = fee_per_trade
                cash -= cost + fee
                position += shares
                total_fees += fee
                trades.append({"date": dates[i], "price": price, "shares": shares, "amount": cost, "fee": fee, "side": "buy"})
        elif short_ema.iloc[i] < long_ema.iloc[i] and position > 0:
            proceeds = position * price
            fee = fee_per_trade
            cash += proceeds - fee
            total_fees += fee
            trades.append({"date": dates[i], "price": price, "shares": position, "amount": proceeds, "fee": fee, "side": "sell"})
            position = 0
        strategy_values[i] = cash + position * price

    final_value = strategy_values[-1]
    total_invested = initial_cash + total_fees
    total_return = (final_value / total_invested - 1.0) * 100.0 if total_invested > 0 else 0.0
    duration_days = max(1, (dates[-1] - dates[0]).days)
    strategy_annual = (1.0 + total_return / 100.0) ** (365.0 / duration_days) - 1.0

    first_price = float(prices[0])
    bh_shares = initial_cash / first_price if first_price > 0 else 0.0
    bh_values = bh_shares * prices
    bh_total_return = (bh_values[-1] / bh_values[0] - 1.0) * 100.0 if bh_values[0] > 0 else 0.0
    bh_annual = (1.0 + bh_total_return / 100.0) ** (365.0 / duration_days) - 1.0

    equity_curve = pd.DataFrame({"strategy": strategy_values, "buy_and_hold": bh_values}, index=dates)
    metrics: Dict[str, float | int] = {
        "n_trades": len(trades),
        "total_fees": float(total_fees),
        "total_invested": float(total_invested),
        "final_value": float(final_value),
        "strategy_return_pct": float(total_return),
        "strategy_annual_return_pct": float(strategy_annual * 100.0),
        "bh_return_pct": float(bh_total_return),
        "bh_annual_return_pct": float(bh_annual * 100.0),
        "outperformance_total_pct": float(total_return - bh_total_return),
        "outperformance_annual_pct": float(strategy_annual * 100.0 - bh_annual * 100.0),
    }
    return BacktestResult(equity_curve=equity_curve, trades=trades, metrics=metrics)


def run_rsi_reversion_backtest(
    df: pd.DataFrame,
    period: int,
    oversold: float,
    overbought: float,
    initial_cash: float = 10_000.0,
    fee_per_trade: float = 0.0,
) -> BacktestResult:
    """RSI reversion: buy when RSI < oversold, sell when RSI > overbought."""
    if "Close" not in df.columns:
        raise ValueError("DataFrame benötigt eine 'Close'-Spalte.")
    if oversold <= 0 or overbought >= 100 or oversold >= overbought:
        raise ValueError("RSI Schwellen ungültig.")

    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.astype(float)
    prices = close.to_numpy()
    dates = pd.to_datetime(close.index)
    n = len(close)
    if n <= period:
        raise ValueError("Zu wenig Daten für RSI.")

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    position = 0
    cash = initial_cash
    total_fees = 0.0
    trades: List[dict] = []
    strategy_values = np.zeros(n, dtype=float)

    for i in range(n):
        price = prices[i]
        r = rsi.iloc[i]
        if np.isnan(r):
            strategy_values[i] = cash + position * price
            continue
        if r < oversold and position == 0:
            shares = math.floor(cash / price)
            if shares > 0:
                cost = shares * price
                fee = fee_per_trade
                cash -= cost + fee
                position += shares
                total_fees += fee
                trades.append({"date": dates[i], "price": price, "shares": shares, "amount": cost, "fee": fee, "side": "buy"})
        elif r > overbought and position > 0:
            proceeds = position * price
            fee = fee_per_trade
            cash += proceeds - fee
            total_fees += fee
            trades.append({"date": dates[i], "price": price, "shares": position, "amount": proceeds, "fee": fee, "side": "sell"})
            position = 0
        strategy_values[i] = cash + position * price

    final_value = strategy_values[-1]
    total_invested = initial_cash + total_fees
    total_return = (final_value / total_invested - 1.0) * 100.0 if total_invested > 0 else 0.0
    duration_days = max(1, (dates[-1] - dates[0]).days)
    strategy_annual = (1.0 + total_return / 100.0) ** (365.0 / duration_days) - 1.0

    first_price = float(prices[0])
    bh_shares = initial_cash / first_price if first_price > 0 else 0.0
    bh_values = bh_shares * prices
    bh_total_return = (bh_values[-1] / bh_values[0] - 1.0) * 100.0 if bh_values[0] > 0 else 0.0
    bh_annual = (1.0 + bh_total_return / 100.0) ** (365.0 / duration_days) - 1.0

    equity_curve = pd.DataFrame({"strategy": strategy_values, "buy_and_hold": bh_values}, index=dates)
    metrics: Dict[str, float | int] = {
        "n_trades": len(trades),
        "total_fees": float(total_fees),
        "total_invested": float(total_invested),
        "final_value": float(final_value),
        "strategy_return_pct": float(total_return),
        "strategy_annual_return_pct": float(strategy_annual * 100.0),
        "bh_return_pct": float(bh_total_return),
        "bh_annual_return_pct": float(bh_annual * 100.0),
        "outperformance_total_pct": float(total_return - bh_total_return),
        "outperformance_annual_pct": float(strategy_annual * 100.0 - bh_annual * 100.0),
    }
    return BacktestResult(equity_curve=equity_curve, trades=trades, metrics=metrics)


def run_bollinger_reversion_backtest(
    df: pd.DataFrame,
    window: int,
    num_std: float,
    initial_cash: float = 10_000.0,
    fee_per_trade: float = 0.0,
) -> BacktestResult:
    """Mean reversion using Bollinger Bands: buy when close < lower band, sell when > upper band."""
    if "Close" not in df.columns:
        raise ValueError("DataFrame benötigt eine 'Close'-Spalte.")
    if window <= 1 or num_std <= 0:
        raise ValueError("Ungültige Bollinger Parameter.")

    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.astype(float)
    prices = close.to_numpy()
    dates = pd.to_datetime(close.index)
    n = len(close)
    if n < window:
        raise ValueError("Zu wenig Daten für Bollinger.")

    ma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std

    position = 0
    cash = initial_cash
    total_fees = 0.0
    trades: List[dict] = []
    strategy_values = np.zeros(n, dtype=float)

    for i in range(n):
        price = prices[i]
        u = upper.iloc[i]
        l = lower.iloc[i]
        if np.isnan(u) or np.isnan(l):
            strategy_values[i] = cash + position * price
            continue
        if price < l and position == 0:
            shares = math.floor(cash / price)
            if shares > 0:
                cost = shares * price
                fee = fee_per_trade
                cash -= cost + fee
                position += shares
                total_fees += fee
                trades.append({"date": dates[i], "price": price, "shares": shares, "amount": cost, "fee": fee, "side": "buy"})
        elif price > u and position > 0:
            proceeds = position * price
            fee = fee_per_trade
            cash += proceeds - fee
            total_fees += fee
            trades.append({"date": dates[i], "price": price, "shares": position, "amount": proceeds, "fee": fee, "side": "sell"})
            position = 0
        strategy_values[i] = cash + position * price

    final_value = strategy_values[-1]
    total_invested = initial_cash + total_fees
    total_return = (final_value / total_invested - 1.0) * 100.0 if total_invested > 0 else 0.0
    duration_days = max(1, (dates[-1] - dates[0]).days)
    strategy_annual = (1.0 + total_return / 100.0) ** (365.0 / duration_days) - 1.0

    first_price = float(prices[0])
    bh_shares = initial_cash / first_price if first_price > 0 else 0.0
    bh_values = bh_shares * prices
    bh_total_return = (bh_values[-1] / bh_values[0] - 1.0) * 100.0 if bh_values[0] > 0 else 0.0
    bh_annual = (1.0 + bh_total_return / 100.0) ** (365.0 / duration_days) - 1.0

    equity_curve = pd.DataFrame({"strategy": strategy_values, "buy_and_hold": bh_values}, index=dates)
    metrics: Dict[str, float | int] = {
        "n_trades": len(trades),
        "total_fees": float(total_fees),
        "total_invested": float(total_invested),
        "final_value": float(final_value),
        "strategy_return_pct": float(total_return),
        "strategy_annual_return_pct": float(strategy_annual * 100.0),
        "bh_return_pct": float(bh_total_return),
        "bh_annual_return_pct": float(bh_annual * 100.0),
        "outperformance_total_pct": float(total_return - bh_total_return),
        "outperformance_annual_pct": float(strategy_annual * 100.0 - bh_annual * 100.0),
    }
    return BacktestResult(equity_curve=equity_curve, trades=trades, metrics=metrics)


def run_momentum_12_1_backtest(
    df: pd.DataFrame,
    initial_cash: float = 10_000.0,
    fee_per_trade: float = 0.0,
) -> BacktestResult:
    """12-1 momentum: invest all when 12m-1m momentum positive, else stay flat."""
    if "Close" not in df.columns:
        raise ValueError("DataFrame benötigt eine 'Close'-Spalte.")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.astype(float)
    dates = pd.to_datetime(close.index)
    n = len(close)
    if n < 260:
        raise ValueError("Zu wenig Daten für 12-1 Momentum.")

    prices = close.to_numpy()
    mom = close.shift(21) / close.shift(252) - 1.0  # skip last month

    position = 0
    cash = initial_cash
    total_fees = 0.0
    trades: List[dict] = []
    strategy_values = np.zeros(n, dtype=float)

    for i in range(n):
        price = prices[i]
        m = mom.iloc[i]
        if np.isnan(m):
            strategy_values[i] = cash + position * price
            continue
        if m > 0 and position == 0:
            shares = math.floor(cash / price)
            if shares > 0:
                cost = shares * price
                fee = fee_per_trade
                cash -= cost + fee
                total_fees += fee
                position += shares
                trades.append({"date": dates[i], "price": price, "shares": shares, "amount": cost, "fee": fee, "side": "buy"})
        elif m <= 0 and position > 0:
            proceeds = position * price
            fee = fee_per_trade
            cash += proceeds - fee
            total_fees += fee
            trades.append({"date": dates[i], "price": price, "shares": position, "amount": proceeds, "fee": fee, "side": "sell"})
            position = 0
        strategy_values[i] = cash + position * price

    final_value = strategy_values[-1]
    total_invested = initial_cash + total_fees
    total_return = (final_value / total_invested - 1.0) * 100.0 if total_invested > 0 else 0.0
    duration_days = max(1, (dates[-1] - dates[0]).days)
    strategy_annual = (1.0 + total_return / 100.0) ** (365.0 / duration_days) - 1.0

    first_price = float(prices[0])
    bh_shares = initial_cash / first_price if first_price > 0 else 0.0
    bh_values = bh_shares * prices
    bh_total_return = (bh_values[-1] / bh_values[0] - 1.0) * 100.0 if bh_values[0] > 0 else 0.0
    bh_annual = (1.0 + bh_total_return / 100.0) ** (365.0 / duration_days) - 1.0

    equity_curve = pd.DataFrame({"strategy": strategy_values, "buy_and_hold": bh_values}, index=dates)
    metrics: Dict[str, float | int] = {
        "n_trades": len(trades),
        "total_fees": float(total_fees),
        "total_invested": float(total_invested),
        "final_value": float(final_value),
        "strategy_return_pct": float(total_return),
        "strategy_annual_return_pct": float(strategy_annual * 100.0),
        "bh_return_pct": float(bh_total_return),
        "bh_annual_return_pct": float(bh_annual * 100.0),
        "outperformance_total_pct": float(total_return - bh_total_return),
        "outperformance_annual_pct": float(strategy_annual * 100.0 - bh_annual * 100.0),
    }
    return BacktestResult(equity_curve=equity_curve, trades=trades, metrics=metrics)


def run_donchian_breakout_backtest(
    df: pd.DataFrame,
    window: int,
    initial_cash: float = 10_000.0,
    fee_per_trade: float = 0.0,
) -> BacktestResult:
    """Donchian breakout: buy on breakout above window high, sell on breakdown below window low."""
    if "Close" not in df.columns:
        raise ValueError("DataFrame benötigt eine 'Close'-Spalte.")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.astype(float)
    prices = close.to_numpy()
    dates = pd.to_datetime(close.index)
    n = len(close)
    if n < window:
        raise ValueError("Zu wenig Daten für Donchian.")

    highs = close.rolling(window).max()
    lows = close.rolling(window).min()

    position = 0
    cash = initial_cash
    total_fees = 0.0
    trades: List[dict] = []
    strategy_values = np.zeros(n, dtype=float)

    for i in range(n):
        price = prices[i]
        hi = highs.iloc[i]
        lo = lows.iloc[i]
        if np.isnan(hi) or np.isnan(lo):
            strategy_values[i] = cash + position * price
            continue
        if price > hi and position == 0:
            shares = math.floor(cash / price)
            if shares > 0:
                cost = shares * price
                fee = fee_per_trade
                cash -= cost + fee
                total_fees += fee
                position += shares
                trades.append({"date": dates[i], "price": price, "shares": shares, "amount": cost, "fee": fee, "side": "buy"})
        elif price < lo and position > 0:
            proceeds = position * price
            fee = fee_per_trade
            cash += proceeds - fee
            total_fees += fee
            trades.append({"date": dates[i], "price": price, "shares": position, "amount": proceeds, "fee": fee, "side": "sell"})
            position = 0
        strategy_values[i] = cash + position * price

    final_value = strategy_values[-1]
    total_invested = initial_cash + total_fees
    total_return = (final_value / total_invested - 1.0) * 100.0 if total_invested > 0 else 0.0
    duration_days = max(1, (dates[-1] - dates[0]).days)
    strategy_annual = (1.0 + total_return / 100.0) ** (365.0 / duration_days) - 1.0

    first_price = float(prices[0])
    bh_shares = initial_cash / first_price if first_price > 0 else 0.0
    bh_values = bh_shares * prices
    bh_total_return = (bh_values[-1] / bh_values[0] - 1.0) * 100.0 if bh_values[0] > 0 else 0.0
    bh_annual = (1.0 + bh_total_return / 100.0) ** (365.0 / duration_days) - 1.0

    equity_curve = pd.DataFrame({"strategy": strategy_values, "buy_and_hold": bh_values}, index=dates)
    metrics: Dict[str, float | int] = {
        "n_trades": len(trades),
        "total_fees": float(total_fees),
        "total_invested": float(total_invested),
        "final_value": float(final_value),
        "strategy_return_pct": float(total_return),
        "strategy_annual_return_pct": float(strategy_annual * 100.0),
        "bh_return_pct": float(bh_total_return),
        "bh_annual_return_pct": float(bh_annual * 100.0),
        "outperformance_total_pct": float(total_return - bh_total_return),
        "outperformance_annual_pct": float(strategy_annual * 100.0 - bh_annual * 100.0),
    }
    return BacktestResult(equity_curve=equity_curve, trades=trades, metrics=metrics)


def run_macd_backtest(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    initial_cash: float = 10_000.0,
    fee_per_trade: float = 0.0,
) -> BacktestResult:
    """MACD crossover: buy when MACD>signal, sell when MACD<signal."""
    if "Close" not in df.columns:
        raise ValueError("DataFrame benötigt eine 'Close'-Spalte.")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.astype(float)
    prices = close.to_numpy()
    dates = pd.to_datetime(close.index)

    macd = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    sig = macd.ewm(span=signal, adjust=False).mean()

    position = 0
    cash = initial_cash
    total_fees = 0.0
    trades: List[dict] = []
    strategy_values = np.zeros(len(close), dtype=float)

    for i in range(len(close)):
        price = prices[i]
        m = macd.iloc[i]
        s = sig.iloc[i]
        if np.isnan(m) or np.isnan(s):
            strategy_values[i] = cash + position * price
            continue
        if m > s and position == 0:
            shares = math.floor(cash / price)
            if shares > 0:
                cost = shares * price
                fee = fee_per_trade
                cash -= cost + fee
                total_fees += fee
                position += shares
                trades.append({"date": dates[i], "price": price, "shares": shares, "amount": cost, "fee": fee, "side": "buy"})
        elif m < s and position > 0:
            proceeds = position * price
            fee = fee_per_trade
            cash += proceeds - fee
            total_fees += fee
            trades.append({"date": dates[i], "price": price, "shares": position, "amount": proceeds, "fee": fee, "side": "sell"})
            position = 0
        strategy_values[i] = cash + position * price

    final_value = strategy_values[-1]
    total_invested = initial_cash + total_fees
    total_return = (final_value / total_invested - 1.0) * 100.0 if total_invested > 0 else 0.0
    duration_days = max(1, (dates[-1] - dates[0]).days)
    strategy_annual = (1.0 + total_return / 100.0) ** (365.0 / duration_days) - 1.0

    first_price = float(prices[0])
    bh_shares = initial_cash / first_price if first_price > 0 else 0.0
    bh_values = bh_shares * prices
    bh_total_return = (bh_values[-1] / bh_values[0] - 1.0) * 100.0 if bh_values[0] > 0 else 0.0
    bh_annual = (1.0 + bh_total_return / 100.0) ** (365.0 / duration_days) - 1.0

    equity_curve = pd.DataFrame({"strategy": strategy_values, "buy_and_hold": bh_values}, index=dates)
    metrics: Dict[str, float | int] = {
        "n_trades": len(trades),
        "total_fees": float(total_fees),
        "total_invested": float(total_invested),
        "final_value": float(final_value),
        "strategy_return_pct": float(total_return),
        "strategy_annual_return_pct": float(strategy_annual * 100.0),
        "bh_return_pct": float(bh_total_return),
        "bh_annual_return_pct": float(bh_annual * 100.0),
        "outperformance_total_pct": float(total_return - bh_total_return),
        "outperformance_annual_pct": float(strategy_annual * 100.0 - bh_annual * 100.0),
    }
    return BacktestResult(equity_curve=equity_curve, trades=trades, metrics=metrics)


def run_stoch_osc_backtest(
    df: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
    oversold: float = 20.0,
    overbought: float = 80.0,
    initial_cash: float = 10_000.0,
    fee_per_trade: float = 0.0,
) -> BacktestResult:
    """Stochastic oscillator: buy when %K crosses above %D below oversold; sell when crosses below above overbought."""
    if "Close" not in df.columns or "High" not in df.columns or "Low" not in df.columns:
        raise ValueError("DataFrame benötigt 'Close', 'High', 'Low'.")
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.astype(float)
    dates = pd.to_datetime(close.index)

    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = (close - lowest_low) / (highest_high - lowest_low) * 100.0
    d = k.rolling(d_period).mean()

    prices = close.to_numpy()
    position = 0
    cash = initial_cash
    total_fees = 0.0
    trades: List[dict] = []
    strategy_values = np.zeros(len(close), dtype=float)

    prev_k = np.nan
    prev_d = np.nan
    for i in range(len(close)):
        price = prices[i]
        ki = k.iloc[i]
        di = d.iloc[i]
        if np.isnan(ki) or np.isnan(di) or np.isnan(prev_k) or np.isnan(prev_d):
            strategy_values[i] = cash + position * price
            prev_k, prev_d = ki, di
            continue
        # cross above in oversold region
        if prev_k < prev_d and ki > di and ki < oversold and position == 0:
            shares = math.floor(cash / price)
            if shares > 0:
                cost = shares * price
                fee = fee_per_trade
                cash -= cost + fee
                total_fees += fee
                position += shares
                trades.append({"date": dates[i], "price": price, "shares": shares, "amount": cost, "fee": fee, "side": "buy"})
        # cross below in overbought region
        elif prev_k > prev_d and ki < di and ki > overbought and position > 0:
            proceeds = position * price
            fee = fee_per_trade
            cash += proceeds - fee
            total_fees += fee
            trades.append({"date": dates[i], "price": price, "shares": position, "amount": proceeds, "fee": fee, "side": "sell"})
            position = 0
        strategy_values[i] = cash + position * price
        prev_k, prev_d = ki, di

    final_value = strategy_values[-1]
    total_invested = initial_cash + total_fees
    total_return = (final_value / total_invested - 1.0) * 100.0 if total_invested > 0 else 0.0
    duration_days = max(1, (dates[-1] - dates[0]).days)
    strategy_annual = (1.0 + total_return / 100.0) ** (365.0 / duration_days) - 1.0

    first_price = float(prices[0])
    bh_shares = initial_cash / first_price if first_price > 0 else 0.0
    bh_values = bh_shares * prices
    bh_total_return = (bh_values[-1] / bh_values[0] - 1.0) * 100.0 if bh_values[0] > 0 else 0.0
    bh_annual = (1.0 + bh_total_return / 100.0) ** (365.0 / duration_days) - 1.0

    equity_curve = pd.DataFrame({"strategy": strategy_values, "buy_and_hold": bh_values}, index=dates)
    metrics: Dict[str, float | int] = {
        "n_trades": len(trades),
        "total_fees": float(total_fees),
        "total_invested": float(total_invested),
        "final_value": float(final_value),
        "strategy_return_pct": float(total_return),
        "strategy_annual_return_pct": float(strategy_annual * 100.0),
        "bh_return_pct": float(bh_total_return),
        "bh_annual_return_pct": float(bh_annual * 100.0),
        "outperformance_total_pct": float(total_return - bh_total_return),
        "outperformance_annual_pct": float(strategy_annual * 100.0 - bh_annual * 100.0),
    }
    return BacktestResult(equity_curve=equity_curve, trades=trades, metrics=metrics)
