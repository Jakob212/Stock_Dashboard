import pandas as pd

from app.backtesting.engine import BacktestResult, run_macd_backtest


def macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    initial_cash: float = 10_000.0,
    fee_per_trade: float = 0.0,
) -> BacktestResult:
    return run_macd_backtest(
        df=df,
        fast=fast,
        slow=slow,
        signal=signal,
        initial_cash=initial_cash,
        fee_per_trade=fee_per_trade,
    )
