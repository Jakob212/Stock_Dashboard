import pandas as pd

from app.backtesting.engine import BacktestResult, run_rsi_reversion_backtest


def rsi_reversion(
    df: pd.DataFrame,
    period: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
    initial_cash: float = 10_000.0,
    fee_per_trade: float = 0.0,
) -> BacktestResult:
    return run_rsi_reversion_backtest(
        df=df,
        period=period,
        oversold=oversold,
        overbought=overbought,
        initial_cash=initial_cash,
        fee_per_trade=fee_per_trade,
    )
