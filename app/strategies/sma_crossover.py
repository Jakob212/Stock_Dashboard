import pandas as pd

from app.backtesting.engine import BacktestResult, run_sma_crossover_backtest


def sma_crossover(
    df: pd.DataFrame,
    short_window: int = 20,
    long_window: int = 50,
    initial_cash: float = 10_000.0,
    fee_per_trade: float = 0.0,
) -> BacktestResult:
    return run_sma_crossover_backtest(
        df=df,
        short_window=short_window,
        long_window=long_window,
        initial_cash=initial_cash,
        fee_per_trade=fee_per_trade,
    )
