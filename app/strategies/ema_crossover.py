import pandas as pd

from app.backtesting.engine import BacktestResult, run_ema_crossover_backtest


def ema_crossover(
    df: pd.DataFrame,
    short_span: int = 12,
    long_span: int = 26,
    initial_cash: float = 10_000.0,
    fee_per_trade: float = 0.0,
) -> BacktestResult:
    return run_ema_crossover_backtest(
        df=df,
        short_span=short_span,
        long_span=long_span,
        initial_cash=initial_cash,
        fee_per_trade=fee_per_trade,
    )
