import pandas as pd

from app.backtesting.engine import BacktestResult, run_bollinger_reversion_backtest


def bollinger_reversion(
    df: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0,
    initial_cash: float = 10_000.0,
    fee_per_trade: float = 0.0,
) -> BacktestResult:
    return run_bollinger_reversion_backtest(
        df=df,
        window=window,
        num_std=num_std,
        initial_cash=initial_cash,
        fee_per_trade=fee_per_trade,
    )
