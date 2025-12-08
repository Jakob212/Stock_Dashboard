import pandas as pd

from app.backtesting.engine import BacktestResult, run_dca_backtest


def dca(
    df: pd.DataFrame,
    interval_days: int = 30,
    amount_per_trade: float = 500.0,
    fee_per_trade: float = 0.0,
) -> BacktestResult:
    return run_dca_backtest(
        df=df,
        interval_days=interval_days,
        amount_per_trade=amount_per_trade,
        fee_per_trade=fee_per_trade,
    )
