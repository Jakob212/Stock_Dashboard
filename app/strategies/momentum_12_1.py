import pandas as pd

from app.backtesting.engine import BacktestResult, run_momentum_12_1_backtest


def momentum_12_1(
    df: pd.DataFrame,
    initial_cash: float = 10_000.0,
    fee_per_trade: float = 0.0,
) -> BacktestResult:
    return run_momentum_12_1_backtest(df=df, initial_cash=initial_cash, fee_per_trade=fee_per_trade)
