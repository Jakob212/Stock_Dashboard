import pandas as pd

from app.backtesting.engine import BacktestResult, run_donchian_breakout_backtest


def donchian_breakout(
    df: pd.DataFrame,
    window: int = 55,
    initial_cash: float = 10_000.0,
    fee_per_trade: float = 0.0,
) -> BacktestResult:
    return run_donchian_breakout_backtest(
        df=df,
        window=window,
        initial_cash=initial_cash,
        fee_per_trade=fee_per_trade,
    )
