import pandas as pd

from app.backtesting.engine import BacktestResult, run_stoch_osc_backtest


def stochastic(
    df: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
    oversold: float = 20.0,
    overbought: float = 80.0,
    initial_cash: float = 10_000.0,
    fee_per_trade: float = 0.0,
) -> BacktestResult:
    return run_stoch_osc_backtest(
        df=df,
        k_period=k_period,
        d_period=d_period,
        oversold=oversold,
        overbought=overbought,
        initial_cash=initial_cash,
        fee_per_trade=fee_per_trade,
    )
