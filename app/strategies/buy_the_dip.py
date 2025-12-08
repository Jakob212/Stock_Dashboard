from typing import Optional

import pandas as pd

from app.backtesting.engine import BacktestResult, run_buy_the_dip_backtest


def buy_the_dip(
    df: pd.DataFrame,
    lookback_days: int = 5,
    drop_pct: float = 5.0,
    max_trades_per_day: int = 1,
    amount_per_trade: float = 1000.0,
    fee_per_trade: float = 0.0,
) -> BacktestResult:
    return run_buy_the_dip_backtest(
        df=df,
        lookback_days=lookback_days,
        drop_pct=drop_pct,
        max_trades_per_day=max_trades_per_day,
        amount_per_trade=amount_per_trade,
        fee_per_trade=fee_per_trade,
    )
