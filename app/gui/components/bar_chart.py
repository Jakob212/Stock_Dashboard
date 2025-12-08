from typing import Optional

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtWidgets import QVBoxLayout, QWidget


class BarChartWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        axis = pg.graphicsItems.DateAxisItem.DateAxisItem(orientation="bottom")
        self._plot = pg.PlotWidget(axisItems={"bottom": axis})
        self._plot.setBackground("#0e1117")
        self._plot.showGrid(x=True, y=True, alpha=0.2)
        self._bar_item: Optional[pg.BarGraphItem] = None

        layout.addWidget(self._plot)

    def clear(self) -> None:
        self._plot.clear()
        self._bar_item = None

    def set_series(self, dates: pd.Index, values: pd.Series, label: str = "") -> None:
        """Render bars for the given date/value series."""
        self.clear()
        if dates is None or values is None or len(dates) == 0 or len(values) == 0:
            return

        idx = pd.to_datetime(dates)
        xs = (idx.astype("int64") // 10**9).to_numpy()
        vals = pd.to_numeric(values, errors="coerce").fillna(0).to_numpy()

        if len(xs) >= 2:
            width = (xs[1] - xs[0]) * 0.6
        else:
            width = 24 * 3600 * 0.6  # ~0.6 day

        self._bar_item = pg.BarGraphItem(x=xs, height=vals, width=width, brush="#3b82f6")
        self._plot.addItem(self._bar_item)
        if label:
            self._plot.setLabel("left", label)

        self._plot.enableAutoRange(axis="xy")
