from typing import Optional

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QVBoxLayout, QWidget
from pyqtgraph.Qt import QtCore, QtGui


class _CandlestickItem(pg.GraphicsObject):
    """
    Custom candlestick item compatible with pyqtgraph >=0.14.
    Expects data as list of tuples: (time, open, close, low, high).
    """

    def __init__(self, data: list[tuple[float, float, float, float, float]]):
        super().__init__()
        self._data: list[tuple[float, float, float, float, float]] = data
        self._picture: Optional[QtGui.QPicture] = None
        self._generate_picture()

    def set_data(self, data: list[tuple[float, float, float, float, float]]) -> None:
        self._data = data
        self._generate_picture()
        self.informViewBoundsChanged()
        self.update()

    def _generate_picture(self) -> None:
        self._picture = QtGui.QPicture()
        painter = QtGui.QPainter(self._picture)
        painter.setPen(pg.mkPen("w"))

        if len(self._data) >= 2:
            width = (self._data[1][0] - self._data[0][0]) / 3.0
        else:
            width = 60 * 60 * 12  # fallback width (12h in seconds)

        for (t, open_, close_, low_, high_) in self._data:
            painter.drawLine(QtCore.QPointF(t, low_), QtCore.QPointF(t, high_))
            painter.setBrush(pg.mkBrush("r" if open_ > close_ else "g"))
            painter.drawRect(QtCore.QRectF(t - width, open_, width * 2, close_ - open_))

        painter.end()

    def paint(self, painter, *args) -> None:
        if self._picture is not None:
            painter.drawPicture(0, 0, self._picture)

    def boundingRect(self) -> QtCore.QRectF:
        if self._picture is None:
            return QtCore.QRectF()
        return QtCore.QRectF(self._picture.boundingRect())


class _TimeSeriesViewBox(pg.ViewBox):
    """Restrict panning to horizontal while keeping rectangle and wheel zoom."""

    def __init__(self) -> None:
        super().__init__(enableMenu=False)
        self.setMouseMode(self.RectMode)
        self.setMouseEnabled(x=True, y=True)

    def mouseDragEvent(self, ev, axis=None) -> None:
        # Limit pan (right drag) to the x-axis only.
        if self.state["mouseMode"] == self.PanMode and ev.button() == QtCore.Qt.MouseButton.RightButton:
            return super().mouseDragEvent(ev, axis=0)
        return super().mouseDragEvent(ev, axis=axis)


class TimeSeriesChart(QWidget):
    """
    Time series chart with:
    - line or candlestick mode
    - rectangle zoom (left drag) and wheel zoom
    - crosshair with hover label (date + close)
    - selection_stats_changed emitted for the visible x-range
    """

    selection_stats_changed = pyqtSignal(str, str, float)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)

        axis = pg.graphicsItems.DateAxisItem.DateAxisItem(orientation="bottom")
        self._view_box = _TimeSeriesViewBox()

        self._plot_widget = pg.PlotWidget(
            viewBox=self._view_box,
            axisItems={"bottom": axis},
        )
        self._plot_widget.showGrid(x=True, y=True, alpha=0.25)
        self._plot_widget.setMenuEnabled(False)
        self._plot_widget.setBackground("#0e1117")

        layout.addWidget(self._plot_widget)

        self._df: Optional[pd.DataFrame] = None
        self._x: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self._mode: str = "line"
        self._candle_item: Optional[_CandlestickItem] = None
        self._event_lines: dict[str, list[pg.InfiniteLine]] = {}

        self._vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(color="#888", width=1))
        self._hline = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen(color="#888", width=1))
        self._text_item = pg.TextItem(anchor=(0, 1), color="w")

        self._proxy = pg.SignalProxy(
            self._plot_widget.scene().sigMouseMoved,
            rateLimit=60,
            slot=self._on_mouse_moved,
        )

        self._view_box.sigXRangeChanged.connect(self._on_xrange_changed)

    def set_data(self, df: pd.DataFrame, column: str = "Close") -> None:
        """Set price data. DataFrame index must be datetime-like."""
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame.")

        self._df = df.copy()

        index = pd.to_datetime(df.index)
        self._x = (index.astype("int64") // 10**9).to_numpy().flatten()
        self._y = df[column].astype(float).to_numpy().flatten()

        self._redraw()
        if len(self._x) >= 2:
            self._emit_stats_for_range(self._x[0], self._x[-1])

    def reset_view(self) -> None:
        """Reset zoom and ranges."""
        self._plot_widget.enableAutoRange(axis="xy")

    def set_mode(self, mode: str) -> None:
        """Switch chart mode: 'line' or 'candlestick'."""
        mode = mode.lower()
        if mode not in {"line", "candlestick"}:
            return
        self._mode = mode
        if self._df is not None:
            self._redraw()

    def set_event_markers(
        self,
        name_or_events,
        dates: list[pd.Timestamp] | pd.Index | None = None,
        color: str = "y",
    ) -> None:
        """
        Draw markers for events.
        - If first argument is a string, treats it as a single event name with provided dates/color.
        - If first argument is a dict, expects mapping name -> dates (or (dates, color)).
        """
        if isinstance(name_or_events, str):
            self._set_single_event_markers(name_or_events, dates, color)
            return

        # Treat as bulk assignment: clear all, then add
        self._clear_all_event_markers()
        events = name_or_events or {}
        for name, data in events.items():
            if isinstance(data, tuple) and len(data) == 2:
                dts, clr = data
                self._set_single_event_markers(name, dts, clr)
            elif isinstance(data, dict):
                self._set_single_event_markers(
                    name,
                    data.get("dates") or data.get("values"),
                    data.get("color", "w"),
                )
            else:
                self._set_single_event_markers(name, data, "w")

    def _set_single_event_markers(
        self, name: str, dates: list[pd.Timestamp] | pd.Index | None, color: str
    ) -> None:
        if name in self._event_lines:
            for line in self._event_lines[name]:
                self._plot_widget.removeItem(line)
            self._event_lines.pop(name, None)

        if dates is None or len(dates) == 0 or self._x is None:
            return

        idx = pd.to_datetime(dates)
        xs = (idx.astype("int64") // 10**9).to_numpy()
        lines: list[pg.InfiniteLine] = []
        for x_val in xs:
            line = pg.InfiniteLine(pos=x_val, angle=90, pen=pg.mkPen(color, width=1))
            line.setZValue(-10)
            self._plot_widget.addItem(line)
            lines.append(line)
        if lines:
            self._event_lines[name] = lines

    def _clear_all_event_markers(self) -> None:
        for lines in self._event_lines.values():
            for line in lines:
                self._plot_widget.removeItem(line)
        self._event_lines.clear()

    def set_dividend_markers(self, dates: list[pd.Timestamp] | pd.Index | None) -> None:
        """Backward-compatible helper for dividend-only markers."""
        self.set_event_markers("dividend", dates, "y")

    def _redraw(self) -> None:
        if self._df is None or self._x is None or self._y is None:
            return

        self._plot_widget.clear()
        self._candle_item = None

        if self._mode == "line":
            self._plot_widget.plot(self._x, self._y, pen="w")
        else:
            required = {"Open", "High", "Low", "Close"}
            if not required.issubset(self._df.columns):
                self._plot_widget.plot(self._x, self._y, pen="w")
            else:
                data: list[tuple[float, float, float, float, float]] = []
                for t, row in zip(self._x, self._df.itertuples()):
                    data.append(
                        (
                            float(t),
                            float(row.Open),
                            float(row.Close),
                            float(row.Low),
                            float(row.High),
                        )
                    )
                self._candle_item = _CandlestickItem(data)
                self._plot_widget.addItem(self._candle_item)

        self._plot_widget.addItem(self._vline, ignoreBounds=True)
        self._plot_widget.addItem(self._hline, ignoreBounds=True)
        self._plot_widget.addItem(self._text_item, ignoreBounds=True)

        self._plot_widget.enableAutoRange(axis="xy")

    def _on_mouse_moved(self, evt) -> None:
        if self._df is None or self._x is None or self._y is None:
            return

        pos = evt[0]
        if not self._plot_widget.sceneBoundingRect().contains(pos):
            return

        mouse_point = self._view_box.mapSceneToView(pos)
        x_val = mouse_point.x()

        idx = int(np.searchsorted(self._x, x_val))
        if idx >= len(self._x):
            idx = len(self._x) - 1
        if idx > 0 and abs(self._x[idx] - x_val) >= abs(self._x[idx - 1] - x_val):
            idx -= 1

        if idx < 0 or idx >= len(self._x):
            return

        x_near = self._x[idx]
        y_near = self._y[idx]
        ts = pd.to_datetime(x_near, unit="s")

        self._vline.setPos(x_near)
        self._hline.setPos(y_near)

        text = f"{ts.date().isoformat()}  |  Close: {y_near:.2f}"
        self._text_item.setText(text)
        self._text_item.setPos(x_near, y_near)

    def _emit_stats_for_range(self, x_min: float, x_max: float) -> None:
        if self._df is None or self._x is None or self._y is None:
            return

        mask = (self._x >= x_min) & (self._x <= x_max)
        if mask.sum() < 2:
            return

        df_range = self._df.iloc[mask]
        close = df_range["Close"].astype(float).squeeze()
        start_raw = close.iloc[0]
        end_raw = close.iloc[-1]
        start_price = float(start_raw.item()) if hasattr(start_raw, "item") else float(start_raw)
        end_price = float(end_raw.item()) if hasattr(end_raw, "item") else float(end_raw)
        ret = (end_price / start_price - 1.0) * 100.0

        start_date = df_range.index[0].date().isoformat()
        end_date = df_range.index[-1].date().isoformat()

        self.selection_stats_changed.emit(start_date, end_date, ret)

    def _on_xrange_changed(self, view_box, x_range) -> None:
        x_min, x_max = x_range
        self._emit_stats_for_range(x_min, x_max)
