import numbers
from typing import Optional

import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.data.loader import (
    basic_metrics,
    load_dividends,
    load_earnings_dates,
    load_price_data,
    load_splits,
)
from app.gui.components.bar_chart import BarChartWidget
from app.gui.components.charts import TimeSeriesChart


class StocksTab(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._full_df: Optional[pd.DataFrame] = None
        self._dividends: Optional[pd.DataFrame] = None
        self._splits: Optional[pd.DataFrame] = None
        self._earnings: Optional[pd.DataFrame] = None
        self._current_range: Optional[tuple[pd.Timestamp, pd.Timestamp]] = None
        self._current_df_range: Optional[pd.DataFrame] = None

        self._build_ui()
        self._connect_signals()

    # ------------------------------------------------------------------ #
    # UI
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)

        # Top control rows
        input_layout = QHBoxLayout()
        self.ticker_input = QLineEdit()
        self.ticker_input.setPlaceholderText("z.B. AAPL, MSFT, TSLA")
        self.load_button = QPushButton("Daten laden")
        input_layout.addWidget(QLabel("Ticker:"))
        input_layout.addWidget(self.ticker_input)
        input_layout.addWidget(self.load_button)

        controls_layout = QHBoxLayout()
        self.range_combo = QComboBox()
        self.range_combo.addItems(["1M", "3M", "6M", "1Y", "3Y", "5Y", "10Y", "Max"])

        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItems(["Line", "Candlestick"])

        self.show_dividends_checkbox = QCheckBox("Dividenden markieren")
        self.show_splits_checkbox = QCheckBox("Splits markieren")
        self.show_earnings_checkbox = QCheckBox("Earnings markieren")
        self.reset_zoom_button = QPushButton("Zoom zuruecksetzen")

        controls_layout.addWidget(QLabel("Zeitraum:"))
        controls_layout.addWidget(self.range_combo)
        controls_layout.addSpacing(12)
        controls_layout.addWidget(QLabel("Chart:"))
        controls_layout.addWidget(self.chart_type_combo)
        controls_layout.addSpacing(12)
        controls_layout.addWidget(self.show_dividends_checkbox)
        controls_layout.addWidget(self.show_splits_checkbox)
        controls_layout.addWidget(self.show_earnings_checkbox)
        controls_layout.addStretch()
        controls_layout.addWidget(self.reset_zoom_button)

        # Chart + bar chart area
        self.chart = TimeSeriesChart(self)
        self.selection_label = QLabel("Auswahl: -")
        self.selection_label.setContentsMargins(4, 4, 4, 4)

        bar_panel = QWidget()
        bar_layout = QVBoxLayout(bar_panel)
        bar_controls = QHBoxLayout()
        self.bar_metric_combo = QComboBox()
        self.bar_metric_combo.addItems(["Volume", "Dividend", "Split", "Earnings"])
        self.bar_label = QLabel("Balken-Metrik:")
        bar_controls.addWidget(self.bar_label)
        bar_controls.addWidget(self.bar_metric_combo)
        bar_controls.addStretch()
        bar_layout.addLayout(bar_controls)

        self.bar_chart = BarChartWidget(self)
        self.bar_table = QTableWidget()
        self.bar_table.setSortingEnabled(True)

        bar_splitter = QSplitter(Qt.Orientation.Horizontal)
        bar_splitter.addWidget(self.bar_chart)
        bar_splitter.addWidget(self.bar_table)
        bar_splitter.setSizes([600, 400])

        bar_layout.addWidget(bar_splitter)

        # Main data table
        self.table = QTableWidget()
        self.table.setSortingEnabled(True)

        main_layout.addLayout(input_layout)
        main_layout.addLayout(controls_layout)
        main_layout.addWidget(self.chart)
        main_layout.addWidget(self.selection_label)
        main_layout.addWidget(bar_panel)
        main_layout.addWidget(self.table)

    def set_language(self, lang: str) -> None:
        if lang == "de":
            self.load_button.setText("Daten laden")
            self.ticker_input.setPlaceholderText("z.B. AAPL, MSFT, TSLA")
            self.reset_zoom_button.setText("Zoom zurücksetzen")
            self.show_dividends_checkbox.setText("Dividenden markieren")
            self.show_splits_checkbox.setText("Splits markieren")
            self.show_earnings_checkbox.setText("Earnings markieren")
            self.bar_label.setText("Balken-Metrik:")
            self.selection_label.setText("Auswahl: -")
        else:
            self.load_button.setText("Load data")
            self.ticker_input.setPlaceholderText("e.g. AAPL, MSFT, TSLA")
            self.reset_zoom_button.setText("Reset zoom")
            self.show_dividends_checkbox.setText("Show dividends")
            self.show_splits_checkbox.setText("Show splits")
            self.show_earnings_checkbox.setText("Show earnings")
            self.bar_label.setText("Bar metric:")
            self.selection_label.setText("Selection: -")

    def _connect_signals(self) -> None:
        self.load_button.clicked.connect(self.on_load_clicked)
        self.range_combo.currentIndexChanged.connect(self.on_range_changed)
        self.reset_zoom_button.clicked.connect(self.on_reset_zoom_clicked)
        self.chart_type_combo.currentTextChanged.connect(self.on_chart_type_changed)
        self.show_dividends_checkbox.stateChanged.connect(self.on_event_toggle)
        self.show_splits_checkbox.stateChanged.connect(self.on_event_toggle)
        self.show_earnings_checkbox.stateChanged.connect(self.on_event_toggle)
        self.bar_metric_combo.currentTextChanged.connect(self.on_bar_metric_changed)

        self.chart.selection_stats_changed.connect(self.on_selection_stats_changed)

    # ------------------------------------------------------------------ #
    # Events
    # ------------------------------------------------------------------ #

    def on_load_clicked(self) -> None:
        ticker = self.ticker_input.text().strip().upper()
        if not ticker:
            return

        try:
            df = load_price_data(ticker)
            self._full_df = df
            self._dividends = load_dividends(ticker)
            self._splits = load_splits(ticker)
            self._earnings = load_earnings_dates(ticker)
            metrics = basic_metrics(df)
        except Exception as exc:  # pragma: no cover - UI feedback only
            self._full_df = None
            self._dividends = None
            self._splits = None
            self._earnings = None
            self._current_range = None
            self._current_df_range = None
            self.selection_label.setText(f"Ladefehler: {exc}")
            self.chart.set_event_markers({})
            self.table.clear()
            self.bar_chart.clear()
            self.bar_table.clear()
            return

        self.selection_label.setText("Auswahl: -")
        self._update_chart_for_current_range()

    def on_range_changed(self) -> None:
        if self._full_df is None:
            return
        self._update_chart_for_current_range()

    def on_reset_zoom_clicked(self) -> None:
        self.chart.reset_view()

    def on_chart_type_changed(self, mode: str) -> None:
        mode = mode.lower()
        self.chart.set_mode(mode)
        if self._full_df is not None:
            self._update_chart_for_current_range()

    def on_selection_stats_changed(self, start_str: str, end_str: str, ret: float) -> None:
        self.selection_label.setText(f"Auswahl: {start_str} -> {end_str}  |  Return: {ret:.2f}%")
        if self._full_df is None:
            return
        start = pd.to_datetime(start_str)
        end = pd.to_datetime(end_str)
        self._current_range = (start, end)
        self._update_bar_chart(start, end)

    def on_event_toggle(self, state: int) -> None:
        self._apply_event_markers()

    def on_bar_metric_changed(self, metric: str) -> None:
        if self._current_range is None:
            return
        self._update_bar_chart(*self._current_range)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _set_metric_labels(self, metrics: dict) -> None:
        # Metrics no longer displayed; keep method for compatibility.
        return

    def _update_chart_for_current_range(self) -> None:
        if self._full_df is None or self._full_df.empty:
            return

        df = self._full_df.copy()
        period = self.range_combo.currentText()

        end = pd.to_datetime(df.index.max())
        if period == "1M":
            start = end - pd.DateOffset(months=1)
        elif period == "3M":
            start = end - pd.DateOffset(months=3)
        elif period == "6M":
            start = end - pd.DateOffset(months=6)
        elif period == "1Y":
            start = end - pd.DateOffset(years=1)
        elif period == "3Y":
            start = end - pd.DateOffset(years=3)
        elif period == "5Y":
            start = end - pd.DateOffset(years=5)
        elif period == "10Y":
            start = end - pd.DateOffset(years=10)
        else:
            start = pd.to_datetime(df.index.min())

        df_range = df[df.index >= start]

        if df_range.empty:
            self.selection_label.setText("Keine Daten im gewaehlten Zeitraum.")
            self.chart.set_event_markers({})
            self.table.clear()
            self.bar_chart.clear()
            self.bar_table.clear()
            self._current_range = None
            self._current_df_range = None
            return

        self._current_range = (pd.to_datetime(start), pd.to_datetime(end))
        self._current_df_range = df_range

        self.chart.set_mode(self.chart_type_combo.currentText().lower())
        self.chart.set_data(df_range, column="Close")

        self._apply_event_markers()
        self._update_selection_label_from_df(df_range)
        self._update_table(df_range)
        self._update_bar_chart(*self._current_range)

    def _apply_event_markers(self) -> None:
        if self._current_range is None:
            self.chart.set_event_markers({})
            return

        start, end = self._current_range

        if self.show_dividends_checkbox.isChecked() and self._dividends is not None and not self._dividends.empty:
            idx = pd.to_datetime(self._dividends.index)
            mask = (idx >= start) & (idx <= end)
            self.chart.set_event_markers("dividend", idx[mask], "y")
        else:
            self.chart.set_event_markers("dividend", None, "y")

        if self.show_splits_checkbox.isChecked() and self._splits is not None and not self._splits.empty:
            idx = pd.to_datetime(self._splits.index)
            mask = (idx >= start) & (idx <= end)
            self.chart.set_event_markers("split", idx[mask], "#22d3ee")
        else:
            self.chart.set_event_markers("split", None, "#22d3ee")

        if self.show_earnings_checkbox.isChecked() and self._earnings is not None and not self._earnings.empty:
            idx = pd.to_datetime(self._earnings.index)
            mask = (idx >= start) & (idx <= end)
            self.chart.set_event_markers("earnings", idx[mask], "orange")
        else:
            self.chart.set_event_markers("earnings", None, "orange")

    def _update_selection_label_from_df(self, df_range: pd.DataFrame) -> None:
        close = df_range["Close"].astype(float).squeeze()
        if len(close) < 2:
            self.selection_label.setText("Auswahl: -")
            return
        start_raw = close.iloc[0]
        end_raw = close.iloc[-1]
        start_price = float(start_raw.item()) if hasattr(start_raw, "item") else float(start_raw)
        end_price = float(end_raw.item()) if hasattr(end_raw, "item") else float(end_raw)
        ret = (end_price / start_price - 1.0) * 100.0
        start_date = df_range.index[0].date().isoformat()
        end_date = df_range.index[-1].date().isoformat()
        self.selection_label.setText(f"Auswahl: {start_date} -> {end_date}  |  Return: {ret:.2f}%")

    def _update_table(self, df: pd.DataFrame) -> None:
        df_display = df.copy()
        df_display.index = pd.to_datetime(df_display.index)
        df_display.index = df_display.index.map(lambda x: x.date().isoformat())

        col_labels: list[str] = []
        for col in df_display.columns:
            if isinstance(col, tuple):
                col_labels.append(" / ".join(str(part) for part in col))
            else:
                col_labels.append(str(col))

        self.table.clear()

        rows, cols = df_display.shape
        self.table.setRowCount(rows)
        self.table.setColumnCount(cols + 1)

        headers = ["Date"] + col_labels
        self.table.setHorizontalHeaderLabels(headers)

        for row_idx, (index, row) in enumerate(df_display.iterrows()):
            self.table.setItem(row_idx, 0, QTableWidgetItem(str(index)))
            for col_idx, col_name in enumerate(df_display.columns, start=1):
                val = row[col_name]
                item_text = self._format_value(val)
                self.table.setItem(row_idx, col_idx, QTableWidgetItem(item_text))

        self.table.resizeColumnsToContents()

    def _format_value(self, value) -> str:
        if pd.isna(value):
            return "-"
        if isinstance(value, numbers.Integral):
            return f"{int(value):,}"
        if isinstance(value, numbers.Real):
            return f"{float(value):,.4f}"
        return str(value)

    # ------------------------------------------------------------------ #
    # Bar chart helpers
    # ------------------------------------------------------------------ #

    def _update_bar_chart(self, start: pd.Timestamp, end: pd.Timestamp) -> None:
        metric = self.bar_metric_combo.currentText()
        self.bar_chart.clear()
        self.bar_table.clear()

        if self._full_df is None:
            return

        if metric == "Volume":
            df_range = self._full_df.loc[(self._full_df.index >= start) & (self._full_df.index <= end)]
            if "Volume" not in df_range.columns or df_range.empty:
                return
            self.bar_chart.set_series(df_range.index, df_range["Volume"], "Volume")
            self._fill_bar_table(df_range.index, df_range["Volume"], ["Datum", "Volume"])
            return

        if metric == "Dividend" and self._dividends is not None and not self._dividends.empty:
            idx = pd.to_datetime(self._dividends.index)
            mask = (idx >= start) & (idx <= end)
            series = self._dividends.loc[mask, "Dividend"]
            if series.empty:
                return
            self.bar_chart.set_series(series.index, series, "Dividend")
            self._fill_bar_table(series.index, series, ["Datum", "Dividend"])
            return

        if metric == "Split" and self._splits is not None and not self._splits.empty:
            idx = pd.to_datetime(self._splits.index)
            mask = (idx >= start) & (idx <= end)
            series = self._splits.loc[mask, "Split"]
            if series.empty:
                return
            self.bar_chart.set_series(series.index, series, "Split")
            self._fill_bar_table(series.index, series, ["Datum", "Split"])
            return

        if metric == "Earnings" and self._earnings is not None and not self._earnings.empty:
            idx = pd.to_datetime(self._earnings.index)
            mask = (idx >= start) & (idx <= end)
            series = self._earnings.loc[mask, "Earnings"]
            if series.empty:
                return
            self.bar_chart.set_series(series.index, series, "Earnings")
            self._fill_bar_table(series.index, series, ["Datum", "Earnings"])
            return

    def _fill_bar_table(self, dates: pd.Index, values: pd.Series, headers: list[str]) -> None:
        self.bar_table.clear()
        self.bar_table.setRowCount(len(values))
        self.bar_table.setColumnCount(2)
        self.bar_table.setHorizontalHeaderLabels(headers)
        for i, (dt_idx, val) in enumerate(zip(dates, values)):
            self.bar_table.setItem(i, 0, QTableWidgetItem(pd.to_datetime(dt_idx).date().isoformat()))
            self.bar_table.setItem(i, 1, QTableWidgetItem(self._format_value(val)))
        self.bar_table.resizeColumnsToContents()
