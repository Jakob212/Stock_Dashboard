from typing import Optional

import pandas as pd
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.backtesting.engine import (
    BacktestResult,
    run_buy_the_dip_backtest,
    run_bollinger_reversion_backtest,
    run_dca_backtest,
    run_ema_crossover_backtest,
    run_macd_backtest,
    run_momentum_12_1_backtest,
    run_rsi_reversion_backtest,
    run_stoch_osc_backtest,
    run_sma_crossover_backtest,
    run_donchian_breakout_backtest,
)
from app.data.loader import load_price_data


def _label_for_metric(key: str) -> str:
    mapping = {
        "n_trades": "N Trades",
        "total_fees": "Total Fees",
        "total_invested": "Total Invested",
        "final_value": "Final Value",
        "strategy_profit": "Strategy Profit",
        "bh_profit": "Buy&Hold Profit",
        "strategy_return_pct": "Strategy Return (%)",
        "strategy_annual_return_pct": "Strategy Annual Return (%)",
        "bh_return_pct": "Buy&Hold Return (%)",
        "bh_annual_return_pct": "Buy&Hold Annual Return (%)",
        "outperformance_total_pct": "Outperformance Total (%)",
        "outperformance_annual_pct": "Outperformance Annual (%)",
    }
    return mapping.get(key, key.replace("_", " ").title())


class BuyTheDipWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._result: Optional[BacktestResult] = None
        self._build_ui()
        self._lang = "de"

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)

        # Inputs
        input_layout = QFormLayout()
        self.ticker_input = QLineEdit()
        self.range_combo = QComboBox()
        self.range_combo.addItems(["1Y", "3Y", "5Y", "10Y", "Max"])

        self.lookback_spin = QSpinBox()
        self.lookback_spin.setRange(1, 120)
        self.lookback_spin.setValue(5)

        self.drop_spin = QDoubleSpinBox()
        self.drop_spin.setRange(0.1, 100.0)
        self.drop_spin.setDecimals(2)
        self.drop_spin.setValue(5.0)

        self.max_trades_spin = QSpinBox()
        self.max_trades_spin.setRange(1, 10)
        self.max_trades_spin.setValue(1)

        self.amount_spin = QDoubleSpinBox()
        self.amount_spin.setRange(1, 1_000_000)
        self.amount_spin.setValue(1000.0)
        self.amount_spin.setDecimals(2)

        self.fee_spin = QDoubleSpinBox()
        self.fee_spin.setRange(0.0, 1000.0)
        self.fee_spin.setDecimals(2)
        self.fee_spin.setValue(0.0)

        input_layout.addRow("Ticker:", self.ticker_input)
        input_layout.addRow("Zeitraum:", self.range_combo)
        input_layout.addRow("Lookback Tage:", self.lookback_spin)
        input_layout.addRow("Drop %:", self.drop_spin)
        input_layout.addRow("Max Trades/Tag:", self.max_trades_spin)
        input_layout.addRow("Betrag/Trade:", self.amount_spin)
        input_layout.addRow("Fee/Trade:", self.fee_spin)

        self.run_button = QPushButton("Backtest ausführen")
        input_row = QHBoxLayout()
        input_row.addLayout(input_layout)
        input_row.addWidget(self.run_button)
        main_layout.addLayout(input_row)

        # Metrics
        self.metrics_form = QFormLayout()
        self.metric_labels = {
            "n_trades": QLabel("-"),
            "total_fees": QLabel("-"),
            "total_invested": QLabel("-"),
            "final_value": QLabel("-"),
            "strategy_profit": QLabel("-"),
            "bh_profit": QLabel("-"),
            "strategy_return_pct": QLabel("-"),
            "strategy_annual_return_pct": QLabel("-"),
            "bh_return_pct": QLabel("-"),
            "bh_annual_return_pct": QLabel("-"),
            "outperformance_total_pct": QLabel("-"),
            "outperformance_annual_pct": QLabel("-"),
        }
        self.metrics_form.addRow("Anzahl Trades:", self.metric_labels["n_trades"])
        self.metrics_form.addRow("Total Fees:", self.metric_labels["total_fees"])
        self.metrics_form.addRow("Total Invested:", self.metric_labels["total_invested"])
        self.metrics_form.addRow("Final Value:", self.metric_labels["final_value"])
        self.metrics_form.addRow("Strategy Profit:", self.metric_labels["strategy_profit"])
        self.metrics_form.addRow("Buy&Hold Profit:", self.metric_labels["bh_profit"])
        self.metrics_form.addRow("Strategy Return (%):", self.metric_labels["strategy_return_pct"])
        self.metrics_form.addRow("Strategy Annual (%):", self.metric_labels["strategy_annual_return_pct"])
        self.metrics_form.addRow("Buy&Hold Return (%):", self.metric_labels["bh_return_pct"])
        self.metrics_form.addRow("Buy&Hold Annual (%):", self.metric_labels["bh_annual_return_pct"])
        self.metrics_form.addRow("Outperformance Total (%):", self.metric_labels["outperformance_total_pct"])
        self.metrics_form.addRow("Outperformance Annual (%):", self.metric_labels["outperformance_annual_pct"])

        # Plot
        self.plot = pg.PlotWidget()
        self.plot.addLegend()
        self.plot.showGrid(x=True, y=True, alpha=0.2)

        # Trades table
        self.trades_table = QTableWidget()
        self.trades_table.setSortingEnabled(True)

        # Layout with splitters
        plot_splitter = QSplitter(Qt.Orientation.Vertical)
        metrics_widget = QWidget()
        metrics_widget.setLayout(self.metrics_form)
        plot_splitter.addWidget(metrics_widget)
        plot_splitter.addWidget(self.plot)
        plot_splitter.setSizes([200, 400])

        main_splitter = QSplitter(Qt.Orientation.Vertical)
        main_splitter.addWidget(plot_splitter)
        main_splitter.addWidget(self.trades_table)
        main_splitter.setSizes([500, 300])

        main_layout.addWidget(main_splitter)

        self.run_button.clicked.connect(self.on_run_clicked)

    def set_language(self, lang: str) -> None:
        self._lang = lang
        if lang == "de":
            self.ticker_input.setPlaceholderText("z.B. AAPL, MSFT, TSLA")
            self.range_combo.setItemText(0, "1Y")
            self.range_combo.setItemText(1, "3Y")
            self.range_combo.setItemText(2, "5Y")
            self.range_combo.setItemText(3, "10Y")
            self.range_combo.setItemText(4, "Max")
            self.run_button.setText("Backtest ausführen")
        else:
            self.ticker_input.setPlaceholderText("e.g. AAPL, MSFT, TSLA")
            self.run_button.setText("Run backtest")

    def on_run_clicked(self) -> None:
        ticker = self.ticker_input.text().strip().upper()
        if not ticker:
            QMessageBox.warning(self, "Fehler" if self._lang == "de" else "Error", "Bitte einen Ticker eingeben." if self._lang == "de" else "Please enter a ticker.")
            return
        try:
            start = None
            end = None
            period = self.range_combo.currentText()
            if period != "Max":
                end = pd.Timestamp.today().date()
                if period == "1Y":
                    start = end - pd.DateOffset(years=1)
                elif period == "3Y":
                    start = end - pd.DateOffset(years=3)
                elif period == "5Y":
                    start = end - pd.DateOffset(years=5)
                elif period == "10Y":
                    start = end - pd.DateOffset(years=10)
                if start is not None:
                    start = start.date()
                    end = end
            df = load_price_data(ticker, start=start, end=end)
            lookback = self.lookback_spin.value()
            if len(df) <= lookback:
                QMessageBox.warning(
                    self,
                    "Fehler" if self._lang == "de" else "Error",
                    "Zu wenig Daten für den gewählten Lookback." if self._lang == "de" else "Not enough data for the chosen lookback.",
                )
                return
            result = run_buy_the_dip_backtest(
                df=df,
                lookback_days=lookback,
                drop_pct=self.drop_spin.value(),
                max_trades_per_day=self.max_trades_spin.value(),
                amount_per_trade=self.amount_spin.value(),
                fee_per_trade=self.fee_spin.value(),
            )
            self._result = result
            self._render_result(result)
        except Exception as exc:
            title = "Fehler" if self._lang == "de" else "Error"
            msg = f"Backtest fehlgeschlagen: {exc}" if self._lang == "de" else f"Backtest failed: {exc}"
            QMessageBox.critical(self, title, msg)

    def _render_result(self, result: BacktestResult) -> None:
        # metrics
        for key, lbl in self.metric_labels.items():
            val = result.metrics.get(key, "-")
            if isinstance(val, float):
                lbl.setText(f"{val:,.2f}")
            else:
                lbl.setText(str(val))
        try:
            total_inv = float(result.metrics.get("total_invested", 0.0))
            strat_final = float(result.metrics.get("final_value", 0.0))
            bh_final = float(result.equity_curve["buy_and_hold"].iloc[-1]) if not result.equity_curve.empty else 0.0
            self.metric_labels["strategy_profit"].setText(f"{strat_final - total_inv:,.2f}")
            self.metric_labels["bh_profit"].setText(f"{bh_final - total_inv:,.2f}")
        except Exception:
            self.metric_labels["strategy_profit"].setText("-")
            self.metric_labels["bh_profit"].setText("-")
        try:
            total_inv = float(result.metrics.get("total_invested", 0.0))
            strat_final = float(result.metrics.get("final_value", 0.0))
            bh_final = float(result.equity_curve["buy_and_hold"].iloc[-1]) if not result.equity_curve.empty else 0.0
            self.metric_labels["strategy_profit"].setText(f"{strat_final - total_inv:,.2f}")
            self.metric_labels["bh_profit"].setText(f"{bh_final - total_inv:,.2f}")
        except Exception:
            self.metric_labels["strategy_profit"].setText("-")
            self.metric_labels["bh_profit"].setText("-")
        # derived profits
        try:
            total_inv = float(result.metrics.get("total_invested", 0.0))
            strat_final = float(result.metrics.get("final_value", 0.0))
            bh_final = float(result.equity_curve["buy_and_hold"].iloc[-1]) if not result.equity_curve.empty else 0.0
            self.metric_labels["strategy_profit"].setText(f"{strat_final - total_inv:,.2f}")
            self.metric_labels["bh_profit"].setText(f"{bh_final - total_inv:,.2f}")
        except Exception:
            self.metric_labels["strategy_profit"].setText("-")
            self.metric_labels["bh_profit"].setText("-")

        # plot
        self.plot.clear()
        self.plot.addLegend()
        eq = result.equity_curve
        if not eq.empty:
            x = (pd.to_datetime(eq.index).astype("int64") // 10**9).to_numpy()
            if "strategy" in eq:
                self.plot.plot(x, eq["strategy"], pen=pg.mkPen("#3b82f6", width=2), name="Strategie")
            if "buy_and_hold" in eq:
                self.plot.plot(x, eq["buy_and_hold"], pen=pg.mkPen("#10b981", width=2), name="Buy & Hold")

        # trades table
        trades = result.trades
        headers = ["Datum", "Preis", "Shares", "Amount", "Fee"]
        self.trades_table.clear()
        self.trades_table.setRowCount(len(trades))
        self.trades_table.setColumnCount(len(headers))
        self.trades_table.setHorizontalHeaderLabels(headers)
        for i, trade in enumerate(trades):
            self.trades_table.setItem(i, 0, QTableWidgetItem(str(trade.get("date"))))
            self.trades_table.setItem(i, 1, QTableWidgetItem(f"{trade.get('price', 0):,.2f}"))
            self.trades_table.setItem(i, 2, QTableWidgetItem(str(trade.get("shares", 0))))
            self.trades_table.setItem(i, 3, QTableWidgetItem(f"{trade.get('amount', 0):,.2f}"))
            self.trades_table.setItem(i, 4, QTableWidgetItem(f"{trade.get('fee', 0):,.2f}"))
        self.trades_table.resizeColumnsToContents()


class SmaCrossoverWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._lang = "de"
        self._result: Optional[BacktestResult] = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.ticker_input = QLineEdit()
        self.range_combo = QComboBox()
        self.range_combo.addItems(["1Y", "3Y", "5Y", "10Y", "Max"])
        self.short_spin = QSpinBox()
        self.short_spin.setRange(1, 250)
        self.short_spin.setValue(20)
        self.long_spin = QSpinBox()
        self.long_spin.setRange(2, 400)
        self.long_spin.setValue(50)
        self.amount_spin = QDoubleSpinBox()
        self.amount_spin.setRange(100.0, 1_000_000.0)
        self.amount_spin.setValue(10_000.0)
        self.fee_spin = QDoubleSpinBox()
        self.fee_spin.setRange(0.0, 1000.0)
        self.fee_spin.setValue(0.0)
        form.addRow("Ticker:", self.ticker_input)
        form.addRow("Zeitraum:", self.range_combo)
        form.addRow("Short SMA:", self.short_spin)
        form.addRow("Long SMA:", self.long_spin)
        form.addRow("Startkapital:", self.amount_spin)
        form.addRow("Fee/Trade:", self.fee_spin)
        self.run_button = QPushButton("Backtest ausführen")

        row = QHBoxLayout()
        row.addLayout(form)
        row.addWidget(self.run_button)
        layout.addLayout(row)

        self.metrics_form = QFormLayout()
        self.metric_labels = {
            "n_trades": QLabel("-"),
            "total_fees": QLabel("-"),
            "total_invested": QLabel("-"),
            "final_value": QLabel("-"),
            "strategy_profit": QLabel("-"),
            "bh_profit": QLabel("-"),
            "strategy_return_pct": QLabel("-"),
            "strategy_annual_return_pct": QLabel("-"),
            "bh_return_pct": QLabel("-"),
            "bh_annual_return_pct": QLabel("-"),
            "outperformance_total_pct": QLabel("-"),
            "outperformance_annual_pct": QLabel("-"),
        }
        for key, lbl in self.metric_labels.items():
            self.metrics_form.addRow(_label_for_metric(key), lbl)

        self.plot = pg.PlotWidget()
        self.plot.addLegend()
        self.plot.showGrid(x=True, y=True, alpha=0.2)

        self.trades_table = QTableWidget()
        self.trades_table.setSortingEnabled(True)

        split = QSplitter(Qt.Orientation.Vertical)
        metrics_widget = QWidget()
        metrics_widget.setLayout(self.metrics_form)
        split.addWidget(metrics_widget)
        split.addWidget(self.plot)
        split.setSizes([200, 400])

        main_splitter = QSplitter(Qt.Orientation.Vertical)
        main_splitter.addWidget(split)
        main_splitter.addWidget(self.trades_table)
        main_splitter.setSizes([500, 300])

        layout.addWidget(main_splitter)

        self.run_button.clicked.connect(self.on_run_clicked)

    def set_language(self, lang: str) -> None:
        self._lang = lang
        if lang == "de":
            self.run_button.setText("Backtest ausführen")
        else:
            self.run_button.setText("Run backtest")

    def on_run_clicked(self) -> None:
        ticker = self.ticker_input.text().strip().upper()
        if not ticker:
            QMessageBox.warning(self, "Fehler" if self._lang == "de" else "Error", "Bitte einen Ticker eingeben." if self._lang == "de" else "Please enter a ticker.")
            return
        try:
            start = None
            end = None
            period = self.range_combo.currentText()
            if period != "Max":
                end = pd.Timestamp.today().date()
                if period == "1Y":
                    start = end - pd.DateOffset(years=1)
                elif period == "3Y":
                    start = end - pd.DateOffset(years=3)
                elif period == "5Y":
                    start = end - pd.DateOffset(years=5)
                elif period == "10Y":
                    start = end - pd.DateOffset(years=10)
                start = start.date()
            df = load_price_data(ticker, start=start, end=end)
            result = run_sma_crossover_backtest(
                df=df,
                short_window=self.short_spin.value(),
                long_window=self.long_spin.value(),
                initial_cash=self.amount_spin.value(),
                fee_per_trade=self.fee_spin.value(),
            )
            self._result = result
            self._render_result(result)
        except Exception as exc:
            title = "Fehler" if self._lang == "de" else "Error"
            msg = f"Backtest fehlgeschlagen: {exc}" if self._lang == "de" else f"Backtest failed: {exc}"
            QMessageBox.critical(self, title, msg)

    def _render_result(self, result: BacktestResult) -> None:
        for key, lbl in self.metric_labels.items():
            val = result.metrics.get(key, "-")
            if isinstance(val, float):
                lbl.setText(f"{val:,.2f}")
            else:
                lbl.setText(str(val))

        self.plot.clear()
        self.plot.addLegend()
        eq = result.equity_curve
        if not eq.empty:
            x = (pd.to_datetime(eq.index).astype("int64") // 10**9).to_numpy()
            if "strategy" in eq:
                self.plot.plot(x, eq["strategy"], pen=pg.mkPen("#f97316", width=2), name="Strategie")
            if "buy_and_hold" in eq:
                self.plot.plot(x, eq["buy_and_hold"], pen=pg.mkPen("#10b981", width=2), name="Buy & Hold")

        trades = result.trades
        headers = ["Datum", "Side", "Preis", "Shares", "Amount", "Fee"]
        self.trades_table.clear()
        self.trades_table.setRowCount(len(trades))
        self.trades_table.setColumnCount(len(headers))
        self.trades_table.setHorizontalHeaderLabels(headers)
        for i, trade in enumerate(trades):
            self.trades_table.setItem(i, 0, QTableWidgetItem(str(trade.get("date"))))
            self.trades_table.setItem(i, 1, QTableWidgetItem(trade.get("side", "")))
            self.trades_table.setItem(i, 2, QTableWidgetItem(f"{trade.get('price', 0):,.2f}"))
            self.trades_table.setItem(i, 3, QTableWidgetItem(str(trade.get("shares", 0))))
            self.trades_table.setItem(i, 4, QTableWidgetItem(f"{trade.get('amount', 0):,.2f}"))
            self.trades_table.setItem(i, 5, QTableWidgetItem(f"{trade.get('fee', 0):,.2f}"))
        self.trades_table.resizeColumnsToContents()


class DcaWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._lang = "de"
        self._result: Optional[BacktestResult] = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.ticker_input = QLineEdit()
        self.range_combo = QComboBox()
        self.range_combo.addItems(["1Y", "3Y", "5Y", "10Y", "Max"])
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(5, 120)
        self.interval_spin.setValue(30)
        self.amount_spin = QDoubleSpinBox()
        self.amount_spin.setRange(10.0, 1_000_000.0)
        self.amount_spin.setValue(500.0)
        self.fee_spin = QDoubleSpinBox()
        self.fee_spin.setRange(0.0, 1000.0)
        self.fee_spin.setValue(0.0)

        form.addRow("Ticker:", self.ticker_input)
        form.addRow("Zeitraum:", self.range_combo)
        form.addRow("Intervall (Tage):", self.interval_spin)
        form.addRow("Betrag/Intervall:", self.amount_spin)
        form.addRow("Fee/Trade:", self.fee_spin)
        self.run_button = QPushButton("Backtest ausführen")

        row = QHBoxLayout()
        row.addLayout(form)
        row.addWidget(self.run_button)
        layout.addLayout(row)

        self.metrics_form = QFormLayout()
        self.metric_labels = {
            "n_trades": QLabel("-"),
            "total_fees": QLabel("-"),
            "total_invested": QLabel("-"),
            "final_value": QLabel("-"),
            "strategy_profit": QLabel("-"),
            "bh_profit": QLabel("-"),
            "strategy_return_pct": QLabel("-"),
            "strategy_annual_return_pct": QLabel("-"),
            "bh_return_pct": QLabel("-"),
            "bh_annual_return_pct": QLabel("-"),
            "outperformance_total_pct": QLabel("-"),
            "outperformance_annual_pct": QLabel("-"),
        }
        for key, lbl in self.metric_labels.items():
            self.metrics_form.addRow(key.replace("_", " ").title(), lbl)

        self.plot = pg.PlotWidget()
        self.plot.addLegend()
        self.plot.showGrid(x=True, y=True, alpha=0.2)

        self.trades_table = QTableWidget()
        self.trades_table.setSortingEnabled(True)

        split = QSplitter(Qt.Orientation.Vertical)
        metrics_widget = QWidget()
        metrics_widget.setLayout(self.metrics_form)
        split.addWidget(metrics_widget)
        split.addWidget(self.plot)
        split.setSizes([200, 400])

        main_splitter = QSplitter(Qt.Orientation.Vertical)
        main_splitter.addWidget(split)
        main_splitter.addWidget(self.trades_table)
        main_splitter.setSizes([500, 300])

        layout.addWidget(main_splitter)

        self.run_button.clicked.connect(self.on_run_clicked)

    def set_language(self, lang: str) -> None:
        self._lang = lang
        if lang == "de":
            self.run_button.setText("Backtest ausführen")
        else:
            self.run_button.setText("Run backtest")

    def on_run_clicked(self) -> None:
        ticker = self.ticker_input.text().strip().upper()
        if not ticker:
            QMessageBox.warning(self, "Fehler" if self._lang == "de" else "Error", "Bitte einen Ticker eingeben." if self._lang == "de" else "Please enter a ticker.")
            return
        try:
            start = None
            end = None
            period = self.range_combo.currentText()
            if period != "Max":
                end = pd.Timestamp.today().date()
                if period == "1Y":
                    start = end - pd.DateOffset(years=1)
                elif period == "3Y":
                    start = end - pd.DateOffset(years=3)
                elif period == "5Y":
                    start = end - pd.DateOffset(years=5)
                elif period == "10Y":
                    start = end - pd.DateOffset(years=10)
                start = start.date()
            df = load_price_data(ticker, start=start, end=end)
            result = run_dca_backtest(
                df=df,
                interval_days=self.interval_spin.value(),
                amount_per_trade=self.amount_spin.value(),
                fee_per_trade=self.fee_spin.value(),
            )
            self._result = result
            self._render_result(result)
        except Exception as exc:
            title = "Fehler" if self._lang == "de" else "Error"
            msg = f"Backtest fehlgeschlagen: {exc}" if self._lang == "de" else f"Backtest failed: {exc}"
            QMessageBox.critical(self, title, msg)

    def _render_result(self, result: BacktestResult) -> None:
        for key, lbl in self.metric_labels.items():
            val = result.metrics.get(key, "-")
            if isinstance(val, float):
                lbl.setText(f"{val:,.2f}")
            else:
                lbl.setText(str(val))

        self.plot.clear()
        self.plot.addLegend()
        eq = result.equity_curve
        if not eq.empty:
            x = (pd.to_datetime(eq.index).astype("int64") // 10**9).to_numpy()
            if "strategy" in eq:
                self.plot.plot(x, eq["strategy"], pen=pg.mkPen("#6366f1", width=2), name="Strategie")
            if "buy_and_hold" in eq:
                self.plot.plot(x, eq["buy_and_hold"], pen=pg.mkPen("#10b981", width=2), name="Buy & Hold")

        trades = result.trades
        headers = ["Datum", "Preis", "Shares", "Amount", "Fee"]
        self.trades_table.clear()
        self.trades_table.setRowCount(len(trades))
        self.trades_table.setColumnCount(len(headers))
        self.trades_table.setHorizontalHeaderLabels(headers)
        for i, trade in enumerate(trades):
            self.trades_table.setItem(i, 0, QTableWidgetItem(str(trade.get("date"))))
            self.trades_table.setItem(i, 1, QTableWidgetItem(f"{trade.get('price', 0):,.2f}"))
            self.trades_table.setItem(i, 2, QTableWidgetItem(str(trade.get("shares", 0))))
            self.trades_table.setItem(i, 3, QTableWidgetItem(f"{trade.get('amount', 0):,.2f}"))
            self.trades_table.setItem(i, 4, QTableWidgetItem(f"{trade.get('fee', 0):,.2f}"))
        self.trades_table.resizeColumnsToContents()


class EmaCrossoverWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._lang = "de"
        self._result: Optional[BacktestResult] = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.ticker_input = QLineEdit()
        self.range_combo = QComboBox()
        self.range_combo.addItems(["1Y", "3Y", "5Y", "10Y", "Max"])
        self.short_spin = QSpinBox()
        self.short_spin.setRange(1, 200)
        self.short_spin.setValue(12)
        self.long_spin = QSpinBox()
        self.long_spin.setRange(2, 400)
        self.long_spin.setValue(26)
        self.amount_spin = QDoubleSpinBox()
        self.amount_spin.setRange(100.0, 1_000_000.0)
        self.amount_spin.setValue(10_000.0)
        self.fee_spin = QDoubleSpinBox()
        self.fee_spin.setRange(0.0, 1000.0)
        self.fee_spin.setValue(0.0)
        form.addRow("Ticker:", self.ticker_input)
        form.addRow("Zeitraum:", self.range_combo)
        form.addRow("Short EMA:", self.short_spin)
        form.addRow("Long EMA:", self.long_spin)
        form.addRow("Startkapital:", self.amount_spin)
        form.addRow("Fee/Trade:", self.fee_spin)
        self.run_button = QPushButton("Backtest ausführen")

        row = QHBoxLayout()
        row.addLayout(form)
        row.addWidget(self.run_button)
        layout.addLayout(row)

        self.metrics_form = QFormLayout()
        self.metric_labels = {k: QLabel("-") for k in [
            "n_trades","total_fees","total_invested","final_value",
            "strategy_return_pct","strategy_annual_return_pct",
            "bh_return_pct","bh_annual_return_pct",
            "outperformance_total_pct","outperformance_annual_pct"
        ]}
        for key, lbl in self.metric_labels.items():
            self.metrics_form.addRow(key.replace("_", " ").title(), lbl)

        self.plot = pg.PlotWidget()
        self.plot.addLegend()
        self.plot.showGrid(x=True, y=True, alpha=0.2)

        self.trades_table = QTableWidget()
        self.trades_table.setSortingEnabled(True)

        split = QSplitter(Qt.Orientation.Vertical)
        metrics_widget = QWidget()
        metrics_widget.setLayout(self.metrics_form)
        split.addWidget(metrics_widget)
        split.addWidget(self.plot)
        split.setSizes([200, 400])

        main_splitter = QSplitter(Qt.Orientation.Vertical)
        main_splitter.addWidget(split)
        main_splitter.addWidget(self.trades_table)
        main_splitter.setSizes([500, 300])

        layout.addWidget(main_splitter)
        self.run_button.clicked.connect(self.on_run_clicked)

    def set_language(self, lang: str) -> None:
        self._lang = lang
        self.run_button.setText("Backtest ausführen" if lang == "de" else "Run backtest")

    def on_run_clicked(self) -> None:
        ticker = self.ticker_input.text().strip().upper()
        if not ticker:
            QMessageBox.warning(self, "Fehler" if self._lang == "de" else "Error",
                                "Bitte einen Ticker eingeben." if self._lang == "de" else "Please enter a ticker.")
            return
        try:
            start = None
            end = None
            period = self.range_combo.currentText()
            if period != "Max":
                end = pd.Timestamp.today().date()
                if period == "1Y":
                    start = end - pd.DateOffset(years=1)
                elif period == "3Y":
                    start = end - pd.DateOffset(years=3)
                elif period == "5Y":
                    start = end - pd.DateOffset(years=5)
                elif period == "10Y":
                    start = end - pd.DateOffset(years=10)
                start = start.date()
            df = load_price_data(ticker, start=start, end=end)
            result = run_ema_crossover_backtest(
                df=df,
                short_span=self.short_spin.value(),
                long_span=self.long_spin.value(),
                initial_cash=self.amount_spin.value(),
                fee_per_trade=self.fee_spin.value(),
            )
            self._result = result
            self._render_result(result)
        except Exception as exc:
            title = "Fehler" if self._lang == "de" else "Error"
            msg = f"Backtest fehlgeschlagen: {exc}" if self._lang == "de" else f"Backtest failed: {exc}"
            QMessageBox.critical(self, title, msg)

    def _render_result(self, result: BacktestResult) -> None:
        for key, lbl in self.metric_labels.items():
            val = result.metrics.get(key, "-")
            lbl.setText(f"{val:,.2f}" if isinstance(val, float) else str(val))
        self.plot.clear()
        self.plot.addLegend()
        eq = result.equity_curve
        if not eq.empty:
            x = (pd.to_datetime(eq.index).astype("int64") // 10**9).to_numpy()
            if "strategy" in eq:
                self.plot.plot(x, eq["strategy"], pen=pg.mkPen("#22c55e", width=2), name="Strategie")
            if "buy_and_hold" in eq:
                self.plot.plot(x, eq["buy_and_hold"], pen=pg.mkPen("#10b981", width=2), name="Buy & Hold")
        trades = result.trades
        headers = ["Datum", "Side", "Preis", "Shares", "Amount", "Fee"]
        self.trades_table.clear()
        self.trades_table.setRowCount(len(trades))
        self.trades_table.setColumnCount(len(headers))
        self.trades_table.setHorizontalHeaderLabels(headers)
        for i, trade in enumerate(trades):
            self.trades_table.setItem(i, 0, QTableWidgetItem(str(trade.get("date"))))
            self.trades_table.setItem(i, 1, QTableWidgetItem(trade.get("side", "")))
            self.trades_table.setItem(i, 2, QTableWidgetItem(f"{trade.get('price', 0):,.2f}"))
            self.trades_table.setItem(i, 3, QTableWidgetItem(str(trade.get("shares", 0))))
            self.trades_table.setItem(i, 4, QTableWidgetItem(f"{trade.get('amount', 0):,.2f}"))
            self.trades_table.setItem(i, 5, QTableWidgetItem(f"{trade.get('fee', 0):,.2f}"))
        self.trades_table.resizeColumnsToContents()


class RsiReversionWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._lang = "de"
        self._result: Optional[BacktestResult] = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.ticker_input = QLineEdit()
        self.range_combo = QComboBox()
        self.range_combo.addItems(["1Y", "3Y", "5Y", "10Y", "Max"])
        self.period_spin = QSpinBox()
        self.period_spin.setRange(2, 200)
        self.period_spin.setValue(14)
        self.oversold_spin = QDoubleSpinBox()
        self.oversold_spin.setRange(1.0, 50.0)
        self.oversold_spin.setValue(30.0)
        self.overbought_spin = QDoubleSpinBox()
        self.overbought_spin.setRange(50.0, 99.0)
        self.overbought_spin.setValue(70.0)
        self.amount_spin = QDoubleSpinBox()
        self.amount_spin.setRange(100.0, 1_000_000.0)
        self.amount_spin.setValue(10_000.0)
        self.fee_spin = QDoubleSpinBox()
        self.fee_spin.setRange(0.0, 1000.0)
        self.fee_spin.setValue(0.0)

        form.addRow("Ticker:", self.ticker_input)
        form.addRow("Zeitraum:", self.range_combo)
        form.addRow("RSI Periode:", self.period_spin)
        form.addRow("Oversold:", self.oversold_spin)
        form.addRow("Overbought:", self.overbought_spin)
        form.addRow("Startkapital:", self.amount_spin)
        form.addRow("Fee/Trade:", self.fee_spin)
        self.run_button = QPushButton("Backtest ausführen")

        row = QHBoxLayout()
        row.addLayout(form)
        row.addWidget(self.run_button)
        layout.addLayout(row)

        self.metrics_form = QFormLayout()
        self.metric_labels = {k: QLabel("-") for k in [
            "n_trades","total_fees","total_invested","final_value",
            "strategy_return_pct","strategy_annual_return_pct",
            "bh_return_pct","bh_annual_return_pct",
            "outperformance_total_pct","outperformance_annual_pct"
        ]}
        for key, lbl in self.metric_labels.items():
            self.metrics_form.addRow(key.replace("_", " ").title(), lbl)

        self.plot = pg.PlotWidget()
        self.plot.addLegend()
        self.plot.showGrid(x=True, y=True, alpha=0.2)

        self.trades_table = QTableWidget()
        self.trades_table.setSortingEnabled(True)

        split = QSplitter(Qt.Orientation.Vertical)
        metrics_widget = QWidget()
        metrics_widget.setLayout(self.metrics_form)
        split.addWidget(metrics_widget)
        split.addWidget(self.plot)
        split.setSizes([200, 400])

        main_splitter = QSplitter(Qt.Orientation.Vertical)
        main_splitter.addWidget(split)
        main_splitter.addWidget(self.trades_table)
        main_splitter.setSizes([500, 300])

        layout.addWidget(main_splitter)
        self.run_button.clicked.connect(self.on_run_clicked)

    def set_language(self, lang: str) -> None:
        self._lang = lang
        self.run_button.setText("Backtest ausführen" if lang == "de" else "Run backtest")

    def on_run_clicked(self) -> None:
        ticker = self.ticker_input.text().strip().upper()
        if not ticker:
            QMessageBox.warning(self, "Fehler" if self._lang == "de" else "Error",
                                "Bitte einen Ticker eingeben." if self._lang == "de" else "Please enter a ticker.")
            return
        try:
            start = None
            end = None
            period_sel = self.range_combo.currentText()
            if period_sel != "Max":
                end = pd.Timestamp.today().date()
                if period_sel == "1Y":
                    start = end - pd.DateOffset(years=1)
                elif period_sel == "3Y":
                    start = end - pd.DateOffset(years=3)
                elif period_sel == "5Y":
                    start = end - pd.DateOffset(years=5)
                elif period_sel == "10Y":
                    start = end - pd.DateOffset(years=10)
                start = start.date()
            df = load_price_data(ticker, start=start, end=end)
            result = run_rsi_reversion_backtest(
                df=df,
                period=self.period_spin.value(),
                oversold=self.oversold_spin.value(),
                overbought=self.overbought_spin.value(),
                initial_cash=self.amount_spin.value(),
                fee_per_trade=self.fee_spin.value(),
            )
            self._result = result
            self._render_result(result)
        except Exception as exc:
            title = "Fehler" if self._lang == "de" else "Error"
            msg = f"Backtest fehlgeschlagen: {exc}" if self._lang == "de" else f"Backtest failed: {exc}"
            QMessageBox.critical(self, title, msg)

    def _render_result(self, result: BacktestResult) -> None:
        for key, lbl in self.metric_labels.items():
            val = result.metrics.get(key, "-")
            lbl.setText(f"{val:,.2f}" if isinstance(val, float) else str(val))
        self.plot.clear()
        self.plot.addLegend()
        eq = result.equity_curve
        if not eq.empty:
            x = (pd.to_datetime(eq.index).astype("int64") // 10**9).to_numpy()
            if "strategy" in eq:
                self.plot.plot(x, eq["strategy"], pen=pg.mkPen("#0ea5e9", width=2), name="Strategie")
            if "buy_and_hold" in eq:
                self.plot.plot(x, eq["buy_and_hold"], pen=pg.mkPen("#10b981", width=2), name="Buy & Hold")
        trades = result.trades
        headers = ["Datum", "Side", "Preis", "Shares", "Amount", "Fee"]
        self.trades_table.clear()
        self.trades_table.setRowCount(len(trades))
        self.trades_table.setColumnCount(len(headers))
        self.trades_table.setHorizontalHeaderLabels(headers)
        for i, trade in enumerate(trades):
            self.trades_table.setItem(i, 0, QTableWidgetItem(str(trade.get("date"))))
            self.trades_table.setItem(i, 1, QTableWidgetItem(trade.get("side", "")))
            self.trades_table.setItem(i, 2, QTableWidgetItem(f"{trade.get('price', 0):,.2f}"))
            self.trades_table.setItem(i, 3, QTableWidgetItem(str(trade.get("shares", 0))))
            self.trades_table.setItem(i, 4, QTableWidgetItem(f"{trade.get('amount', 0):,.2f}"))
            self.trades_table.setItem(i, 5, QTableWidgetItem(f"{trade.get('fee', 0):,.2f}"))
        self.trades_table.resizeColumnsToContents()


class BollingerReversionWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._lang = "de"
        self._result: Optional[BacktestResult] = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.ticker_input = QLineEdit()
        self.range_combo = QComboBox()
        self.range_combo.addItems(["1Y", "3Y", "5Y", "10Y", "Max"])
        self.window_spin = QSpinBox()
        self.window_spin.setRange(5, 200)
        self.window_spin.setValue(20)
        self.std_spin = QDoubleSpinBox()
        self.std_spin.setRange(0.5, 5.0)
        self.std_spin.setSingleStep(0.1)
        self.std_spin.setValue(2.0)
        self.amount_spin = QDoubleSpinBox()
        self.amount_spin.setRange(100.0, 1_000_000.0)
        self.amount_spin.setValue(10_000.0)
        self.fee_spin = QDoubleSpinBox()
        self.fee_spin.setRange(0.0, 1000.0)
        self.fee_spin.setValue(0.0)
        form.addRow("Ticker:", self.ticker_input)
        form.addRow("Zeitraum:", self.range_combo)
        form.addRow("Fenster:", self.window_spin)
        form.addRow("Std-Faktor:", self.std_spin)
        form.addRow("Startkapital:", self.amount_spin)
        form.addRow("Fee/Trade:", self.fee_spin)
        self.run_button = QPushButton("Backtest ausführen")

        row = QHBoxLayout()
        row.addLayout(form)
        row.addWidget(self.run_button)
        layout.addLayout(row)

        self.metrics_form = QFormLayout()
        self.metric_labels = {k: QLabel("-") for k in [
            "n_trades","total_fees","total_invested","final_value",
            "strategy_return_pct","strategy_annual_return_pct",
            "bh_return_pct","bh_annual_return_pct",
            "outperformance_total_pct","outperformance_annual_pct"
        ]}
        for key, lbl in self.metric_labels.items():
            self.metrics_form.addRow(key.replace("_", " ").title(), lbl)

        self.plot = pg.PlotWidget()
        self.plot.addLegend()
        self.plot.showGrid(x=True, y=True, alpha=0.2)

        self.trades_table = QTableWidget()
        self.trades_table.setSortingEnabled(True)

        split = QSplitter(Qt.Orientation.Vertical)
        metrics_widget = QWidget()
        metrics_widget.setLayout(self.metrics_form)
        split.addWidget(metrics_widget)
        split.addWidget(self.plot)
        split.setSizes([200, 400])

        main_splitter = QSplitter(Qt.Orientation.Vertical)
        main_splitter.addWidget(split)
        main_splitter.addWidget(self.trades_table)
        main_splitter.setSizes([500, 300])

        layout.addWidget(main_splitter)
        self.run_button.clicked.connect(self.on_run_clicked)

    def set_language(self, lang: str) -> None:
        self._lang = lang
        self.run_button.setText("Backtest ausführen" if lang == "de" else "Run backtest")

    def on_run_clicked(self) -> None:
        ticker = self.ticker_input.text().strip().upper()
        if not ticker:
            QMessageBox.warning(self, "Fehler" if self._lang == "de" else "Error",
                                "Bitte einen Ticker eingeben." if self._lang == "de" else "Please enter a ticker.")
            return
        try:
            start = None
            end = None
            period_sel = self.range_combo.currentText()
            if period_sel != "Max":
                end = pd.Timestamp.today().date()
                if period_sel == "1Y":
                    start = end - pd.DateOffset(years=1)
                elif period_sel == "3Y":
                    start = end - pd.DateOffset(years=3)
                elif period_sel == "5Y":
                    start = end - pd.DateOffset(years=5)
                elif period_sel == "10Y":
                    start = end - pd.DateOffset(years=10)
                start = start.date()
            df = load_price_data(ticker, start=start, end=end)
            result = run_bollinger_reversion_backtest(
                df=df,
                window=self.window_spin.value(),
                num_std=self.std_spin.value(),
                initial_cash=self.amount_spin.value(),
                fee_per_trade=self.fee_spin.value(),
            )
            self._result = result
            self._render_result(result)
        except Exception as exc:
            title = "Fehler" if self._lang == "de" else "Error"
            msg = f"Backtest fehlgeschlagen: {exc}" if self._lang == "de" else f"Backtest failed: {exc}"
            QMessageBox.critical(self, title, msg)

    def _render_result(self, result: BacktestResult) -> None:
        for key, lbl in self.metric_labels.items():
            val = result.metrics.get(key, "-")
            lbl.setText(f"{val:,.2f}" if isinstance(val, float) else str(val))
        self.plot.clear()
        self.plot.addLegend()
        eq = result.equity_curve
        if not eq.empty:
            x = (pd.to_datetime(eq.index).astype("int64") // 10**9).to_numpy()
            if "strategy" in eq:
                self.plot.plot(x, eq["strategy"], pen=pg.mkPen("#fb7185", width=2), name="Strategie")
            if "buy_and_hold" in eq:
                self.plot.plot(x, eq["buy_and_hold"], pen=pg.mkPen("#10b981", width=2), name="Buy & Hold")
        trades = result.trades
        headers = ["Datum", "Side", "Preis", "Shares", "Amount", "Fee"]
        self.trades_table.clear()
        self.trades_table.setRowCount(len(trades))
        self.trades_table.setColumnCount(len(headers))
        self.trades_table.setHorizontalHeaderLabels(headers)
        for i, trade in enumerate(trades):
            self.trades_table.setItem(i, 0, QTableWidgetItem(str(trade.get("date"))))
            self.trades_table.setItem(i, 1, QTableWidgetItem(trade.get("side", "")))
            self.trades_table.setItem(i, 2, QTableWidgetItem(f"{trade.get('price', 0):,.2f}"))
            self.trades_table.setItem(i, 3, QTableWidgetItem(str(trade.get("shares", 0))))
            self.trades_table.setItem(i, 4, QTableWidgetItem(f"{trade.get('amount', 0):,.2f}"))
            self.trades_table.setItem(i, 5, QTableWidgetItem(f"{trade.get('fee', 0):,.2f}"))
        self.trades_table.resizeColumnsToContents()


class MomentumWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._lang = "de"
        self._result: Optional[BacktestResult] = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.ticker_input = QLineEdit()
        self.range_combo = QComboBox()
        self.range_combo.addItems(["1Y", "3Y", "5Y", "10Y", "Max"])
        self.amount_spin = QDoubleSpinBox()
        self.amount_spin.setRange(100.0, 1_000_000.0)
        self.amount_spin.setValue(10_000.0)
        self.fee_spin = QDoubleSpinBox()
        self.fee_spin.setRange(0.0, 1000.0)
        self.fee_spin.setValue(0.0)
        form.addRow("Ticker:", self.ticker_input)
        form.addRow("Zeitraum:", self.range_combo)
        form.addRow("Startkapital:", self.amount_spin)
        form.addRow("Fee/Trade:", self.fee_spin)
        self.run_button = QPushButton("Backtest ausführen")

        row = QHBoxLayout()
        row.addLayout(form)
        row.addWidget(self.run_button)
        layout.addLayout(row)

        self.metrics_form = QFormLayout()
        self.metric_labels = {k: QLabel("-") for k in [
            "n_trades","total_fees","total_invested","final_value","strategy_profit","bh_profit",
            "strategy_return_pct","strategy_annual_return_pct","bh_return_pct","bh_annual_return_pct",
            "outperformance_total_pct","outperformance_annual_pct"
        ]}
        for key, lbl in self.metric_labels.items():
            self.metrics_form.addRow(key.replace("_", " ").title(), lbl)

        self.plot = pg.PlotWidget()
        self.plot.addLegend()
        self.plot.showGrid(x=True, y=True, alpha=0.2)

        self.trades_table = QTableWidget()
        self.trades_table.setSortingEnabled(True)

        split = QSplitter(Qt.Orientation.Vertical)
        metrics_widget = QWidget()
        metrics_widget.setLayout(self.metrics_form)
        split.addWidget(metrics_widget)
        split.addWidget(self.plot)
        split.setSizes([200, 400])

        main_splitter = QSplitter(Qt.Orientation.Vertical)
        main_splitter.addWidget(split)
        main_splitter.addWidget(self.trades_table)
        main_splitter.setSizes([500, 300])

        layout.addWidget(main_splitter)
        self.run_button.clicked.connect(self.on_run_clicked)

    def set_language(self, lang: str) -> None:
        self._lang = lang
        self.run_button.setText("Backtest ausführen" if lang == "de" else "Run backtest")

    def on_run_clicked(self) -> None:
        ticker = self.ticker_input.text().strip().upper()
        if not ticker:
            QMessageBox.warning(self, "Fehler" if self._lang == "de" else "Error",
                                "Bitte einen Ticker eingeben." if self._lang == "de" else "Please enter a ticker.")
            return
        try:
            start = None
            end = None
            period_sel = self.range_combo.currentText()
            if period_sel != "Max":
                end = pd.Timestamp.today().date()
                if period_sel == "1Y":
                    start = end - pd.DateOffset(years=1)
                elif period_sel == "3Y":
                    start = end - pd.DateOffset(years=3)
                elif period_sel == "5Y":
                    start = end - pd.DateOffset(years=5)
                elif period_sel == "10Y":
                    start = end - pd.DateOffset(years=10)
                start = start.date()
            df = load_price_data(ticker, start=start, end=end)
            result = run_momentum_12_1_backtest(
                df=df,
                initial_cash=self.amount_spin.value(),
                fee_per_trade=self.fee_spin.value(),
            )
            self._result = result
            self._render_result(result)
        except Exception as exc:
            title = "Fehler" if self._lang == "de" else "Error"
            msg = f"Backtest fehlgeschlagen: {exc}" if self._lang == "de" else f"Backtest failed: {exc}"
            QMessageBox.critical(self, title, msg)

    def _render_result(self, result: BacktestResult) -> None:
        for key, lbl in self.metric_labels.items():
            val = result.metrics.get(key, "-")
            lbl.setText(f"{val:,.2f}" if isinstance(val, float) else str(val))
        try:
            total_inv = float(result.metrics.get("total_invested", 0.0))
            strat_final = float(result.metrics.get("final_value", 0.0))
            bh_final = float(result.equity_curve["buy_and_hold"].iloc[-1]) if not result.equity_curve.empty else 0.0
            self.metric_labels["strategy_profit"].setText(f"{strat_final - total_inv:,.2f}")
            self.metric_labels["bh_profit"].setText(f"{bh_final - total_inv:,.2f}")
        except Exception:
            self.metric_labels["strategy_profit"].setText("-")
            self.metric_labels["bh_profit"].setText("-")

        self.plot.clear()
        self.plot.addLegend()
        eq = result.equity_curve
        if not eq.empty:
            x = (pd.to_datetime(eq.index).astype("int64") // 10**9).to_numpy()
            if "strategy" in eq:
                self.plot.plot(x, eq["strategy"], pen=pg.mkPen("#eab308", width=2), name="Strategie")
            if "buy_and_hold" in eq:
                self.plot.plot(x, eq["buy_and_hold"], pen=pg.mkPen("#10b981", width=2), name="Buy & Hold")
        trades = result.trades
        headers = ["Datum", "Side", "Preis", "Shares", "Amount", "Fee"]
        self.trades_table.clear()
        self.trades_table.setRowCount(len(trades))
        self.trades_table.setColumnCount(len(headers))
        self.trades_table.setHorizontalHeaderLabels(headers)
        for i, trade in enumerate(trades):
            self.trades_table.setItem(i, 0, QTableWidgetItem(str(trade.get("date"))))
            self.trades_table.setItem(i, 1, QTableWidgetItem(trade.get("side", "")))
            self.trades_table.setItem(i, 2, QTableWidgetItem(f"{trade.get('price', 0):,.2f}"))
            self.trades_table.setItem(i, 3, QTableWidgetItem(str(trade.get("shares", 0))))
            self.trades_table.setItem(i, 4, QTableWidgetItem(f"{trade.get('amount', 0):,.2f}"))
            self.trades_table.setItem(i, 5, QTableWidgetItem(f"{trade.get('fee', 0):,.2f}"))
        self.trades_table.resizeColumnsToContents()


class DonchianWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._lang = "de"
        self._result: Optional[BacktestResult] = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.ticker_input = QLineEdit()
        self.range_combo = QComboBox()
        self.range_combo.addItems(["1Y", "3Y", "5Y", "10Y", "Max"])
        self.window_spin = QSpinBox()
        self.window_spin.setRange(5, 200)
        self.window_spin.setValue(55)
        self.amount_spin = QDoubleSpinBox()
        self.amount_spin.setRange(100.0, 1_000_000.0)
        self.amount_spin.setValue(10_000.0)
        self.fee_spin = QDoubleSpinBox()
        self.fee_spin.setRange(0.0, 1000.0)
        self.fee_spin.setValue(0.0)
        form.addRow("Ticker:", self.ticker_input)
        form.addRow("Zeitraum:", self.range_combo)
        form.addRow("Fenster:", self.window_spin)
        form.addRow("Startkapital:", self.amount_spin)
        form.addRow("Fee/Trade:", self.fee_spin)
        self.run_button = QPushButton("Backtest ausführen")

        row = QHBoxLayout()
        row.addLayout(form)
        row.addWidget(self.run_button)
        layout.addLayout(row)

        self.metrics_form = QFormLayout()
        self.metric_labels = {k: QLabel("-") for k in [
            "n_trades","total_fees","total_invested","final_value","strategy_profit","bh_profit",
            "strategy_return_pct","strategy_annual_return_pct","bh_return_pct","bh_annual_return_pct",
            "outperformance_total_pct","outperformance_annual_pct"
        ]}
        for key, lbl in self.metric_labels.items():
            self.metrics_form.addRow(key.replace("_", " ").title(), lbl)

        self.plot = pg.PlotWidget()
        self.plot.addLegend()
        self.plot.showGrid(x=True, y=True, alpha=0.2)

        self.trades_table = QTableWidget()
        self.trades_table.setSortingEnabled(True)

        split = QSplitter(Qt.Orientation.Vertical)
        metrics_widget = QWidget()
        metrics_widget.setLayout(self.metrics_form)
        split.addWidget(metrics_widget)
        split.addWidget(self.plot)
        split.setSizes([200, 400])

        main_splitter = QSplitter(Qt.Orientation.Vertical)
        main_splitter.addWidget(split)
        main_splitter.addWidget(self.trades_table)
        main_splitter.setSizes([500, 300])

        layout.addWidget(main_splitter)
        self.run_button.clicked.connect(self.on_run_clicked)

    def set_language(self, lang: str) -> None:
        self._lang = lang
        self.run_button.setText("Backtest ausführen" if lang == "de" else "Run backtest")

    def on_run_clicked(self) -> None:
        ticker = self.ticker_input.text().strip().upper()
        if not ticker:
            QMessageBox.warning(self, "Fehler" if self._lang == "de" else "Error",
                                "Bitte einen Ticker eingeben." if self._lang == "de" else "Please enter a ticker.")
            return
        try:
            start = None
            end = None
            period_sel = self.range_combo.currentText()
            if period_sel != "Max":
                end = pd.Timestamp.today().date()
                if period_sel == "1Y":
                    start = end - pd.DateOffset(years=1)
                elif period_sel == "3Y":
                    start = end - pd.DateOffset(years=3)
                elif period_sel == "5Y":
                    start = end - pd.DateOffset(years=5)
                elif period_sel == "10Y":
                    start = end - pd.DateOffset(years=10)
                start = start.date()
            df = load_price_data(ticker, start=start, end=end)
            result = run_donchian_breakout_backtest(
                df=df,
                window=self.window_spin.value(),
                initial_cash=self.amount_spin.value(),
                fee_per_trade=self.fee_spin.value(),
            )
            self._result = result
            self._render_result(result)
        except Exception as exc:
            title = "Fehler" if self._lang == "de" else "Error"
            msg = f"Backtest fehlgeschlagen: {exc}" if self._lang == "de" else f"Backtest failed: {exc}"
            QMessageBox.critical(self, title, msg)

    def _render_result(self, result: BacktestResult) -> None:
        for key, lbl in self.metric_labels.items():
            val = result.metrics.get(key, "-")
            lbl.setText(f"{val:,.2f}" if isinstance(val, float) else str(val))
        try:
            total_inv = float(result.metrics.get("total_invested", 0.0))
            strat_final = float(result.metrics.get("final_value", 0.0))
            bh_final = float(result.equity_curve["buy_and_hold"].iloc[-1]) if not result.equity_curve.empty else 0.0
            self.metric_labels["strategy_profit"].setText(f"{strat_final - total_inv:,.2f}")
            self.metric_labels["bh_profit"].setText(f"{bh_final - total_inv:,.2f}")
        except Exception:
            self.metric_labels["strategy_profit"].setText("-")
            self.metric_labels["bh_profit"].setText("-")

        self.plot.clear()
        self.plot.addLegend()
        eq = result.equity_curve
        if not eq.empty:
            x = (pd.to_datetime(eq.index).astype("int64") // 10**9).to_numpy()
            if "strategy" in eq:
                self.plot.plot(x, eq["strategy"], pen=pg.mkPen("#a855f7", width=2), name="Strategie")
            if "buy_and_hold" in eq:
                self.plot.plot(x, eq["buy_and_hold"], pen=pg.mkPen("#10b981", width=2), name="Buy & Hold")
        trades = result.trades
        headers = ["Datum", "Side", "Preis", "Shares", "Amount", "Fee"]
        self.trades_table.clear()
        self.trades_table.setRowCount(len(trades))
        self.trades_table.setColumnCount(len(headers))
        self.trades_table.setHorizontalHeaderLabels(headers)
        for i, trade in enumerate(trades):
            self.trades_table.setItem(i, 0, QTableWidgetItem(str(trade.get("date"))))
            self.trades_table.setItem(i, 1, QTableWidgetItem(trade.get("side", "")))
            self.trades_table.setItem(i, 2, QTableWidgetItem(f"{trade.get('price', 0):,.2f}"))
            self.trades_table.setItem(i, 3, QTableWidgetItem(str(trade.get("shares", 0))))
            self.trades_table.setItem(i, 4, QTableWidgetItem(f"{trade.get('amount', 0):,.2f}"))
            self.trades_table.setItem(i, 5, QTableWidgetItem(f"{trade.get('fee', 0):,.2f}"))
        self.trades_table.resizeColumnsToContents()


class MacdWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._lang = "de"
        self._result: Optional[BacktestResult] = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.ticker_input = QLineEdit()
        self.range_combo = QComboBox()
        self.range_combo.addItems(["1Y", "3Y", "5Y", "10Y", "Max"])
        self.fast_spin = QSpinBox()
        self.fast_spin.setRange(2, 50)
        self.fast_spin.setValue(12)
        self.slow_spin = QSpinBox()
        self.slow_spin.setRange(5, 200)
        self.slow_spin.setValue(26)
        self.signal_spin = QSpinBox()
        self.signal_spin.setRange(2, 50)
        self.signal_spin.setValue(9)
        self.amount_spin = QDoubleSpinBox()
        self.amount_spin.setRange(100.0, 1_000_000.0)
        self.amount_spin.setValue(10_000.0)
        self.fee_spin = QDoubleSpinBox()
        self.fee_spin.setRange(0.0, 1000.0)
        self.fee_spin.setValue(0.0)
        form.addRow("Ticker:", self.ticker_input)
        form.addRow("Zeitraum:", self.range_combo)
        form.addRow("MACD Fast:", self.fast_spin)
        form.addRow("MACD Slow:", self.slow_spin)
        form.addRow("MACD Signal:", self.signal_spin)
        form.addRow("Startkapital:", self.amount_spin)
        form.addRow("Fee/Trade:", self.fee_spin)
        self.run_button = QPushButton("Backtest ausführen")

        row = QHBoxLayout()
        row.addLayout(form)
        row.addWidget(self.run_button)
        layout.addLayout(row)

        self.metrics_form = QFormLayout()
        self.metric_labels = {k: QLabel("-") for k in [
            "n_trades","total_fees","total_invested","final_value","strategy_profit","bh_profit",
            "strategy_return_pct","strategy_annual_return_pct","bh_return_pct","bh_annual_return_pct",
            "outperformance_total_pct","outperformance_annual_pct"
        ]}
        for key, lbl in self.metric_labels.items():
            self.metrics_form.addRow(key.replace("_", " ").title(), lbl)

        self.plot = pg.PlotWidget()
        self.plot.addLegend()
        self.plot.showGrid(x=True, y=True, alpha=0.2)

        self.trades_table = QTableWidget()
        self.trades_table.setSortingEnabled(True)

        split = QSplitter(Qt.Orientation.Vertical)
        metrics_widget = QWidget()
        metrics_widget.setLayout(self.metrics_form)
        split.addWidget(metrics_widget)
        split.addWidget(self.plot)
        split.setSizes([200, 400])

        main_splitter = QSplitter(Qt.Orientation.Vertical)
        main_splitter.addWidget(split)
        main_splitter.addWidget(self.trades_table)
        main_splitter.setSizes([500, 300])

        layout.addWidget(main_splitter)
        self.run_button.clicked.connect(self.on_run_clicked)

    def set_language(self, lang: str) -> None:
        self._lang = lang
        self.run_button.setText("Backtest ausführen" if lang == "de" else "Run backtest")

    def on_run_clicked(self) -> None:
        ticker = self.ticker_input.text().strip().upper()
        if not ticker:
            QMessageBox.warning(self, "Fehler" if self._lang == "de" else "Error",
                                "Bitte einen Ticker eingeben." if self._lang == "de" else "Please enter a ticker.")
            return
        try:
            start = None
            end = None
            period_sel = self.range_combo.currentText()
            if period_sel != "Max":
                end = pd.Timestamp.today().date()
                if period_sel == "1Y":
                    start = end - pd.DateOffset(years=1)
                elif period_sel == "3Y":
                    start = end - pd.DateOffset(years=3)
                elif period_sel == "5Y":
                    start = end - pd.DateOffset(years=5)
                elif period_sel == "10Y":
                    start = end - pd.DateOffset(years=10)
                start = start.date()
            df = load_price_data(ticker, start=start, end=end)
            result = run_macd_backtest(
                df=df,
                fast=self.fast_spin.value(),
                slow=self.slow_spin.value(),
                signal=self.signal_spin.value(),
                initial_cash=self.amount_spin.value(),
                fee_per_trade=self.fee_spin.value(),
            )
            self._result = result
            self._render_result(result)
        except Exception as exc:
            title = "Fehler" if self._lang == "de" else "Error"
            msg = f"Backtest fehlgeschlagen: {exc}" if self._lang == "de" else f"Backtest failed: {exc}"
            QMessageBox.critical(self, title, msg)

    def _render_result(self, result: BacktestResult) -> None:
        for key, lbl in self.metric_labels.items():
            val = result.metrics.get(key, "-")
            lbl.setText(f"{val:,.2f}" if isinstance(val, float) else str(val))
        try:
            total_inv = float(result.metrics.get("total_invested", 0.0))
            strat_final = float(result.metrics.get("final_value", 0.0))
            bh_final = float(result.equity_curve["buy_and_hold"].iloc[-1]) if not result.equity_curve.empty else 0.0
            self.metric_labels["strategy_profit"].setText(f"{strat_final - total_inv:,.2f}")
            self.metric_labels["bh_profit"].setText(f"{bh_final - total_inv:,.2f}")
        except Exception:
            self.metric_labels["strategy_profit"].setText("-")
            self.metric_labels["bh_profit"].setText("-")

        self.plot.clear()
        self.plot.addLegend()
        eq = result.equity_curve
        if not eq.empty:
            x = (pd.to_datetime(eq.index).astype("int64") // 10**9).to_numpy()
            if "strategy" in eq:
                self.plot.plot(x, eq["strategy"], pen=pg.mkPen("#38bdf8", width=2), name="Strategie")
            if "buy_and_hold" in eq:
                self.plot.plot(x, eq["buy_and_hold"], pen=pg.mkPen("#10b981", width=2), name="Buy & Hold")
        trades = result.trades
        headers = ["Datum", "Side", "Preis", "Shares", "Amount", "Fee"]
        self.trades_table.clear()
        self.trades_table.setRowCount(len(trades))
        self.trades_table.setColumnCount(len(headers))
        self.trades_table.setHorizontalHeaderLabels(headers)
        for i, trade in enumerate(trades):
            self.trades_table.setItem(i, 0, QTableWidgetItem(str(trade.get("date"))))
            self.trades_table.setItem(i, 1, QTableWidgetItem(trade.get("side", "")))
            self.trades_table.setItem(i, 2, QTableWidgetItem(f"{trade.get('price', 0):,.2f}"))
            self.trades_table.setItem(i, 3, QTableWidgetItem(str(trade.get("shares", 0))))
            self.trades_table.setItem(i, 4, QTableWidgetItem(f"{trade.get('amount', 0):,.2f}"))
            self.trades_table.setItem(i, 5, QTableWidgetItem(f"{trade.get('fee', 0):,.2f}"))
        self.trades_table.resizeColumnsToContents()


class StochasticWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._lang = "de"
        self._result: Optional[BacktestResult] = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.ticker_input = QLineEdit()
        self.range_combo = QComboBox()
        self.range_combo.addItems(["1Y", "3Y", "5Y", "10Y", "Max"])
        self.k_spin = QSpinBox()
        self.k_spin.setRange(5, 50)
        self.k_spin.setValue(14)
        self.d_spin = QSpinBox()
        self.d_spin.setRange(2, 20)
        self.d_spin.setValue(3)
        self.oversold_spin = QDoubleSpinBox()
        self.oversold_spin.setRange(1.0, 50.0)
        self.oversold_spin.setValue(20.0)
        self.overbought_spin = QDoubleSpinBox()
        self.overbought_spin.setRange(50.0, 99.0)
        self.overbought_spin.setValue(80.0)
        self.amount_spin = QDoubleSpinBox()
        self.amount_spin.setRange(100.0, 1_000_000.0)
        self.amount_spin.setValue(10_000.0)
        self.fee_spin = QDoubleSpinBox()
        self.fee_spin.setRange(0.0, 1000.0)
        self.fee_spin.setValue(0.0)
        form.addRow("Ticker:", self.ticker_input)
        form.addRow("Zeitraum:", self.range_combo)
        form.addRow("%K Periode:", self.k_spin)
        form.addRow("%D Periode:", self.d_spin)
        form.addRow("Oversold:", self.oversold_spin)
        form.addRow("Overbought:", self.overbought_spin)
        form.addRow("Startkapital:", self.amount_spin)
        form.addRow("Fee/Trade:", self.fee_spin)
        self.run_button = QPushButton("Backtest ausführen")

        row = QHBoxLayout()
        row.addLayout(form)
        row.addWidget(self.run_button)
        layout.addLayout(row)

        self.metrics_form = QFormLayout()
        self.metric_labels = {k: QLabel("-") for k in [
            "n_trades","total_fees","total_invested","final_value","strategy_profit","bh_profit",
            "strategy_return_pct","strategy_annual_return_pct","bh_return_pct","bh_annual_return_pct",
            "outperformance_total_pct","outperformance_annual_pct"
        ]}
        for key, lbl in self.metric_labels.items():
            self.metrics_form.addRow(key.replace("_", " ").title(), lbl)

        self.plot = pg.PlotWidget()
        self.plot.addLegend()
        self.plot.showGrid(x=True, y=True, alpha=0.2)

        self.trades_table = QTableWidget()
        self.trades_table.setSortingEnabled(True)

        split = QSplitter(Qt.Orientation.Vertical)
        metrics_widget = QWidget()
        metrics_widget.setLayout(self.metrics_form)
        split.addWidget(metrics_widget)
        split.addWidget(self.plot)
        split.setSizes([200, 400])

        main_splitter = QSplitter(Qt.Orientation.Vertical)
        main_splitter.addWidget(split)
        main_splitter.addWidget(self.trades_table)
        main_splitter.setSizes([500, 300])

        layout.addWidget(main_splitter)
        self.run_button.clicked.connect(self.on_run_clicked)

    def set_language(self, lang: str) -> None:
        self._lang = lang
        self.run_button.setText("Backtest ausführen" if lang == "de" else "Run backtest")

    def on_run_clicked(self) -> None:
        ticker = self.ticker_input.text().strip().upper()
        if not ticker:
            QMessageBox.warning(self, "Fehler" if self._lang == "de" else "Error",
                                "Bitte einen Ticker eingeben." if self._lang == "de" else "Please enter a ticker.")
            return
        try:
            start = None
            end = None
            period_sel = self.range_combo.currentText()
            if period_sel != "Max":
                end = pd.Timestamp.today().date()
                if period_sel == "1Y":
                    start = end - pd.DateOffset(years=1)
                elif period_sel == "3Y":
                    start = end - pd.DateOffset(years=3)
                elif period_sel == "5Y":
                    start = end - pd.DateOffset(years=5)
                elif period_sel == "10Y":
                    start = end - pd.DateOffset(years=10)
                start = start.date()
            df = load_price_data(ticker, start=start, end=end)
            result = run_stoch_osc_backtest(
                df=df,
                k_period=self.k_spin.value(),
                d_period=self.d_spin.value(),
                oversold=self.oversold_spin.value(),
                overbought=self.overbought_spin.value(),
                initial_cash=self.amount_spin.value(),
                fee_per_trade=self.fee_spin.value(),
            )
            self._result = result
            self._render_result(result)
        except Exception as exc:
            title = "Fehler" if self._lang == "de" else "Error"
            msg = f"Backtest fehlgeschlagen: {exc}" if self._lang == "de" else f"Backtest failed: {exc}"
            QMessageBox.critical(self, title, msg)

    def _render_result(self, result: BacktestResult) -> None:
        for key, lbl in self.metric_labels.items():
            val = result.metrics.get(key, "-")
            lbl.setText(f"{val:,.2f}" if isinstance(val, float) else str(val))
        try:
            total_inv = float(result.metrics.get("total_invested", 0.0))
            strat_final = float(result.metrics.get("final_value", 0.0))
            bh_final = float(result.equity_curve["buy_and_hold"].iloc[-1]) if not result.equity_curve.empty else 0.0
            self.metric_labels["strategy_profit"].setText(f"{strat_final - total_inv:,.2f}")
            self.metric_labels["bh_profit"].setText(f"{bh_final - total_inv:,.2f}")
        except Exception:
            self.metric_labels["strategy_profit"].setText("-")
            self.metric_labels["bh_profit"].setText("-")

        self.plot.clear()
        self.plot.addLegend()
        eq = result.equity_curve
        if not eq.empty:
            x = (pd.to_datetime(eq.index).astype("int64") // 10**9).to_numpy()
            if "strategy" in eq:
                self.plot.plot(x, eq["strategy"], pen=pg.mkPen("#f59e0b", width=2), name="Strategie")
            if "buy_and_hold" in eq:
                self.plot.plot(x, eq["buy_and_hold"], pen=pg.mkPen("#10b981", width=2), name="Buy & Hold")
        trades = result.trades
        headers = ["Datum", "Side", "Preis", "Shares", "Amount", "Fee"]
        self.trades_table.clear()
        self.trades_table.setRowCount(len(trades))
        self.trades_table.setColumnCount(len(headers))
        self.trades_table.setHorizontalHeaderLabels(headers)
        for i, trade in enumerate(trades):
            self.trades_table.setItem(i, 0, QTableWidgetItem(str(trade.get("date"))))
            self.trades_table.setItem(i, 1, QTableWidgetItem(trade.get("side", "")))
            self.trades_table.setItem(i, 2, QTableWidgetItem(f"{trade.get('price', 0):,.2f}"))
            self.trades_table.setItem(i, 3, QTableWidgetItem(str(trade.get("shares", 0))))
            self.trades_table.setItem(i, 4, QTableWidgetItem(f"{trade.get('amount', 0):,.2f}"))
            self.trades_table.setItem(i, 5, QTableWidgetItem(f"{trade.get('fee', 0):,.2f}"))
        self.trades_table.resizeColumnsToContents()
class BenchmarkWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._lang = "de"
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        header_form = QFormLayout()
        self.ticker_input = QLineEdit()
        self.range_combo = QComboBox()
        self.range_combo.addItems(["1Y", "3Y", "5Y", "10Y", "Max"])
        self.strategy_combo = QComboBox()
        self._strategies = [
            ("Buy the Dip", "buy"),
            ("SMA Crossover", "sma"),
            ("DCA", "dca"),
            ("EMA Crossover", "ema"),
            ("RSI Reversion", "rsi"),
            ("Bollinger Reversion", "boll"),
            ("Momentum 12-1", "mom"),
            ("Donchian Breakout", "donchian"),
            ("MACD", "macd"),
            ("Stochastic Oscillator", "stoch"),
        ]
        for name, _ in self._strategies:
            self.strategy_combo.addItem(name)

        header_form.addRow("Ticker:", self.ticker_input)
        header_form.addRow("Zeitraum:", self.range_combo)
        header_form.addRow("Backtest against:", self.strategy_combo)
        layout.addLayout(header_form)

        self.param_stack = QStackedWidget()
        self.param_controls: dict[str, dict[str, QWidget]] = {}
        for name, key in self._strategies:
            form = QFormLayout()
            widget = QWidget()
            widget.setLayout(form)
            controls: dict[str, QWidget] = {}
            if key == "buy":
                controls["lookback"] = QSpinBox(); controls["lookback"].setRange(1, 120); controls["lookback"].setValue(5)
                controls["drop"] = QDoubleSpinBox(); controls["drop"].setRange(0.1, 100.0); controls["drop"].setValue(5.0); controls["drop"].setDecimals(2)
                controls["max_trades"] = QSpinBox(); controls["max_trades"].setRange(1, 10); controls["max_trades"].setValue(1)
                controls["amount"] = QDoubleSpinBox(); controls["amount"].setRange(1, 1_000_000); controls["amount"].setValue(1000.0); controls["amount"].setDecimals(2)
                controls["fee"] = QDoubleSpinBox(); controls["fee"].setRange(0.0, 1000.0); controls["fee"].setValue(0.0); controls["fee"].setDecimals(2)
                form.addRow("Lookback Tage:", controls["lookback"])
                form.addRow("Drop %:", controls["drop"])
                form.addRow("Max Trades/Tag:", controls["max_trades"])
                form.addRow("Betrag/Trade:", controls["amount"])
                form.addRow("Fee/Trade:", controls["fee"])
            elif key in {"sma", "ema"}:
                controls["short"] = QSpinBox(); controls["short"].setRange(1, 250); controls["short"].setValue(20 if key == "sma" else 12)
                controls["long"] = QSpinBox(); controls["long"].setRange(2, 400); controls["long"].setValue(50 if key == "sma" else 26)
                controls["initial"] = QDoubleSpinBox(); controls["initial"].setRange(100.0, 1_000_000.0); controls["initial"].setValue(10_000.0)
                controls["fee"] = QDoubleSpinBox(); controls["fee"].setRange(0.0, 1000.0); controls["fee"].setValue(0.0)
                form.addRow("Short:", controls["short"])
                form.addRow("Long:", controls["long"])
                form.addRow("Startkapital:", controls["initial"])
                form.addRow("Fee/Trade:", controls["fee"])
            elif key == "dca":
                controls["interval"] = QSpinBox(); controls["interval"].setRange(5, 120); controls["interval"].setValue(30)
                controls["amount"] = QDoubleSpinBox(); controls["amount"].setRange(10.0, 1_000_000.0); controls["amount"].setValue(500.0)
                controls["fee"] = QDoubleSpinBox(); controls["fee"].setRange(0.0, 1000.0); controls["fee"].setValue(0.0)
                form.addRow("Intervall (Tage):", controls["interval"])
                form.addRow("Betrag/Intervall:", controls["amount"])
                form.addRow("Fee/Trade:", controls["fee"])
            elif key == "rsi":
                controls["period"] = QSpinBox(); controls["period"].setRange(2, 200); controls["period"].setValue(14)
                controls["oversold"] = QDoubleSpinBox(); controls["oversold"].setRange(1.0, 50.0); controls["oversold"].setValue(30.0)
                controls["overbought"] = QDoubleSpinBox(); controls["overbought"].setRange(50.0, 99.0); controls["overbought"].setValue(70.0)
                controls["initial"] = QDoubleSpinBox(); controls["initial"].setRange(100.0, 1_000_000.0); controls["initial"].setValue(10_000.0)
                controls["fee"] = QDoubleSpinBox(); controls["fee"].setRange(0.0, 1000.0); controls["fee"].setValue(0.0)
                form.addRow("RSI Periode:", controls["period"])
                form.addRow("Oversold:", controls["oversold"])
                form.addRow("Overbought:", controls["overbought"])
                form.addRow("Startkapital:", controls["initial"])
                form.addRow("Fee/Trade:", controls["fee"])
            elif key == "boll":
                controls["window"] = QSpinBox(); controls["window"].setRange(5, 200); controls["window"].setValue(20)
                controls["std"] = QDoubleSpinBox(); controls["std"].setRange(0.5, 5.0); controls["std"].setValue(2.0); controls["std"].setSingleStep(0.1)
                controls["initial"] = QDoubleSpinBox(); controls["initial"].setRange(100.0, 1_000_000.0); controls["initial"].setValue(10_000.0)
                controls["fee"] = QDoubleSpinBox(); controls["fee"].setRange(0.0, 1000.0); controls["fee"].setValue(0.0)
                form.addRow("Fenster:", controls["window"])
                form.addRow("Std-Faktor:", controls["std"])
                form.addRow("Startkapital:", controls["initial"])
                form.addRow("Fee/Trade:", controls["fee"])
            elif key == "mom":
                controls["initial"] = QDoubleSpinBox(); controls["initial"].setRange(100.0, 1_000_000.0); controls["initial"].setValue(10_000.0)
                controls["fee"] = QDoubleSpinBox(); controls["fee"].setRange(0.0, 1000.0); controls["fee"].setValue(0.0)
                form.addRow("Startkapital:", controls["initial"])
                form.addRow("Fee/Trade:", controls["fee"])
            elif key == "donchian":
                controls["window"] = QSpinBox(); controls["window"].setRange(5, 200); controls["window"].setValue(55)
                controls["initial"] = QDoubleSpinBox(); controls["initial"].setRange(100.0, 1_000_000.0); controls["initial"].setValue(10_000.0)
                controls["fee"] = QDoubleSpinBox(); controls["fee"].setRange(0.0, 1000.0); controls["fee"].setValue(0.0)
                form.addRow("Fenster:", controls["window"])
                form.addRow("Startkapital:", controls["initial"])
                form.addRow("Fee/Trade:", controls["fee"])
            elif key == "macd":
                controls["fast"] = QSpinBox(); controls["fast"].setRange(2, 50); controls["fast"].setValue(12)
                controls["slow"] = QSpinBox(); controls["slow"].setRange(5, 200); controls["slow"].setValue(26)
                controls["signal"] = QSpinBox(); controls["signal"].setRange(2, 50); controls["signal"].setValue(9)
                controls["initial"] = QDoubleSpinBox(); controls["initial"].setRange(100.0, 1_000_000.0); controls["initial"].setValue(10_000.0)
                controls["fee"] = QDoubleSpinBox(); controls["fee"].setRange(0.0, 1000.0); controls["fee"].setValue(0.0)
                form.addRow("MACD Fast:", controls["fast"])
                form.addRow("MACD Slow:", controls["slow"])
                form.addRow("MACD Signal:", controls["signal"])
                form.addRow("Startkapital:", controls["initial"])
                form.addRow("Fee/Trade:", controls["fee"])
            elif key == "stoch":
                controls["k"] = QSpinBox(); controls["k"].setRange(5, 50); controls["k"].setValue(14)
                controls["d"] = QSpinBox(); controls["d"].setRange(2, 20); controls["d"].setValue(3)
                controls["oversold"] = QDoubleSpinBox(); controls["oversold"].setRange(1.0, 50.0); controls["oversold"].setValue(20.0)
                controls["overbought"] = QDoubleSpinBox(); controls["overbought"].setRange(50.0, 99.0); controls["overbought"].setValue(80.0)
                controls["initial"] = QDoubleSpinBox(); controls["initial"].setRange(100.0, 1_000_000.0); controls["initial"].setValue(10_000.0)
                controls["fee"] = QDoubleSpinBox(); controls["fee"].setRange(0.0, 1000.0); controls["fee"].setValue(0.0)
                form.addRow("%K Periode:", controls["k"])
                form.addRow("%D Periode:", controls["d"])
                form.addRow("Oversold:", controls["oversold"])
                form.addRow("Overbought:", controls["overbought"])
                form.addRow("Startkapital:", controls["initial"])
                form.addRow("Fee/Trade:", controls["fee"])

            self.param_controls[key] = controls
            self.param_stack.addWidget(widget)

        layout.addWidget(self.param_stack)
        self.strategy_combo.currentIndexChanged.connect(self.param_stack.setCurrentIndex)

        self.run_button = QPushButton("Benchmark ausfuehren")
        layout.addWidget(self.run_button)

        self.metrics_form = QFormLayout()
        self.metric_labels = {
            "n_trades": QLabel("-"),
            "total_fees": QLabel("-"),
            "total_invested": QLabel("-"),
            "final_value": QLabel("-"),
            "strategy_profit": QLabel("-"),
            "strategy_return_pct": QLabel("-"),
            "strategy_annual_return_pct": QLabel("-"),
        }
        self.metrics_form.addRow("Anzahl Trades:", self.metric_labels["n_trades"])
        self.metrics_form.addRow("Total Fees:", self.metric_labels["total_fees"])
        self.metrics_form.addRow("Total Invested:", self.metric_labels["total_invested"])
        self.metrics_form.addRow("Final Value:", self.metric_labels["final_value"])
        self.metrics_form.addRow("Profit:", self.metric_labels["strategy_profit"])
        self.metrics_form.addRow("Return (%):", self.metric_labels["strategy_return_pct"])
        self.metrics_form.addRow("Annual Return (%):", self.metric_labels["strategy_annual_return_pct"])

        self.plot = pg.PlotWidget()
        self.plot.addLegend()
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.trades_table = QTableWidget()
        self.trades_table.setSortingEnabled(True)

        split = QSplitter(Qt.Orientation.Vertical)
        metrics_widget = QWidget()
        metrics_widget.setLayout(self.metrics_form)
        split.addWidget(metrics_widget)
        split.addWidget(self.plot)
        split.setSizes([200, 400])

        main_splitter = QSplitter(Qt.Orientation.Vertical)
        main_splitter.addWidget(split)
        main_splitter.addWidget(self.trades_table)
        main_splitter.setSizes([500, 300])
        layout.addWidget(main_splitter)

        self.run_button.clicked.connect(self.on_run_clicked)

    def set_language(self, lang: str) -> None:
        self._lang = lang
        self.run_button.setText("Benchmark ausfuehren" if lang == "de" else "Run benchmark")

    def on_run_clicked(self) -> None:
        ticker = self.ticker_input.text().strip().upper()
        if not ticker:
            QMessageBox.warning(self, "Fehler" if self._lang == "de" else "Error",
                                "Bitte einen Ticker eingeben." if self._lang == "de" else "Please enter a ticker.")
            return
        try:
            start = None
            end = None
            period_sel = self.range_combo.currentText()
            if period_sel != "Max":
                end = pd.Timestamp.today().date()
                if period_sel == "1Y":
                    start = end - pd.DateOffset(years=1)
                elif period_sel == "3Y":
                    start = end - pd.DateOffset(years=3)
                elif period_sel == "5Y":
                    start = end - pd.DateOffset(years=5)
                elif period_sel == "10Y":
                    start = end - pd.DateOffset(years=10)
                start = start.date()
            df = load_price_data(ticker, start=start, end=end)
            key = self._strategies[self.strategy_combo.currentIndex()][1]
            ctrls = self.param_controls[key]
            if key == "buy":
                result = run_buy_the_dip_backtest(
                    df=df,
                    lookback_days=ctrls["lookback"].value(),
                    drop_pct=ctrls["drop"].value(),
                    max_trades_per_day=ctrls["max_trades"].value(),
                    amount_per_trade=ctrls["amount"].value(),
                    fee_per_trade=ctrls["fee"].value(),
                )
            elif key == "sma":
                result = run_sma_crossover_backtest(
                    df=df,
                    short_window=ctrls["short"].value(),
                    long_window=ctrls["long"].value(),
                    initial_cash=ctrls["initial"].value(),
                    fee_per_trade=ctrls["fee"].value(),
                )
            elif key == "dca":
                result = run_dca_backtest(
                    df=df,
                    interval_days=ctrls["interval"].value(),
                    amount_per_trade=ctrls["amount"].value(),
                    fee_per_trade=ctrls["fee"].value(),
                )
            elif key == "ema":
                result = run_ema_crossover_backtest(
                    df=df,
                    short_span=ctrls["short"].value(),
                    long_span=ctrls["long"].value(),
                    initial_cash=ctrls["initial"].value(),
                    fee_per_trade=ctrls["fee"].value(),
                )
            elif key == "rsi":
                result = run_rsi_reversion_backtest(
                    df=df,
                    period=ctrls["period"].value(),
                    oversold=ctrls["oversold"].value(),
                    overbought=ctrls["overbought"].value(),
                    initial_cash=ctrls["initial"].value(),
                    fee_per_trade=ctrls["fee"].value(),
                )
            elif key == "boll":
                result = run_bollinger_reversion_backtest(
                    df=df,
                    window=ctrls["window"].value(),
                    num_std=ctrls["std"].value(),
                    initial_cash=ctrls["initial"].value(),
                    fee_per_trade=ctrls["fee"].value(),
                )
            elif key == "mom":
                result = run_momentum_12_1_backtest(
                    df=df,
                    initial_cash=ctrls["initial"].value(),
                    fee_per_trade=ctrls["fee"].value(),
                )
            elif key == "donchian":
                result = run_donchian_breakout_backtest(
                    df=df,
                    window=ctrls["window"].value(),
                    initial_cash=ctrls["initial"].value(),
                    fee_per_trade=ctrls["fee"].value(),
                )
            elif key == "macd":
                result = run_macd_backtest(
                    df=df,
                    fast=ctrls["fast"].value(),
                    slow=ctrls["slow"].value(),
                    signal=ctrls["signal"].value(),
                    initial_cash=ctrls["initial"].value(),
                    fee_per_trade=ctrls["fee"].value(),
                )
            else:
                result = run_stoch_osc_backtest(
                    df=df,
                    k_period=ctrls["k"].value(),
                    d_period=ctrls["d"].value(),
                    oversold=ctrls["oversold"].value(),
                    overbought=ctrls["overbought"].value(),
                    initial_cash=ctrls["initial"].value(),
                    fee_per_trade=ctrls["fee"].value(),
                )
            self._render_result(result)
        except Exception as exc:
            title = "Fehler" if self._lang == "de" else "Error"
            msg = f"Benchmark fehlgeschlagen: {exc}" if self._lang == "de" else f"Benchmark failed: {exc}"
            QMessageBox.critical(self, title, msg)

    def _render_result(self, result: BacktestResult) -> None:
        for key, lbl in self.metric_labels.items():
            val = result.metrics.get(key, "-")
            lbl.setText(f"{val:,.2f}" if isinstance(val, float) else str(val))
        try:
            total_inv = float(result.metrics.get("total_invested", 0.0))
            strat_final = float(result.metrics.get("final_value", 0.0))
            self.metric_labels["strategy_profit"].setText(f"{strat_final - total_inv:,.2f}")
        except Exception:
            self.metric_labels["strategy_profit"].setText("-")

        self.plot.clear()
        self.plot.addLegend()
        eq = result.equity_curve
        if not eq.empty:
            x = (pd.to_datetime(eq.index).astype("int64") // 10**9).to_numpy()
            if "strategy" in eq:
                self.plot.plot(x, eq["strategy"], pen=pg.mkPen("#f97316", width=2), name="Benchmark")
        trades = result.trades
        headers = ["Datum", "Side", "Preis", "Shares", "Amount", "Fee"]
        self.trades_table.clear()
        self.trades_table.setRowCount(len(trades))
        self.trades_table.setColumnCount(len(headers))
        self.trades_table.setHorizontalHeaderLabels(headers)
        for i, trade in enumerate(trades):
            self.trades_table.setItem(i, 0, QTableWidgetItem(str(trade.get("date"))))
            self.trades_table.setItem(i, 1, QTableWidgetItem(trade.get("side", "")))
            self.trades_table.setItem(i, 2, QTableWidgetItem(f"{trade.get('price', 0):,.2f}"))
            self.trades_table.setItem(i, 3, QTableWidgetItem(str(trade.get("shares", 0))))
            self.trades_table.setItem(i, 4, QTableWidgetItem(f"{trade.get('amount', 0):,.2f}"))
            self.trades_table.setItem(i, 5, QTableWidgetItem(f"{trade.get('fee', 0):,.2f}"))
        self.trades_table.resizeColumnsToContents()


class StrategiesTab(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        layout = QHBoxLayout(self)

        self.list_widget = QListWidget()
        self.stack = QStackedWidget()
        self.benchmark_widget = BenchmarkWidget(self)

        # register strategies
        self.buy_dip_widget = BuyTheDipWidget(self)
        self.sma_widget = SmaCrossoverWidget(self)
        self.dca_widget = DcaWidget(self)
        self.ema_widget = EmaCrossoverWidget(self)
        self.rsi_widget = RsiReversionWidget(self)
        self.boll_widget = BollingerReversionWidget(self)
        self.momentum_widget = MomentumWidget(self)
        self.donchian_widget = DonchianWidget(self)
        self.macd_widget = MacdWidget(self)
        self.stoch_widget = StochasticWidget(self)
        for w in [
            self.buy_dip_widget,
            self.sma_widget,
            self.dca_widget,
            self.ema_widget,
            self.rsi_widget,
            self.boll_widget,
            self.momentum_widget,
            self.donchian_widget,
            self.macd_widget,
            self.stoch_widget,
        ]:
            self.stack.addWidget(w)
        for name in [
            "Buy the Dip",
            "SMA Crossover",
            "DCA",
            "EMA Crossover",
            "RSI Reversion",
            "Bollinger Reversion",
            "Momentum 12-1",
            "Donchian Breakout",
            "MACD",
            "Stochastic Oscillator",
        ]:
            self.list_widget.addItem(QListWidgetItem(name))

        right_split = QSplitter(Qt.Orientation.Horizontal)
        right_split.addWidget(self.stack)
        right_split.addWidget(self.benchmark_widget)
        right_split.setSizes([900, 600])

        layout.addWidget(self.list_widget)
        layout.addWidget(right_split, 1)

        self.list_widget.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.list_widget.setCurrentRow(0)

    def set_language(self, lang: str) -> None:
        names_de = [
            "Buy the Dip",
            "SMA Crossover",
            "DCA",
            "EMA Crossover",
            "RSI Reversion",
            "Bollinger Reversion",
            "Momentum 12-1",
            "Donchian Breakout",
            "MACD",
            "Stochastic Oscillator",
        ]
        names_en = names_de
        names = names_de if lang == "de" else names_en
        for i, name in enumerate(names):
            if self.list_widget.item(i):
                self.list_widget.item(i).setText(name)
        self.buy_dip_widget.set_language(lang)
        self.sma_widget.set_language(lang)
        self.dca_widget.set_language(lang)
        self.ema_widget.set_language(lang)
        self.rsi_widget.set_language(lang)
        self.boll_widget.set_language(lang)
        self.momentum_widget.set_language(lang)
        self.donchian_widget.set_language(lang)
        self.macd_widget.set_language(lang)
        self.stoch_widget.set_language(lang)
        self.benchmark_widget.set_language(lang)
