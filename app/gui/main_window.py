from PyQt6.QtWidgets import QMainWindow, QTabWidget

from .home_tab import HomeTab
from .settings_tab import SettingsTab
from .stocks_tab import StocksTab
from .strategies_tab import StrategiesTab


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("Stock Analysis & Backtesting")
        self.resize(1200, 800)

        self._init_ui()

    def _init_ui(self) -> None:
        self.tab_widget = QTabWidget()

        self.home_tab = HomeTab()
        self.stocks_tab = StocksTab()
        self.strategies_tab = StrategiesTab()
        self.settings_tab = SettingsTab(self._on_language_changed)

        self.tab_widget.addTab(self.home_tab, "Home")
        self.tab_widget.addTab(self.stocks_tab, "Stocks")
        self.tab_widget.addTab(self.strategies_tab, "Strategies")
        self.tab_widget.addTab(self.settings_tab, "Settings")

        self.setCentralWidget(self.tab_widget)

    def _on_language_changed(self, lang: str) -> None:
        if lang == "de":
            self.setWindowTitle("Aktienanalyse & Backtesting")
            self.tab_widget.setTabText(0, "Home")
            self.tab_widget.setTabText(1, "Aktien")
            self.tab_widget.setTabText(2, "Strategien")
            self.tab_widget.setTabText(3, "Einstellungen")
            if hasattr(self.stocks_tab, "set_language"):
                self.stocks_tab.set_language("de")
            if hasattr(self.strategies_tab, "set_language"):
                self.strategies_tab.set_language("de")
        else:
            self.setWindowTitle("Stock Analysis & Backtesting")
            self.tab_widget.setTabText(0, "Home")
            self.tab_widget.setTabText(1, "Stocks")
            self.tab_widget.setTabText(2, "Strategies")
            self.tab_widget.setTabText(3, "Settings")
            if hasattr(self.stocks_tab, "set_language"):
                self.stocks_tab.set_language("en")
            if hasattr(self.strategies_tab, "set_language"):
                self.strategies_tab.set_language("en")
