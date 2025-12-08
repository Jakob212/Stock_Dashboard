from typing import Callable, Optional

from PyQt6.QtWidgets import QComboBox, QFormLayout, QLabel, QWidget


class SettingsTab(QWidget):
    def __init__(self, on_language_changed: Callable[[str], None], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._on_language_changed = on_language_changed

        layout = QFormLayout(self)
        self.language_combo = QComboBox()
        self.language_combo.addItems(["Deutsch", "English"])
        layout.addRow(QLabel("Sprache / Language:"), self.language_combo)

        self.language_combo.currentIndexChanged.connect(self._handle_language_change)

    def _handle_language_change(self, idx: int) -> None:
        lang = "de" if idx == 0 else "en"
        self._on_language_changed(lang)
