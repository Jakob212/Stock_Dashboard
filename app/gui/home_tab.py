from typing import Optional

from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget


class HomeTab(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self.label = QLabel("Home - Willkommen!")
        layout = QVBoxLayout(self)
        layout.addWidget(self.label)

    def set_language(self, lang: str) -> None:
        if lang == "de":
            self.label.setText("Home - Willkommen!")
        else:
            self.label.setText("Home - Welcome!")
