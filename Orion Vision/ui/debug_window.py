from datetime import datetime

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QHBoxLayout, QPlainTextEdit, QPushButton, QVBoxLayout


class CameraDebugDialog(QDialog):
    """Klein debug-venster met log-output voor GigE/USB camera's."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("ORION CAMERA DEBUG")
        self.resize(700, 500)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        btn_row = QHBoxLayout()
        self.btn_clear = QPushButton("CLEAR")
        self.btn_copy = QPushButton("COPY")
        self.btn_close = QPushButton("CLOSE")

        self.btn_clear.clicked.connect(self.clear_logs)
        self.btn_copy.clicked.connect(self.copy_logs)
        self.btn_close.clicked.connect(self.close)

        btn_row.addWidget(self.btn_clear)
        btn_row.addWidget(self.btn_copy)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_close)
        layout.addLayout(btn_row)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("Camera debug output verschijnt hier...")
        layout.addWidget(self.log_view)

        self.append_log("DEBUG WINDOW OPENED")
        self.append_log("Use this window to inspect camera startup, pixel formats and live status.")

    def append_log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_view.appendPlainText(f"[{timestamp}] {message}")
        self.log_view.ensureCursorVisible()

    def clear_logs(self) -> None:
        self.log_view.clear()

    def copy_logs(self) -> None:
        text = self.log_view.toPlainText()
        if text:
            clipboard = self.parent().windowHandle().clipboard() if self.parent() else None
            if clipboard is not None:
                clipboard.setText(text)
            else:
                self.log_view.selectAll()
                self.log_view.copy()
