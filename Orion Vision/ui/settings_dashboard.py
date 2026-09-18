from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, QCheckBox


class OrionSettingsDashboard(QWidget):
    def __init__(self, on_back_callback=None, hover_sound_manager=None):
        super().__init__()
        self.on_back = on_back_callback
        self.hover_sound_manager = hover_sound_manager
        self.init_ui()

    def init_ui(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #120d0b;
                color: #f0d8ba;
            }
            #HeaderFrame {
                background-color: #17110f;
                border: 1px solid #503422;
                border-radius: 10px;
            }
            #CardFrame {
                background-color: #17110f;
                border: 1px solid #503422;
                border-radius: 12px;
            }
            QLabel {
                color: #e9c89d;
            }
            QCheckBox {
                color: #f4d8b0;
                spacing: 10px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                background: #1c1412;
                border: 1px solid #b86230;
                border-radius: 4px;
            }
            QCheckBox::indicator:checked {
                background: #b86230;
                border: 1px solid #d99b62;
            }
            QPushButton {
                background-color: #1a1412;
                border: 1px solid #503422;
                border-radius: 8px;
                color: #f5d9b8;
                padding: 6px 10px;
            }
            QPushButton:hover {
                background-color: #251d19;
                border: 1px solid #9d6639;
            }
            QPushButton:pressed {
                background-color: #110d0b;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 20, 30, 30)
        layout.setSpacing(20)

        header = QFrame()
        header.setObjectName("HeaderFrame")
        hbox = QHBoxLayout(header)
        hbox.setContentsMargins(20, 12, 20, 12)

        if self.on_back:
            btn_back = QPushButton("◀ MENU")
            btn_back.clicked.connect(self.go_back)
            hbox.addWidget(btn_back)

        title = QLabel("ORION")
        title.setObjectName("TitleLabel")
        subtitle = QLabel("AI VISION // SETTINGS")
        subtitle.setObjectName("SubTitleLabel")

        hbox.addWidget(title)
        hbox.addWidget(subtitle)
        hbox.addStretch()
        layout.addWidget(header)

        panel = QFrame()
        panel.setObjectName("CardFrame")
        pnl_layout = QVBoxLayout(panel)
        pnl_layout.setContentsMargins(20, 20, 20, 20)
        pnl_layout.setSpacing(20)

        self.hover_sound_toggle = QCheckBox("Button hover sound aan / uit")
        self.hover_sound_toggle.setChecked(True)
        self.hover_sound_toggle.toggled.connect(self._toggle_hover_sound)

        pnl_layout.addWidget(QLabel("INTERFACE"))
        pnl_layout.addWidget(self.hover_sound_toggle)
        pnl_layout.addStretch()

        layout.addWidget(panel)

    def _toggle_hover_sound(self, checked: bool):
        if self.hover_sound_manager is not None:
            self.hover_sound_manager.set_enabled(not checked)

    def go_back(self):
        if self.on_back:
            self.on_back()
