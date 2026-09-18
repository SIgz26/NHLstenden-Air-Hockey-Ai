from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt


class OrionMainMenu(QWidget):
    def __init__(self, on_select_module_callback):
        super().__init__()
        self.on_select_module = on_select_module_callback
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 30, 40, 40)
        layout.setSpacing(25)

        # Top Bar met Titel & Tandwiel Instellingenknop
        top_bar = QHBoxLayout()
        
        header_text_layout = QVBoxLayout()
        title_label = QLabel("ORION AI VISION ENGINE BETA 1.0.3")
        title_label.setObjectName("TitleLabel")
        subtitle_label = QLabel("// SELECT MODULE TO LAUNCH")
        subtitle_label.setObjectName("SubTitleLabel")
        header_text_layout.addWidget(title_label)
        header_text_layout.addWidget(subtitle_label)

        # Settings Tandwiel Knop (Rechtsboven)
        btn_settings = QPushButton("⚙")
        btn_settings.setFixedSize(52, 52)
        btn_settings.setFont(QFont("Arial", 20))
        btn_settings.setToolTip("Global Settings")
        btn_settings.setStyleSheet(
            "QPushButton {"
            "  background-color: #161212;"
            "  border: 1px solid #503422;"
            "  border-radius: 16px;"
            "  color: #d9b38c;"
            "  padding: 0;"
            "}"
            "QPushButton:hover {"
            "  background-color: #1e1a1a;"
            "  border: 1px solid #8c5a34;"
            "}"
            "QPushButton:pressed {"
            "  background-color: #0f0d0d;"
            "}"
            ""
        )
        btn_settings.clicked.connect(lambda: self.on_select_module("settings"))

        top_bar.addLayout(header_text_layout)
        top_bar.addStretch()
        top_bar.addWidget(btn_settings)
        layout.addLayout(top_bar)

        # Grid van Module Kaarten (3 stuks)
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(20)

        # --- Optie 1: Video File Analyzer ---
        card_analyzer = QFrame()
        card_analyzer.setObjectName("CardFrame")
        ca_layout = QVBoxLayout(card_analyzer)
        ca_layout.setContentsMargins(20, 20, 20, 20)

        ca_title = QLabel("VIDEO FILE ANALYZER")
        ca_title.setFont(QFont("Arial", 14, QFont.Bold))
        ca_title.setStyleSheet("color: rgb(184, 98, 48);")

        ca_body = QLabel("Analyseer opgenomen video's van de airhockeytafel.\nInclusief HSV-tuning, Kalman-tracking en tijdbalk-bediening.")
        ca_body.setWordWrap(True)

        btn_launch_analyzer = QPushButton("OPEN ANALYZER")
        btn_launch_analyzer.clicked.connect(lambda: self.on_select_module("analyzer"))

        ca_layout.addWidget(ca_title)
        ca_layout.addWidget(ca_body)
        ca_layout.addStretch()
        ca_layout.addWidget(btn_launch_analyzer)

        # --- Optie 2: Live Camera Stream ---
        card_live = QFrame()
        card_live.setObjectName("CardFrame")
        cl_layout = QVBoxLayout(card_live)
        cl_layout.setContentsMargins(20, 20, 20, 20)

        cl_title = QLabel("LIVE CAMERA STREAM")
        cl_title.setFont(QFont("Arial", 14, QFont.Bold))
        cl_title.setStyleSheet("color: rgb(184, 98, 48);")

        cl_body = QLabel("Koppel direct een USB-camera voor realtime puck-detectie en autonome reacties.")
        cl_body.setWordWrap(True)

        # === 1. ZET OP TRUE EN KOPPEL HET SIGNAAL ===
        btn_launch_live = QPushButton("START LIVE FEED")
        btn_launch_live.setEnabled(True)  # Hiermee wordt de knop klikbaar!
        btn_launch_live.clicked.connect(lambda: self.on_select_module("live"))

        cl_layout.addWidget(cl_title)
        cl_layout.addWidget(cl_body)
        cl_layout.addStretch()
        cl_layout.addWidget(btn_launch_live)

        # --- Optie 3: AI Model Trainer ---
        card_ai = QFrame()
        card_ai.setObjectName("CardFrame")
        ai_layout = QVBoxLayout(card_ai)
        ai_layout.setContentsMargins(20, 20, 20, 20)

        ai_title = QLabel("AI TRAINER // MODEL BUILD")
        ai_title.setFont(QFont("Arial", 14, QFont.Bold))
        ai_title.setStyleSheet("color: rgb(184, 98, 48);")

        ai_body = QLabel("Train en configureer neurale netwerken voor geavanceerde objectherkenning.")
        ai_body.setWordWrap(True)

        btn_launch_ai = QPushButton("LAUNCH BUILDER")
        btn_launch_ai.setEnabled(False)

        ai_layout.addWidget(ai_title)
        ai_layout.addWidget(ai_body)
        ai_layout.addStretch()
        ai_layout.addWidget(btn_launch_ai)

        # Toevoegen van alle 3 de kaarten
        cards_layout.addWidget(card_analyzer, 1)
        cards_layout.addWidget(card_live, 1)
        cards_layout.addWidget(card_ai, 1)
        

        layout.addLayout(cards_layout)