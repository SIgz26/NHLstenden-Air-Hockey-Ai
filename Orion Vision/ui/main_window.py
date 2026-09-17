import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from PyQt5.QtWidgets import QMainWindow, QStackedWidget
from PyQt5.QtGui import QIcon

from intro.intro_widget import OrionIntroWidget
from ui.main_menu import OrionMainMenu
from ui.dashboard import OrionDashboard
from ui.live_dashboard import OrionLiveDashboard  # <-- Importeer het live dashboard


class OrionMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ORION - AI Vision System")
        self.resize(1150, 720)

        icon_path = os.path.join(BASE_DIR, "assets", "icons", "logo.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # Screens toevoegen
        self.intro_widget = OrionIntroWidget(on_finished_callback=self.show_main_menu)
        self.stack.addWidget(self.intro_widget)

        self.main_menu_widget = OrionMainMenu(on_select_module_callback=self.launch_module)
        self.stack.addWidget(self.main_menu_widget)

        self.dashboard_widget = OrionDashboard(on_back_callback=self.show_main_menu)
        self.stack.addWidget(self.dashboard_widget)

        self.live_dashboard_widget = OrionLiveDashboard(on_back_callback=self.show_main_menu)
        self.stack.addWidget(self.live_dashboard_widget)

        self.stack.setCurrentWidget(self.intro_widget)

    def show_main_menu(self):
        self.stack.setCurrentWidget(self.main_menu_widget)

    def launch_module(self, module_name):
        if module_name == "analyzer":
            self.stack.setCurrentWidget(self.dashboard_widget)
        elif module_name == "live":
            self.stack.setCurrentWidget(self.live_dashboard_widget)
        elif module_name == "settings":
            print("Settings geopend")