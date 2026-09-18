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
from ui.sound_manager import HoverSoundManager


class OrionMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ORION - AI Vision System")
        self.resize(1150, 720)

        self.hover_sound_manager = HoverSoundManager(self)

        icon_path = os.path.join(BASE_DIR, "assets", "icons", "logo.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.main_menu_widget = None
        self.dashboard_widget = None
        self.live_dashboard_widget = None

        self.intro_widget = OrionIntroWidget(on_finished_callback=self.show_main_menu)
        self.stack.addWidget(self.intro_widget)
        self.stack.setCurrentWidget(self.intro_widget)

    def _register_hover_sounds(self):
        widgets = [self.intro_widget]
        if self.main_menu_widget is not None:
            widgets.append(self.main_menu_widget)
        if self.dashboard_widget is not None:
            widgets.append(self.dashboard_widget)
        if self.live_dashboard_widget is not None:
            widgets.append(self.live_dashboard_widget)

        for widget in widgets:
            self.hover_sound_manager.register_widget(widget)

    def _ensure_main_menu(self):
        if self.main_menu_widget is None:
            self.main_menu_widget = OrionMainMenu(on_select_module_callback=self.launch_module)
            self.stack.addWidget(self.main_menu_widget)
            self.hover_sound_manager.register_widget(self.main_menu_widget)

    def _ensure_dashboard(self):
        if self.dashboard_widget is None:
            self.dashboard_widget = OrionDashboard(on_back_callback=self.show_main_menu)
            self.stack.addWidget(self.dashboard_widget)
            self.hover_sound_manager.register_widget(self.dashboard_widget)

    def _ensure_live_dashboard(self):
        if self.live_dashboard_widget is None:
            self.live_dashboard_widget = OrionLiveDashboard(on_back_callback=self.show_main_menu)
            self.stack.addWidget(self.live_dashboard_widget)
            self.hover_sound_manager.register_widget(self.live_dashboard_widget)

    def show_main_menu(self):
        self._ensure_main_menu()
        self.stack.setCurrentWidget(self.main_menu_widget)

    def launch_module(self, module_name):
        if module_name == "analyzer":
            self._ensure_dashboard()
            self.stack.setCurrentWidget(self.dashboard_widget)
        elif module_name == "live":
            self._ensure_live_dashboard()
            self.stack.setCurrentWidget(self.live_dashboard_widget)
        elif module_name == "settings":
            print("Settings geopend")