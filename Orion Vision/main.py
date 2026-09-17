#
#versie: 1.0.1
#door: Timo Zylstra
#datum: 17-09-2026
#nieuw:gige_worker.py, Diagnose_Tool.py, gige_diagnose.py en gige_vmbpy.py zijn aan gemaakt om de GigE camera werkende te krijgen.
#
import os
import sys
import ctypes
from PyQt5.QtWidgets import QApplication

from ui.main_window import OrionMainWindow
import os
from PyQt5.QtWidgets import QMainWindow, QStackedWidget
from PyQt5.QtGui import QIcon

from intro.intro_widget import OrionIntroWidget
from ui.main_menu import OrionMainMenu
from ui.dashboard import OrionDashboard

ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("orion.vision.app.1.0")





def load_stylesheet(app, qss_path):
    if os.path.exists(qss_path):
        with open(qss_path, "r") as f:
            app.setStyleSheet(f.read())


def main():
    app = QApplication(sys.argv)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    qss_file = os.path.join(base_dir, "assets", "styles", "main.qss")
    load_stylesheet(app, qss_file)

    window = OrionMainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()