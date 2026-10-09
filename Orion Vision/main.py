#
#versie: BETA 1.2.1
#door: Timo Zylstra
#datum: 09-10-2026
#

import os
import sys
import ctypes

try:
    import torch
    TORCH_IMPORT_ERROR = None
except Exception as exc:
    torch = None
    TORCH_IMPORT_ERROR = exc
    print(f"PyTorch unavailable; SAC inference disabled: {exc}", file=sys.stderr)

from ai.sac_controller import check_sac_runtime, preload_sac_runtime

if not preload_sac_runtime(
    torch,
    TORCH_IMPORT_ERROR,
    torch_import_attempted=True,
):
    try:
        check_sac_runtime()
    except RuntimeError as exc:
        print(f"SAC inference unavailable: {exc}", file=sys.stderr)

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