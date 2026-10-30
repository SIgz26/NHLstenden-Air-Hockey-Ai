#
#versie: BETA 1.2.2
#door: Timo Zylstra
#datum: 09-10-2026
#
import PyQt5    
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

# Only run taskbar icon setup on Windows
if sys.platform == "win32":
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("orion.vision.app.1.0")


def _clear_opencv_qt_plugin_override():
    plugin_path = os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH")
    if not plugin_path:
        return

    normalized_path = os.path.normcase(os.path.normpath(plugin_path))
    opencv_plugin_suffix = os.path.normcase(os.path.join("cv2", "qt", "plugins"))
    if normalized_path.endswith(opencv_plugin_suffix):
        os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")


def load_stylesheet(app, qss_path):
    if os.path.exists(qss_path):
        with open(qss_path, "r", encoding="utf-8") as f:
            app.setStyleSheet(f.read())


def main():
    _clear_opencv_qt_plugin_override()
    app = QApplication(sys.argv)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    qss_file = os.path.join(base_dir, "assets", "styles", "main.qss")
    load_stylesheet(app, qss_file)

    window = OrionMainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()