import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

opencv_plugin_path = os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH")
if opencv_plugin_path:
    normalized_path = os.path.normcase(os.path.normpath(opencv_plugin_path))
    opencv_plugin_suffix = os.path.normcase(os.path.join("cv2", "qt", "plugins"))
    if normalized_path.endswith(opencv_plugin_suffix):
        os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
