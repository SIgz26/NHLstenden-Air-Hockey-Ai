import os
import time

from PyQt5.QtCore import QEvent, QObject, QUrl
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtWidgets import QApplication, QPushButton


class HoverSoundManager(QObject):
    """Plays a short hover blip whenever a button is entered."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.player = QMediaPlayer(self)
        self.player.setVolume(30)
        self.hover_sound_path = self._resolve_hover_sound_path()
        self.enabled = True
        self._last_hover_time = 0.0
        self._media_loaded = False

        if os.path.exists(self.hover_sound_path):
            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(self.hover_sound_path)))
            self._media_loaded = True

    def _resolve_hover_sound_path(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        audio_dir = os.path.abspath(os.path.join(base_dir, "assets", "audio"))

        candidates = [
            os.path.join(audio_dir, "hover.wav"),
            os.path.join(audio_dir, "ui_menuMove.wav"),
            os.path.join(audio_dir, "battlefront_hover.wav"),
        ]

        for path in candidates:
            if os.path.exists(path):
                return path

        return os.path.join(audio_dir, "hover.wav")

    def register_widget(self, widget):
        if widget is None:
            return

        if isinstance(widget, QPushButton):
            self.register_button(widget)
            return

        for child in widget.findChildren(QPushButton):
            self.register_button(child)

    def register_button(self, button):
        if not isinstance(button, QPushButton):
            return

        if button.property("hover_sound_registered"):
            return

        button.installEventFilter(self)
        button.setMouseTracking(True)
        button.setProperty("hover_sound_registered", True)

    def eventFilter(self, obj, event):
        if isinstance(obj, QPushButton) and event.type() == QEvent.Enter:
            self.play_hover()
        return super().eventFilter(obj, event)

    def set_enabled(self, enabled: bool):
        self.enabled = bool(enabled)
        if not self.enabled:
            try:
                self.player.stop()
            except Exception:
                pass

    def play_hover(self):
        if not self.enabled:
            return

        if not os.path.exists(self.hover_sound_path):
            return

        now = time.monotonic()
        if now - self._last_hover_time < 0.18:
            return
        self._last_hover_time = now

        try:
            if not self._media_loaded:
                self.player.setMedia(QMediaContent(QUrl.fromLocalFile(self.hover_sound_path)))
                self._media_loaded = True
            self.player.stop()
            self.player.play()
        except Exception:
            pass

    @staticmethod
    def install_for_app(app):
        manager = HoverSoundManager(app)
        app.hover_sound_manager = manager
        return manager
