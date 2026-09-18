import os
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, QTimer, QRectF, QUrl
from PyQt5.QtGui import QPainter, QColor, QFont, QPen, QBrush, QImage, QPixmap
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent

class OrionIntroWidget(QWidget):
    def __init__(self, on_finished_callback):
        super().__init__()
        self.on_finished_callback = on_finished_callback
        self.elapsed = 0.0

        # Kleuren
        self.BG_COLOR = QColor(24, 22, 22)
        self.ARCH_OUTER = QColor(80, 52, 34)
        self.ARCH_INNER = QColor(168, 108, 56)
        self.SUN_COLOR = QColor(184, 98, 48)
        self.TEXT_BG = QColor(12, 10, 10)
        self.TEXT_COLOR = QColor(212, 195, 153)

        self.scanline_overlay = None
        self.audio_started = False

        # Audio
        self.player = QMediaPlayer()

        # Animation Loop (60 FPS)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(16)
    def showEvent(self, event):
        super().showEvent(event)
        if not self.audio_started:
            self.audio_started = True
            self.play_audio()

    def reset(self):
        """Reset de intro-animatie zodat deze opnieuw afgespeeld kan worden."""
        self.elapsed = 0.0
        self.timer.start(16)
        self.audio_started = False
        self.play_audio()
        self.update()
        
    def play_audio(self):
        # Haalt de absolute root directory op van het project
        intro_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(intro_dir, ".."))
        
        # Zorg voor een absoluut pad naar de audioclip
        audio_path = os.path.abspath(os.path.join(root_dir, "assets", "audio", "Audio01_intro.wav"))

        if os.path.exists(audio_path):
            url = QUrl.fromLocalFile(audio_path)
            content = QMediaContent(url)
            self.player.setMedia(content)
            self.player.setVolume(80)
            self.player.play()
        else:
            print(f"[ORION AUDIO ERROR] Bestand niet gevonden op pad: {audio_path}")

    def update_animation(self):
        self.elapsed += 0.016
        if self.elapsed >= 6.0:
            self.finish_intro()
            return
        self.update()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Space, Qt.Key_Escape):
            self.finish_intro()

    def mousePressEvent(self, event):
        self.finish_intro()

    def finish_intro(self):
        self.timer.stop()
        self.player.stop()
        if self.on_finished_callback:
            self.on_finished_callback()

    def resizeEvent(self, event):
        w, h = self.width(), self.height()
        img = QImage(w, h, QImage.Format_ARGB32_Premultiplied)
        img.fill(Qt.transparent)

        p = QPainter(img)
        p.setPen(QPen(QColor(0, 0, 0, 80), 1))
        for y in range(0, h, 4):
            p.drawLine(0, y, w, y)
        p.end()

        self.scanline_overlay = QPixmap.fromImage(img)
        super().resizeEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        w, h = self.width(), self.height()
        cx, cy = w // 2, h // 2

        # 1. Achtergrond
        painter.fillRect(self.rect(), self.BG_COLOR)
        painter.setRenderHint(QPainter.Antialiasing)

        # 2. Zon
        if self.elapsed > 0:
            sun_alpha = int(min(1.0, self.elapsed / 1.0) * 255)
            sun_color = QColor(self.SUN_COLOR)
            sun_color.setAlpha(sun_alpha)
            painter.setBrush(QBrush(sun_color))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(cx - 55, cy - 30 - 55, 110, 110)

        # 3. Bogen
        if self.elapsed > 1.0:
            arch_progress = min(1.0, (self.elapsed - 1.0) / 3.0)
            span_angle = int(arch_progress * 180 * 16)
            painter.setBrush(Qt.NoBrush)

            pen_outer = QPen(self.ARCH_OUTER, 75)
            pen_outer.setCapStyle(Qt.FlatCap)
            painter.setPen(pen_outer)
            rect_outer = QRectF(cx - 200 + 20, cy - 220 + 20, 360, 360)
            painter.drawArc(rect_outer, 0, span_angle)

            pen_inner = QPen(self.ARCH_INNER, 45)
            pen_inner.setCapStyle(Qt.FlatCap)
            painter.setPen(pen_inner)
            rect_inner = QRectF(cx - 200 + 70, cy - 220 + 70, 260, 260)
            painter.drawArc(rect_inner, 0, span_angle)

        # 4. Tekst ORION
        if self.elapsed > 3.0:
            text_alpha = int(min(1.0, (self.elapsed - 3.0) / 1.0) * 255)
            bg_color = QColor(self.TEXT_BG)
            bg_color.setAlpha(text_alpha)
            box_rect = QRectF(cx - 150, cy + 30 - 40, 300, 80)
            painter.fillRect(box_rect, bg_color)

            text_color = QColor(self.TEXT_COLOR)
            text_color.setAlpha(text_alpha)
            painter.setPen(text_color)
            painter.setFont(QFont("Impact", 50))
            painter.drawText(box_rect, Qt.AlignCenter, "ORION")

        # 5. Onderschrift
        if self.elapsed > 4.0:
            sub_alpha = int(min(1.0, (self.elapsed - 4.0) / 1.0) * 255)
            sub_color = QColor(self.TEXT_COLOR)
            sub_color.setAlpha(sub_alpha)
            painter.setPen(sub_color)
            painter.setFont(QFont("Arial", 16, QFont.Bold))
            sub_rect = QRectF(cx - 200, cy + 150 - 20, 400, 40)
            painter.drawText(sub_rect, Qt.AlignCenter, "AI VISION")

        # 6. CRT Overlay
        painter.setRenderHint(QPainter.Antialiasing, True)
        if self.scanline_overlay:
            painter.drawPixmap(0, 0, self.scanline_overlay)