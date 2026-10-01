"""Lightweight 2D air-hockey digital twin rendered with Qt's QPainter."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt5.QtGui import QColor, QPaintEvent, QPainter, QPen, QPolygonF
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QSizePolicy, QWidget

if TYPE_CHECKING:
    from core.sim_bridge import SimBridge


class DigitalTwinWidget(QWidget):
    """Paint a schematic air-hockey field from a SimBridge observation.

    The widget polls the bridge on a GUI-thread timer (30 Hz by default),
    caches a copy of the latest vector, and schedules a lightweight repaint.
    The camera worker is never called from ``paintEvent``.

    Observation order is ``[puck_x, puck_y, puck_vx, puck_vy, opp_x, opp_y,
    robot_x, robot_y]`` with normalized values in the range ``[-1, 1]``.
    """

    FIELD_ASPECT_RATIO = 1.8
    CENTER_CIRCLE_RADIUS_RATIO = 0.12
    ACTION_PIXELS_PER_MPS = 100.0
    MIN_ACTION_VECTOR_PIXELS = 40.0
    MIN_VISIBLE_ACTION_MPS = 0.01

    def __init__(
        self,
        sim_bridge: SimBridge | None = None,
        refresh_hz: float = 30.0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._sim_bridge: SimBridge | None = None
        self._external_observation = False
        self._observation = np.zeros(8, dtype=np.float32)
        self._sac_action: tuple[float, float] | None = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh_observation)
        self.setMinimumSize(180, 150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.set_refresh_rate(refresh_hz)
        self.update_bridge(sim_bridge)
        self._timer.start()

    def set_refresh_rate(self, refresh_hz: float) -> None:
        """Set the display polling rate in hertz; default is 30 Hz."""
        if refresh_hz <= 0.0:
            raise ValueError("refresh_hz must be greater than zero")
        self._timer.setInterval(max(1, round(1000.0 / refresh_hz)))

    def update_bridge(self, sim_bridge: SimBridge | None) -> None:
        """Connect to a bridge, or clear the twin when passed ``None``."""
        self._sim_bridge = sim_bridge
        self._external_observation = False
        if sim_bridge is None:
            self._observation = np.zeros(8, dtype=np.float32)
        self._refresh_observation()

    def update_observation_vector(self, observation: np.ndarray) -> None:
        """Set an already-normalized observation, for simulated previews."""
        vector = np.asarray(observation, dtype=np.float32).reshape(-1)
        if vector.shape == (8,) and np.isfinite(vector).all():
            self._external_observation = True
            self._observation = vector.copy()
            self.update()

    def update_sac_action(self, vx: float, vy: float) -> None:
        """Update target velocity in m/s and schedule the target overlay paint."""
        if not np.isfinite((vx, vy)).all():
            return
        self._sac_action = (float(vx), float(vy))
        self.update()

    def get_action_vector_endpoints(
        self, field: QRectF | None = None
    ) -> tuple[QPointF, QPointF, QPointF] | None:
        """Return screen-space endpoints for Vx, Vy, and resultant vectors.

        Each vector starts at the current robot center. Component arrows remain
        axis-aligned. Non-trivial vectors are given a minimum display length,
        preserving their direction while making small actions legible.
        """
        if self._sac_action is None:
            return None
        field = field or self._field_rect()
        if field.width() <= 0.0 or field.height() <= 0.0:
            return None

        robot_x = float(self._observation[6])
        robot_y = float(self._observation[7])
        robot_center = self._field_point(field, robot_x, robot_y)
        vx, vy = self._sac_action

        vx_length = self._visible_vector_length(vx)
        vy_length = self._visible_vector_length(vy)
        result_speed = float(np.hypot(vx, vy))
        result_length = self._visible_vector_length(result_speed)

        vx_endpoint = self._clip_to_field(
            field,
            QPointF(robot_center.x() + np.sign(vx) * vx_length, robot_center.y()),
        )
        vy_endpoint = self._clip_to_field(
            field,
            QPointF(robot_center.x(), robot_center.y() - np.sign(vy) * vy_length),
        )

        result_direction_x = vx / result_speed if result_speed > 0.0 else 0.0
        result_direction_y = vy / result_speed if result_speed > 0.0 else 0.0
        resultant_endpoint = self._clip_to_field(
            field,
            QPointF(
                robot_center.x() + result_direction_x * result_length,
                robot_center.y() - result_direction_y * result_length,
            ),
        )
        return vx_endpoint, vy_endpoint, resultant_endpoint

    @classmethod
    def _visible_vector_length(cls, speed: float) -> float:
        """Scale an action into pixels with a minimum visible nonzero length."""
        magnitude = abs(float(speed))
        if magnitude < cls.MIN_VISIBLE_ACTION_MPS:
            return 0.0
        return max(magnitude * cls.ACTION_PIXELS_PER_MPS, cls.MIN_ACTION_VECTOR_PIXELS)

    @staticmethod
    def _clip_to_field(field: QRectF, point: QPointF) -> QPointF:
        return QPointF(
            float(np.clip(point.x(), field.left() + 2.0, field.right() - 2.0)),
            float(np.clip(point.y(), field.top() + 2.0, field.bottom() - 2.0)),
        )

    @staticmethod
    def _robot_marker_radius(field: QRectF) -> float:
        """Return the exact radius used by the painted robot mallet."""
        return max(9.0, min(field.width(), field.height()) * 0.075)

    def get_action_endpoint_clearances(
        self, field: QRectF | None = None
    ) -> tuple[float, float, float] | None:
        """Return vector endpoint distances from robot center in screen pixels."""
        field = field or self._field_rect()
        endpoints = self.get_action_vector_endpoints(field)
        if endpoints is None:
            return None
        robot_center = self._field_point(
            field, float(self._observation[6]), float(self._observation[7])
        )
        return tuple(
            float(np.hypot(point.x() - robot_center.x(), point.y() - robot_center.y()))
            for point in endpoints
        )

    def get_robot_action_target(self) -> tuple[float, float] | None:
        """Return the normalized resultant target using the rendered vector scale."""
        if self._sac_action is None:
            return None
        field = self._field_rect()
        endpoints = self.get_action_vector_endpoints(field)
        if endpoints is None:
            return None
        resultant_endpoint = endpoints[2]
        return (
            float(np.clip(2.0 * (resultant_endpoint.x() - field.left()) / field.width() - 1.0, -1.0, 1.0)),
            float(np.clip(1.0 - 2.0 * (resultant_endpoint.y() - field.top()) / field.height(), -1.0, 1.0)),
        )

    def _refresh_observation(self) -> None:
        if self._sim_bridge is None:
            if not self._external_observation:
                self._observation = np.zeros(8, dtype=np.float32)
        else:
            observation = np.asarray(
                self._sim_bridge.get_observation_vector(), dtype=np.float32
            ).reshape(-1)
            if observation.size == 8 and np.all(np.isfinite(observation)):
                self._observation = observation.copy()
        self.update()

    @staticmethod
    def _field_point(field: QRectF, x: float, y: float) -> QPointF:
        x = float(np.clip(x, -1.0, 1.0))
        y = float(np.clip(y, -1.0, 1.0))
        return QPointF(
            field.left() + (x + 1.0) * field.width() / 2.0,
            field.top() + (1.0 - y) * field.height() / 2.0,
        )

    def _field_rect(self) -> QRectF:
        """Return a centered landscape field rectangle without stretching."""
        available = QRectF(self.rect()).adjusted(12.0, 12.0, -12.0, -12.0)
        if available.width() <= 0.0 or available.height() <= 0.0:
            return QRectF()

        aspect = self.FIELD_ASPECT_RATIO
        width = min(available.width(), available.height() * aspect)
        height = width / aspect
        return QRectF(
            available.center().x() - width / 2.0,
            available.center().y() - height / 2.0,
            width,
            height,
        )

    def paintEvent(self, event: QPaintEvent) -> None:
        """Draw the field, puck velocity, and H-style robot using primitives."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#101715"))

        field = self._field_rect()
        if field.width() <= 0.0 or field.height() <= 0.0:
            painter.end()
            return

        painter.setPen(QPen(QColor("#8ab6a6"), 2.0))
        painter.setBrush(QColor("#172721"))
        painter.drawRoundedRect(field, 10.0, 10.0)

        center_x = field.center().x()
        painter.setPen(QPen(QColor("#527466"), 1.2))
        painter.drawLine(QPointF(center_x, field.top()), QPointF(center_x, field.bottom()))
        center_circle_radius = field.height() * self.CENTER_CIRCLE_RADIUS_RATIO
        painter.drawEllipse(
            QPointF(center_x, field.center().y()),
            center_circle_radius,
            center_circle_radius,
        )

        goal_width = field.width() * 0.055
        goal_height = field.height() * 0.32
        goal_y = field.center().y() - goal_height / 2.0
        painter.setPen(QPen(QColor("#d47b4a"), 3.0))
        painter.drawLine(QPointF(field.left(), goal_y), QPointF(field.left(), goal_y + goal_height))
        painter.drawLine(QPointF(field.left(), goal_y), QPointF(field.left() + goal_width, goal_y))
        painter.drawLine(QPointF(field.left(), goal_y + goal_height), QPointF(field.left() + goal_width, goal_y + goal_height))
        painter.drawLine(QPointF(field.right(), goal_y), QPointF(field.right(), goal_y + goal_height))
        painter.drawLine(QPointF(field.right() - goal_width, goal_y), QPointF(field.right(), goal_y))
        painter.drawLine(QPointF(field.right() - goal_width, goal_y + goal_height), QPointF(field.right(), goal_y + goal_height))

        puck_x, puck_y, puck_vx, puck_vy = map(float, self._observation[:4])
        puck_center = self._field_point(field, puck_x, puck_y)
        radius = max(4.0, min(field.width(), field.height()) * 0.035)

        if self._sim_bridge is not None:
            trajectory = self._sim_bridge.predict_puck_trajectory()
            if len(trajectory) > 1:
                trajectory_line = QPolygonF()
                for point_x, point_y in trajectory:
                    trajectory_line.append(self._field_point(field, point_x, point_y))
                painter.setBrush(Qt.NoBrush)
                painter.setPen(
                    QPen(QColor("#ffcf33"), 2.5, Qt.DashLine, Qt.RoundCap, Qt.RoundJoin)
                )
                painter.drawPolyline(trajectory_line)

                robot_goal_x = -1.0 if float(self._observation[6]) <= 0.0 else 1.0
                final_x, final_y = trajectory[-1]
                if abs(final_x - robot_goal_x) <= 1e-6:
                    catch_point = self._field_point(field, final_x, final_y)
                    self._draw_goal_marker(painter, catch_point)

        if self._sim_bridge is not None:
            opponent_vx, opponent_vy = self._sim_bridge.get_opponent_velocity()
            if abs(opponent_vx) + abs(opponent_vy) > 1e-5:
                opponent_x = float(self._observation[4])
                opponent_y = float(self._observation[5])
                projection_seconds = 0.25
                opponent_start = self._field_point(field, opponent_x, opponent_y)
                opponent_end = self._field_point(
                    field,
                    opponent_x + opponent_vx * projection_seconds,
                    opponent_y + opponent_vy * projection_seconds,
                )
                self._draw_action_vector(
                    painter,
                    opponent_start,
                    opponent_end,
                    QColor("#ed5968"),
                    label=None,
                    width=2.0,
                    style=Qt.DashLine,
                )

        velocity_end = self._field_point(
            field,
            puck_x + puck_vx * 3.0,
            puck_y + puck_vy * 3.0,
        )
        painter.setPen(QPen(QColor("#ffd166"), 2.0, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(puck_center, velocity_end)
        self._draw_action_vector(
            painter,
            puck_center,
            velocity_end,
            QColor("#ffd166"),
            label=None,
            width=2.0,
        )

        painter.setPen(QPen(QColor("#fff0b3"), 1.5))
        painter.setBrush(QColor("#f4b942"))
        painter.drawEllipse(puck_center, radius, radius)

        opponent_center = self._field_point(
            field, self._observation[4], self._observation[5]
        )
        opponent_radius = max(8.0, min(field.width(), field.height()) * 0.065)
        painter.setPen(QPen(QColor("#ffb4a2"), 2.0))
        painter.setBrush(QColor("#d94f45"))
        painter.drawEllipse(opponent_center, opponent_radius, opponent_radius)

        robot_center = self._field_point(field, self._observation[6], self._observation[7])
        robot_radius = self._robot_marker_radius(field)
        painter.setPen(QPen(QColor("#56cfe1"), 2.0))
        painter.setBrush(QColor(25, 91, 101, 180))
        painter.drawEllipse(robot_center, robot_radius, robot_radius)
        painter.setPen(QPen(QColor("#8be9f5"), 3.0, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(
            QPointF(robot_center.x() - robot_radius * 0.55, robot_center.y() - robot_radius * 0.45),
            QPointF(robot_center.x() - robot_radius * 0.55, robot_center.y() + robot_radius * 0.45),
        )
        painter.drawLine(
            QPointF(robot_center.x() + robot_radius * 0.55, robot_center.y() - robot_radius * 0.45),
            QPointF(robot_center.x() + robot_radius * 0.55, robot_center.y() + robot_radius * 0.45),
        )
        painter.drawLine(
            QPointF(robot_center.x() - robot_radius * 0.55, robot_center.y()),
            QPointF(robot_center.x() + robot_radius * 0.55, robot_center.y()),
        )
        if self._sac_action is not None:
            vx, vy = self._sac_action
            painter.setPen(QPen(QColor("#c4d2cb"), 1.0))
            painter.drawText(
                QRectF(field.left() + 8.0, field.bottom() - 24.0, field.width() - 16.0, 18.0),
                Qt.AlignLeft | Qt.AlignVCenter,
                f"Vx: {vx:.2f} m/s | Vy: {vy:.2f} m/s",
            )

        action_endpoints = self.get_action_vector_endpoints(field)
        if action_endpoints is not None and self._sac_action is not None:
            vx_endpoint, vy_endpoint, target_center = action_endpoints
            self._draw_action_vector(
                painter,
                robot_center,
                vx_endpoint,
                QColor("#f25f5c"),
                "Vx",
                label_offset=(7.0, -9.0),
                width=3.0,
            )
            self._draw_action_vector(
                painter,
                robot_center,
                vy_endpoint,
                QColor("#70d687"),
                "Vy",
                label_offset=(8.0, 13.0),
                width=3.0,
            )
            self._draw_action_vector(
                painter,
                robot_center,
                target_center,
                QColor("#ffb000"),
                None,
                width=3.0,
            )
            target_radius = max(4.0, min(field.width(), field.height()) * 0.025)
            painter.setPen(QPen(QColor("#fff2a6"), 2.0))
            painter.setBrush(QColor("#ffb000"))
            painter.drawEllipse(target_center, target_radius, target_radius)
        painter.end()

    @staticmethod
    def _draw_action_vector(
        painter: QPainter,
        start: QPointF,
        end: QPointF,
        color: QColor,
        label: str | None,
        vertical_offset: float = 0.0,
        label_offset: tuple[float, float] = (0.0, -6.0),
        width: float = 2.5,
        style: Qt.PenStyle = Qt.SolidLine,
    ) -> None:
        """Draw a colored vector with a filled triangular head and optional label."""
        dx = end.x() - start.x()
        dy = end.y() - start.y()
        length = float(np.hypot(dx, dy))
        if length < 2.0:
            return

        ux, uy = dx / length, dy / length
        head_length = min(10.0, max(6.0, length * 0.35))
        half_width = head_length * 0.55
        base_x = end.x() - ux * head_length
        base_y = end.y() - uy * head_length
        left = QPointF(base_x - uy * half_width, base_y + ux * half_width)
        right = QPointF(base_x + uy * half_width, base_y - ux * half_width)

        painter.setPen(QPen(color, width, style, Qt.RoundCap))
        painter.drawLine(start, QPointF(base_x, base_y))
        painter.setPen(QPen(color, 1.0))
        painter.setBrush(color)
        arrow_head = QPolygonF()
        arrow_head.append(end)
        arrow_head.append(left)
        arrow_head.append(right)
        painter.drawPolygon(arrow_head)

        if label:
            label_point = QPointF(
                end.x() + label_offset[0],
                end.y() + vertical_offset + label_offset[1],
            )
            painter.setPen(QPen(QColor("#080b0a"), 4.0))
            painter.setBrush(Qt.NoBrush)
            for offset_x, offset_y in (
                (-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0),
                (-1.0, -1.0), (1.0, 1.0),
            ):
                painter.drawText(
                    QPointF(label_point.x() + offset_x, label_point.y() + offset_y),
                    label,
                )
            painter.setPen(QPen(color, 1.0))
            painter.drawText(label_point, label)

    @staticmethod
    def _draw_goal_marker(painter: QPainter, point: QPointF) -> None:
        """Draw a small crosshair at the predicted robot-side intercept."""
        painter.setPen(QPen(QColor("#fff2a6"), 2.0, Qt.SolidLine, Qt.RoundCap))
        painter.setBrush(QColor("#ffcf33"))
        painter.drawEllipse(point, 6.0, 6.0)
        painter.drawLine(QPointF(point.x() - 10.0, point.y()), QPointF(point.x() + 10.0, point.y()))
        painter.drawLine(QPointF(point.x(), point.y() - 10.0), QPointF(point.x(), point.y() + 10.0))


class DigitalTwinWindow(QDialog):
    """Resizable standalone window containing the live digital twin."""

    def __init__(self, sim_bridge: SimBridge | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Orion Vision | Air Hockey Digital Twin")
        self.resize(960, 540)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.twin_widget = DigitalTwinWidget(sim_bridge, parent=self)
        layout.addWidget(self.twin_widget)

    def update_bridge(self, sim_bridge: SimBridge | None) -> None:
        """Set the live worker bridge shown in this window."""
        self.twin_widget.update_bridge(sim_bridge)

    def update_sac_action(self, vx: float, vy: float) -> None:
        """Forward current SAC target velocity to the twin widget."""
        self.twin_widget.update_sac_action(vx, vy)
