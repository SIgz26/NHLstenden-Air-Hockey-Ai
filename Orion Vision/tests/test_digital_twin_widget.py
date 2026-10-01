"""Standalone smoke test and dummy-data demo for DigitalTwinWidget."""

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QColor, QImage
from PyQt5.QtWidgets import QApplication

from core.sim_bridge import SimBridge
from ui.widgets.digital_twin_widget import DigitalTwinWidget, DigitalTwinWindow


def _make_dummy_bridge() -> SimBridge:
    bridge = SimBridge(
        x_bounds=(0.0, 1000.0),
        y_bounds=(0.0, 500.0),
        sim_dt=1.0 / 60.0,
    )
    bridge.update(
        puck_x=620.0,
        puck_y=190.0,
        puck_vx=3.0,
        puck_vy=-1.5,
        frame_rate=120.0,
        opponent_x=800.0,
        opponent_y=250.0,
        robot_x=150.0,
        robot_y=250.0,
    )
    return bridge


def test_digital_twin_renders_dummy_observation():
    app = QApplication.instance() or QApplication([])
    bridge = _make_dummy_bridge()
    widget = DigitalTwinWidget(bridge)
    widget.resize(480, 280)

    image = QImage(widget.size(), QImage.Format_ARGB32)
    image.fill(QColor("#000000"))
    widget.render(image)

    assert widget._timer.interval() == 33
    assert widget._observation.shape == (8,)
    assert np.isfinite(widget._observation).all()
    field = widget._field_rect()
    assert abs(field.width() / field.height() - 1.8) < 1e-6
    assert image.pixelColor(20, 20) != QColor("#000000")

    widget.close()
    app.processEvents()


def test_digital_twin_window_displays_active_bridge():
    app = QApplication.instance() or QApplication([])
    bridge = _make_dummy_bridge()
    window = DigitalTwinWindow(bridge)

    np.testing.assert_array_equal(
        window.twin_widget._observation,
        bridge.get_observation_vector(),
    )
    assert window.width() >= 900

    window.close()
    app.processEvents()


def test_sac_action_target_vector_scales_from_robot_and_renders_at_sizes():
    app = QApplication.instance() or QApplication([])
    bridge = _make_dummy_bridge()
    widget = DigitalTwinWidget(bridge)
    widget.resize(480, 280)
    widget.update_sac_action(0.8, -0.4)

    field = widget._field_rect()
    robot = widget._field_point(field, *_make_dummy_bridge().get_observation_vector()[6:8])
    endpoints = widget.get_action_vector_endpoints(field)
    assert endpoints is not None
    vx_endpoint, vy_endpoint, target_endpoint = endpoints
    assert vx_endpoint.x() > robot.x()
    assert vy_endpoint.y() > robot.y()
    assert target_endpoint.x() > robot.x()
    assert target_endpoint.y() > robot.y()
    assert widget.get_robot_action_target() is not None
    clearances = widget.get_action_endpoint_clearances(field)
    assert clearances is not None
    assert all(distance > widget._robot_marker_radius(field) for distance in clearances)

    for size in ((480, 280), (300, 220), (900, 500)):
        widget.resize(*size)
        image = QImage(widget.size(), QImage.Format_ARGB32)
        image.fill(QColor("#000000"))
        widget.render(image)
        assert image.width() == size[0]
        assert image.height() == size[1]
        assert widget._field_rect().width() > 0

    widget.update_sac_action(100.0, -100.0)
    clipped_target = widget.get_robot_action_target()
    assert clipped_target is not None
    assert clipped_target[0] > 0.97
    assert clipped_target[1] < -0.97

    widget.close()
    app.processEvents()


def test_small_sac_actions_get_visible_component_and_resultant_vectors():
    app = QApplication.instance() or QApplication([])
    bridge = _make_dummy_bridge()
    widget = DigitalTwinWidget(bridge)
    widget.resize(480, 280)
    widget.update_sac_action(0.05, -0.01)

    field = widget._field_rect()
    observation = bridge.get_observation_vector()
    robot = widget._field_point(field, float(observation[6]), float(observation[7]))
    endpoints = widget.get_action_vector_endpoints(field)
    assert endpoints is not None
    vx_endpoint, vy_endpoint, resultant_endpoint = endpoints
    clearances = widget.get_action_endpoint_clearances(field)
    robot_radius = widget._robot_marker_radius(field)
    assert clearances is not None
    assert all(distance >= robot_radius + 10.0 for distance in clearances)

    assert abs(vx_endpoint.x() - robot.x()) >= widget.MIN_ACTION_VECTOR_PIXELS - 0.01
    assert abs(vx_endpoint.y() - robot.y()) < 0.01
    assert abs(vy_endpoint.y() - robot.y()) >= widget.MIN_ACTION_VECTOR_PIXELS - 0.01
    assert abs(vy_endpoint.x() - robot.x()) < 0.01
    assert vy_endpoint.y() > robot.y()
    assert np.hypot(
        resultant_endpoint.x() - robot.x(),
        resultant_endpoint.y() - robot.y(),
    ) >= widget.MIN_ACTION_VECTOR_PIXELS - 0.01

    image = QImage(widget.size(), QImage.Format_ARGB32)
    image.fill(QColor("#000000"))
    widget.render(image)
    center_pixel = image.pixelColor(round(robot.x()), round(robot.y()))
    assert center_pixel.red() > 200 and center_pixel.green() > 100
    assert image.pixelColor(round(vx_endpoint.x()), round(vx_endpoint.y())) != QColor("#000000")
    assert image.pixelColor(round(vy_endpoint.x()), round(vy_endpoint.y())) != QColor("#000000")
    assert image.pixelColor(round(resultant_endpoint.x()), round(resultant_endpoint.y())) != QColor("#000000")

    widget.close()
    app.processEvents()


def test_predicted_puck_path_and_robot_goal_intercept_render():
    app = QApplication.instance() or QApplication([])
    bridge = SimBridge(
        x_bounds=(0.0, 2.0),
        y_bounds=(0.0, 2.0),
        sim_dt=1.0,
        invert_y=False,
    )
    bridge.update(
        puck_x=0.5,
        puck_y=1.0,
        puck_vx=-0.5,
        puck_vy=0.0,
        frame_rate=1.0,
        robot_x=0.1,
        robot_y=1.0,
        opponent_x=1.8,
        opponent_y=1.0,
    )
    widget = DigitalTwinWidget(bridge)
    widget.resize(480, 280)
    widget.update_sac_action(0.2, 0.1)

    trajectory = bridge.predict_puck_trajectory()
    assert trajectory[-1][0] == -1.0
    assert abs(trajectory[-1][1]) < 1e-6

    image = QImage(widget.size(), QImage.Format_ARGB32)
    image.fill(QColor("#000000"))
    widget.render(image)
    assert image.pixelColor(20, 20) != QColor("#000000")

    widget.close()
    app.processEvents()


def run_dummy_demo() -> None:
    """Open a small animated twin window using generated telemetry."""
    app = QApplication.instance() or QApplication(sys.argv)
    bridge = SimBridge(
        x_bounds=(0.0, 1000.0),
        y_bounds=(0.0, 500.0),
        sim_dt=1.0 / 60.0,
    )
    bridge.set_opponent_position(820.0, 250.0)
    bridge.set_robot_position(160.0, 250.0)

    widget = DigitalTwinWidget(bridge, refresh_hz=30.0)
    widget.setWindowTitle("Orion Vision | Digital Twin Demo")
    widget.resize(640, 360)
    widget.show()

    state = {"phase": 0.0}

    def push_dummy_observation() -> None:
        phase = state["phase"]
        puck_x = 500.0 + 300.0 * np.sin(phase)
        puck_y = 250.0 + 150.0 * np.sin(phase * 1.7)
        puck_vx = 3.0 * np.cos(phase)
        puck_vy = 2.55 * np.cos(phase * 1.7)
        bridge.update(
            puck_x=puck_x,
            puck_y=puck_y,
            puck_vx=puck_vx,
            puck_vy=puck_vy,
            frame_rate=120.0,
        )
        state["phase"] = phase + 0.06

    demo_timer = QTimer(widget)
    demo_timer.timeout.connect(push_dummy_observation)
    demo_timer.start(16)
    app.exec_()


if __name__ == "__main__":
    run_dummy_demo()
