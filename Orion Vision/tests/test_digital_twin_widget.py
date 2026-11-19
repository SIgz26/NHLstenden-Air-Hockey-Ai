"""Standalone smoke test and dummy-data demo for DigitalTwinWidget."""

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PyQt5.QtCore import QPoint, Qt, QTimer
from PyQt5.QtGui import QColor, QImage
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

from core.sim_bridge import SimBridge
from ui.widgets.digital_twin_widget import DigitalTwinWidget, DigitalTwinWindow


def test_scripted_plc_velocity_signs_align_with_twin_action_axes():
    app = QApplication.instance() or QApplication([])
    widget = DigitalTwinWidget()
    field = widget._field_rect()
    observation = np.zeros(8, dtype=np.float32)
    observation[7] = 0.0
    widget.update_observation_vector(observation)

    plc_vx_mm_s = 120.0
    plc_vy_mm_s = -160.0
    widget.update_sac_action(
        plc_vx_mm_s / 1000.0,
        -plc_vy_mm_s / 1000.0,
    )

    robot = widget._robot_point(field)
    endpoints = widget.get_action_vector_endpoints(field)
    assert endpoints is not None
    vx_endpoint, vy_endpoint, result_endpoint = endpoints
    assert vx_endpoint.x() == robot.x()
    assert vx_endpoint.y() < robot.y()
    assert vy_endpoint.x() > robot.x()
    assert vy_endpoint.y() == robot.y()
    assert result_endpoint.x() > robot.x()
    assert result_endpoint.y() < robot.y()

    widget.close()
    app.processEvents()


def test_puck_uses_robot_screen_axes_for_position_and_velocity():
    app = QApplication.instance() or QApplication([])
    widget = DigitalTwinWidget()
    field = widget._field_rect()

    own_goal_and_bottom = widget._puck_point(field, -1.0, -1.0)
    opponent_goal_and_top = widget._puck_point(field, 1.0, 1.0)
    assert own_goal_and_bottom.x() == field.left()
    assert own_goal_and_bottom.y() == field.bottom()
    assert opponent_goal_and_top.x() == field.right()
    assert opponent_goal_and_top.y() == field.top()

    puck_x, puck_y = -0.5, -0.5
    start = widget._puck_point(field, puck_x, puck_y)
    width_axis_end = widget._puck_point(field, puck_x + 0.5, puck_y)
    length_axis_end = widget._puck_point(field, puck_x, puck_y + 0.5)
    assert width_axis_end.x() == start.x()
    assert width_axis_end.y() < start.y()
    assert length_axis_end.x() > start.x()
    assert length_axis_end.y() == start.y()

    velocity_end = widget._puck_point(
        field,
        puck_x + 0.25 * 3.0,
        puck_y + 0.5 * 3.0,
    )
    assert velocity_end.x() > start.x()
    assert velocity_end.y() < start.y()

    widget.close()
    app.processEvents()


def test_mouse_drag_emits_clamped_table_position_only_when_enabled():
    app = QApplication.instance() or QApplication([])
    bridge = SimBridge(invert_y=True)
    bridge.set_table_corners(
        top_left=(0.0, 0.0),
        top_right=(1020.0, 0.0),
        bottom_right=(1020.0, 1940.0),
        bottom_left=(0.0, 1940.0),
    )
    bridge.configure_physical_table_bounds()
    bridge.update(510.0, 970.0, 0.0, 0.0, frame_rate=30.0)
    bridge.set_robot_position_mm(bridge.PLC_X_CENTER_MM, 421.6)

    widget = DigitalTwinWidget(bridge)
    widget.resize(600, 400)
    widget.show()
    app.processEvents()
    field = widget._field_rect()
    robot = widget._robot_point(field)
    targets = []
    widget.manual_target_requested.connect(
        lambda x_mm, y_mm: targets.append((x_mm, y_mm))
    )

    press = QPoint(round(robot.x()), round(robot.y()))
    move = QPoint(round(field.center().x() - 15), round(field.top() + 30))
    QTest.mousePress(widget, Qt.LeftButton, Qt.NoModifier, press)
    QTest.mouseRelease(widget, Qt.LeftButton, Qt.NoModifier, press)
    assert not targets

    widget.set_manual_control_enabled(True)
    QTest.mousePress(widget, Qt.LeftButton, Qt.NoModifier, press)
    QTest.mouseMove(widget, move)
    QTest.mouseRelease(widget, Qt.LeftButton, Qt.NoModifier, move)

    assert targets
    expected = widget._table_target_from_point(move, field)
    assert expected is not None
    np.testing.assert_allclose(targets[-1], expected, atol=1e-6)
    assert targets[-1][1] > 0.0
    half_x_range = (
        bridge.PLC_X_BOUNDS_MM[1] - bridge.PLC_X_BOUNDS_MM[0]
    ) / 2.0
    assert -half_x_range <= targets[-1][0] <= half_x_range

    widget.close()
    app.processEvents()


def test_mouse_target_maps_to_plc_robot_workspace():
    app = QApplication.instance() or QApplication([])
    bridge = SimBridge()
    bridge.set_table_corners(
        top_left=(0.0, 0.0),
        top_right=(1020.0, 0.0),
        bottom_right=(1020.0, 1940.0),
        bottom_left=(0.0, 1940.0),
    )
    bridge.configure_physical_table_bounds()
    widget = DigitalTwinWidget(bridge)
    widget.resize(600, 400)
    field = widget._field_rect()

    for point in (
        QPoint(round(field.left()), round(field.top())),
        QPoint(round(field.left()), round(field.bottom())),
        QPoint(round(field.center().x()), round(field.center().y())),
    ):
        table_target = widget._table_target_from_point(point, field)
        assert table_target is not None
        plc_target = bridge.physical_target_to_plc_mm(*table_target)
        assert bridge.PLC_X_BOUNDS_MM[0] - 1e-6 <= plc_target[0] <= bridge.PLC_X_BOUNDS_MM[1] + 1e-6
        assert bridge.PLC_Y_BOUNDS_MM[0] - 1e-6 <= plc_target[1] <= bridge.PLC_Y_BOUNDS_MM[1] + 1e-6

    top_target = widget._table_target_from_point(
        QPoint(round(field.center().x() - 10), round(field.top())),
        field,
    )
    bottom_target = widget._table_target_from_point(
        QPoint(round(field.center().x() - 10), round(field.bottom())),
        field,
    )
    assert top_target is not None
    assert bottom_target is not None
    assert bridge.physical_target_to_plc_mm(*top_target)[0] > 0.0
    assert bridge.physical_target_to_plc_mm(*bottom_target)[0] < 0.0

    widget.close()
    app.processEvents()


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
    assert abs(
        field.width() / field.height()
        - DigitalTwinWidget.FIELD_WIDTH_MM / DigitalTwinWidget.FIELD_DEPTH_MM
    ) < 1e-6
    assert DigitalTwinWidget.FIELD_WIDTH_MM == 1940.0
    assert DigitalTwinWidget.FIELD_DEPTH_MM == 1020.0
    assert DigitalTwinWidget._twin_point(field, -1.0, -1.0).x() == field.left()
    assert DigitalTwinWidget._twin_point(field, -1.0, -1.0).y() == field.top()
    assert DigitalTwinWidget._twin_point(field, 1.0, 1.0).x() == field.right()
    assert DigitalTwinWidget._twin_point(field, 1.0, 1.0).y() == field.bottom()
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
    robot = widget._twin_point(field, *_make_dummy_bridge().get_observation_vector()[6:8])
    endpoints = widget.get_action_vector_endpoints(field)
    assert endpoints is not None
    vx_endpoint, vy_endpoint, target_endpoint = endpoints
    assert vx_endpoint.y() > robot.y()
    assert vy_endpoint.x() > robot.x()
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
    observation = bridge.get_observation_vector()
    observation[6:8] = 0.0
    widget.update_observation_vector(observation)
    widget.update_sac_action(0.05, -0.01)

    field = widget._field_rect()
    robot = widget._twin_point(field, float(observation[6]), float(observation[7]))
    endpoints = widget.get_action_vector_endpoints(field)
    assert endpoints is not None
    vx_endpoint, vy_endpoint, resultant_endpoint = endpoints
    clearances = widget.get_action_endpoint_clearances(field)
    robot_radius = widget._robot_marker_radius(field)
    assert clearances is not None
    assert all(distance >= robot_radius + 10.0 for distance in clearances)

    assert abs(vx_endpoint.y() - robot.y()) >= widget.MIN_ACTION_VECTOR_PIXELS - 0.01
    assert abs(vx_endpoint.x() - robot.x()) < 0.01
    assert abs(vy_endpoint.x() - robot.x()) >= widget.MIN_ACTION_VECTOR_PIXELS - 0.01
    assert abs(vy_endpoint.y() - robot.y()) < 0.01
    assert vy_endpoint.x() > robot.x()
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
        puck_vx=0.0,
        puck_vy=-0.5,
        frame_rate=1.0,
        robot_x=0.1,
        robot_y=1.0,
        opponent_x=1.8,
        opponent_y=1.0,
    )
    widget = DigitalTwinWidget(bridge)
    widget.resize(480, 280)
    widget.update_sac_action(0.2, 0.1)

    field = widget._field_rect()
    trajectory = bridge.predict_puck_trajectory()
    assert trajectory[-1][1] == -1.0
    assert trajectory[-1][0] == -0.5
    trajectory_start = widget._puck_point(field, *trajectory[0])
    trajectory_end = widget._puck_point(field, *trajectory[-1])
    assert trajectory_end.x() < trajectory_start.x()
    assert trajectory_end.y() == trajectory_start.y()

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
