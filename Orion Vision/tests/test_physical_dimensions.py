"""Verify shared puck and mallet dimensions across physics and rendering."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtCore import QRectF

from ai.air_hockey_env import AirHockeyGymEnv
from core.coordinate_contract import (
    MALLET_DIAMETER_M,
    MALLET_DIAMETER_MM,
    MALLET_PUCK_COLLISION_DISTANCE_M,
    MALLET_PUCK_COLLISION_DISTANCE_MM,
    MALLET_RADIUS_M,
    MALLET_RADIUS_MM,
    PUCK_DIAMETER_M,
    PUCK_DIAMETER_MM,
    PUCK_RADIUS_M,
    PUCK_RADIUS_MM,
)
from core.sim_bridge import SimBridge
from ui.widgets.digital_twin_widget import DigitalTwinWidget


def test_physical_dimensions_are_shared_by_simulation_and_digital_twin():
    assert MALLET_DIAMETER_M == 0.095
    assert MALLET_RADIUS_M == 0.0475
    assert MALLET_DIAMETER_MM == 95.0
    assert MALLET_RADIUS_MM == 47.5
    assert PUCK_DIAMETER_M == 0.073
    assert PUCK_RADIUS_M == 0.0365
    assert PUCK_DIAMETER_MM == 73.0
    assert PUCK_RADIUS_MM == 36.5
    assert MALLET_PUCK_COLLISION_DISTANCE_M == 0.084
    assert MALLET_PUCK_COLLISION_DISTANCE_MM == 84.0

    for consumer in (AirHockeyGymEnv, SimBridge, DigitalTwinWidget):
        assert consumer.MALLET_RADIUS_M == 0.0475
        assert consumer.PUCK_RADIUS_M == 0.0365
        assert consumer.MALLET_PUCK_COLLISION_DISTANCE_M == 0.084

    physical_field = QRectF(0.0, 0.0, 1940.0, 1020.0)
    assert DigitalTwinWidget._robot_marker_radius(physical_field) == 47.5
    assert DigitalTwinWidget._puck_marker_radius(physical_field) == 36.5
    assert DigitalTwinWidget.FIELD_ASPECT_RATIO == pytest.approx(1940.0 / 1020.0)


@pytest.mark.parametrize(
    ("position", "velocity", "expected_position", "expected_velocity", "axis"),
    [
        ((0.0, 0.4745), (0.0, 1.0), (0.0, 0.4735), (0.0, -0.9), 1),
        ((0.9345, 0.2), (1.0, 0.0), (0.9335, 0.2), (-0.9, 0.0), 0),
    ],
)
def test_puck_wall_limits_use_the_shared_physical_radius(
    position, velocity, expected_position, expected_velocity, axis
):
    env = AirHockeyGymEnv(seed=1)
    env._puck_position[:] = position
    env._puck_velocity[:] = velocity

    env._resolve_puck_walls()

    assert env._puck_position[axis] == pytest.approx(expected_position[axis])
    assert env._puck_velocity[axis] == pytest.approx(expected_velocity[axis])
