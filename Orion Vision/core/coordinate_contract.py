"""Shared SI coordinate, observation, and H-bot action contract."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

TABLE_LENGTH_M = 1.94
TABLE_WIDTH_M = 1.02
FIELD_LENGTH_M = TABLE_LENGTH_M
FIELD_WIDTH_M = TABLE_WIDTH_M

PUCK_MAX_SPEED_MPS = 5.0
MALLET_FORWARD_MAX_SPEED_MPS = 1.0
MALLET_REVERSE_MAX_SPEED_MPS = 0.85
MALLET_LATERAL_MAX_SPEED_MPS = 0.85
MALLET_MAX_ACCELERATION_MPS2 = 2.8
MOTOR_MAX_SPEED_MPS = 1.0
CONTROL_DT_S = 0.01
MAX_RELATIVE_SPEED_MPS = PUCK_MAX_SPEED_MPS + MOTOR_MAX_SPEED_MPS

PLC_X_BOUNDS_MM = (-463.2, 370.5)
PLC_Y_BOUNDS_MM = (53.7, 789.5)
EDGE_MARGIN_M = 0.095
ROBOT_LENGTH_BOUNDS_M = (
    -(PLC_Y_BOUNDS_MM[1] - PLC_Y_BOUNDS_MM[0]) / 1000.0 + EDGE_MARGIN_M,
    -EDGE_MARGIN_M,
)
ROBOT_WIDTH_BOUNDS_M = (
    -TABLE_WIDTH_M / 2.0 + EDGE_MARGIN_M,
    TABLE_WIDTH_M / 2.0 - EDGE_MARGIN_M,
)


def _vector2(values: Sequence[float], name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.shape != (2,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain two finite values")
    return vector


def physical_state_to_observation(
    puck_position_m: Sequence[float],
    puck_velocity_mps: Sequence[float],
    opponent_position_m: Sequence[float],
    robot_position_m: Sequence[float],
    robot_velocity_mps: Sequence[float],
) -> np.ndarray:
    """Convert SI states in [length, width] axes to the shared 12D vector."""
    puck_position = _vector2(puck_position_m, "puck_position_m")
    puck_velocity = _vector2(puck_velocity_mps, "puck_velocity_mps")
    opponent_position = _vector2(opponent_position_m, "opponent_position_m")
    robot_position = _vector2(robot_position_m, "robot_position_m")
    robot_velocity = _vector2(robot_velocity_mps, "robot_velocity_mps")

    observation = np.asarray(
        [
            puck_position[0] / (FIELD_LENGTH_M / 2.0),
            puck_position[1] / (FIELD_WIDTH_M / 2.0),
            puck_velocity[0] / PUCK_MAX_SPEED_MPS,
            puck_velocity[1] / PUCK_MAX_SPEED_MPS,
            opponent_position[0] / (FIELD_LENGTH_M / 2.0),
            opponent_position[1] / (FIELD_WIDTH_M / 2.0),
            robot_position[0] / (FIELD_LENGTH_M / 2.0),
            robot_position[1] / (FIELD_WIDTH_M / 2.0),
            (puck_position[0] - robot_position[0]) / FIELD_LENGTH_M,
            (puck_position[1] - robot_position[1]) / FIELD_WIDTH_M,
            (puck_velocity[0] - robot_velocity[0]) / MAX_RELATIVE_SPEED_MPS,
            (puck_velocity[1] - robot_velocity[1]) / MAX_RELATIVE_SPEED_MPS,
        ],
        dtype=np.float32,
    )
    return np.clip(observation, -1.0, 1.0)


def limit_plc_velocity_mps(vx_mps: float, vy_mps: float) -> tuple[float, float]:
    """Apply signed PLC axis and coupled H-bot motor-speed limits in m/s."""
    velocity = _vector2((vx_mps, vy_mps), "PLC velocity")
    vx = float(np.clip(
        velocity[0],
        -MALLET_LATERAL_MAX_SPEED_MPS,
        MALLET_LATERAL_MAX_SPEED_MPS,
    ))
    vy_limit = (
        MALLET_FORWARD_MAX_SPEED_MPS
        if velocity[1] < 0.0
        else MALLET_REVERSE_MAX_SPEED_MPS
    )
    vy = float(np.clip(velocity[1], -vy_limit, vy_limit))

    motor_speed = abs(vx) + abs(vy)
    if motor_speed > MOTOR_MAX_SPEED_MPS:
        scale = MOTOR_MAX_SPEED_MPS / motor_speed
        vx *= scale
        vy *= scale
    return vx, vy


def action_to_velocity_mps(action: Sequence[float]) -> np.ndarray:
    """Convert normalized [forward-length, lateral-width] action to SI velocity."""
    normalized_action = _vector2(action, "action")
    normalized_action = np.clip(normalized_action, -1.0, 1.0)
    max_length_speed = (
        MALLET_FORWARD_MAX_SPEED_MPS
        if normalized_action[0] >= 0.0
        else MALLET_REVERSE_MAX_SPEED_MPS
    )
    length_speed = float(normalized_action[0]) * max_length_speed
    width_speed = float(normalized_action[1]) * MALLET_LATERAL_MAX_SPEED_MPS

    plc_vx, plc_vy = limit_plc_velocity_mps(width_speed, -length_speed)
    return np.asarray([-plc_vy, plc_vx], dtype=np.float64)


def action_to_plc_velocity_mm_s(action: Sequence[float]) -> tuple[float, float]:
    """Convert normalized SAC action to bounded PLC (X, Y) targets in mm/s."""
    length_speed, width_speed = action_to_velocity_mps(action)
    return width_speed * 1000.0, -length_speed * 1000.0


def velocity_mps_to_action(velocity_mps: Sequence[float]) -> np.ndarray:
    """Map a feasible [length, width] SI velocity back to normalized action."""
    length_speed, width_speed = _vector2(velocity_mps, "velocity_mps")
    length_limit = (
        MALLET_FORWARD_MAX_SPEED_MPS
        if length_speed >= 0.0
        else MALLET_REVERSE_MAX_SPEED_MPS
    )
    return np.clip(
        np.asarray(
            [
                length_speed / length_limit,
                width_speed / MALLET_LATERAL_MAX_SPEED_MPS,
            ],
            dtype=np.float32,
        ),
        -1.0,
        1.0,
    )
