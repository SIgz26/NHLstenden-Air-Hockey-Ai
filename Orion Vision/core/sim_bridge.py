"""Translate Orion Vision telemetry into a Gymnasium-style observation.

The bridge keeps only the newest observation in a one-item queue, so a slow
simulation consumer cannot create an ever-growing backlog. Workers can publish
with :meth:`update_from_engine`; the simulation can consume the newest vector
with :meth:`get_observation_vector` or ``bridge.observations.get_nowait()``.
"""

from __future__ import annotations

from queue import Empty, Full, Queue
import time
from typing import Any, Mapping

import numpy as np
import cv2


class SimBridge:
    """Normalize vision telemetry and hand observations to a local simulator.

    Coordinates are normalized independently to ``[-1, 1]`` using configurable
    table bounds. Kalman velocities are assumed to be source-units per camera
    frame and are converted to normalized displacement per simulation step.

    Args:
        x_bounds: Minimum and maximum source-space x coordinates.
        y_bounds: Minimum and maximum source-space y coordinates.
        sim_dt: Duration of one simulation step in seconds.
        invert_y: Invert image-style y coordinates (down-positive) for a
            simulation whose y-axis is up-positive.
        queue_observations: Create a one-item, latest-value queue for a local
            simulation consumer.
    """

    def __init__(
        self,
        x_bounds: tuple[float, float] = (0.0, 1.0),
        y_bounds: tuple[float, float] = (0.0, 1.0),
        sim_dt: float = 1.0 / 60.0,
        invert_y: bool = True,
        queue_observations: bool = True,
    ) -> None:
        self._validate_bounds("x_bounds", x_bounds)
        self._validate_bounds("y_bounds", y_bounds)
        if sim_dt <= 0.0:
            raise ValueError("sim_dt must be greater than zero")

        self.x_bounds = tuple(map(float, x_bounds))
        self.y_bounds = tuple(map(float, y_bounds))
        self.table_x_min, self.table_x_max = self.x_bounds
        self.table_y_min, self.table_y_max = self.y_bounds
        self._table_bounds_explicit = False
        self._frame_size: tuple[int, int] | None = None
        self._perspective_matrix: np.ndarray | None = None
        self._table_corners: tuple[
            tuple[float, float], tuple[float, float],
            tuple[float, float], tuple[float, float],
        ] | None = None
        self.sim_dt = float(sim_dt)
        self.invert_y = bool(invert_y)
        self.observations: Queue[np.ndarray] | None = (
            Queue(maxsize=1) if queue_observations else None
        )
        self._observation = np.zeros(8, dtype=np.float32)
        self._raw_puck: tuple[float, float, float, float, float] | None = None
        self._opponent_xy: tuple[float, float] | None = None
        self._robot_xy: tuple[float, float] | None = None
        self._opponent_velocity: tuple[float, float] = (0.0, 0.0)
        self._opponent_sample: tuple[tuple[float, float], float] | None = None

    @staticmethod
    def _validate_bounds(name: str, bounds: tuple[float, float]) -> None:
        if len(bounds) != 2 or not bounds[0] < bounds[1]:
            raise ValueError(f"{name} must be a (minimum, maximum) pair")

    @staticmethod
    def _normalize(value: float, bounds: tuple[float, float]) -> float:
        low, high = bounds
        return float(np.clip(2.0 * (value - low) / (high - low) - 1.0, -1.0, 1.0))

    def set_table_bounds(
        self,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
    ) -> None:
        """Set calibrated table edges in source pixel or millimeter coordinates.

        Calibration remains active across calls to :meth:`update_from_engine`.
        The pixel at each minimum/maximum edge maps exactly to ``-1``/``1``
        on X; Y follows the configured ``invert_y`` convention.
        """
        self._validate_bounds("x_bounds", (x_min, x_max))
        self._validate_bounds("y_bounds", (y_min, y_max))
        self.table_x_min = float(x_min)
        self.table_y_min = float(y_min)
        self.table_x_max = float(x_max)
        self.table_y_max = float(y_max)
        self.x_bounds = (self.table_x_min, self.table_x_max)
        self.y_bounds = (self.table_y_min, self.table_y_max)
        self._table_bounds_explicit = True
        if self._raw_puck is not None:
            self._observation = self._calculate_observation()
            self._publish_latest(self._observation)

    def set_table_corners(
        self,
        top_left: tuple[float, float],
        top_right: tuple[float, float],
        bottom_right: tuple[float, float],
        bottom_left: tuple[float, float],
    ) -> None:
        """Rectify four camera-space table corners to an axis-aligned table.

        Points must be supplied in clockwise order starting at the image's
        top-left. When explicit table bounds have not been set, the output
        rectangle is sized from the average opposing edge lengths. If table
        bounds are already configured, they define the output rectangle.

        Args:
            top_left: Camera pixel coordinate of the table's top-left corner.
            top_right: Camera pixel coordinate of the table's top-right corner.
            bottom_right: Camera pixel coordinate of the table's bottom-right corner.
            bottom_left: Camera pixel coordinate of the table's bottom-left corner.
        """
        source = np.asarray(
            [top_left, top_right, bottom_right, bottom_left], dtype=np.float32
        )
        if source.shape != (4, 2) or not np.isfinite(source).all():
            raise ValueError("table corners must be four finite (x, y) points")
        if abs(float(cv2.contourArea(source))) < 1.0:
            raise ValueError("table corners must form a non-degenerate quadrilateral")
        if not cv2.isContourConvex(source.reshape((-1, 1, 2))):
            raise ValueError("table corners must form a convex quadrilateral in TL, TR, BR, BL order")

        if not self._table_bounds_explicit:
            top_width = float(np.linalg.norm(source[1] - source[0]))
            bottom_width = float(np.linalg.norm(source[2] - source[3]))
            right_height = float(np.linalg.norm(source[2] - source[1]))
            left_height = float(np.linalg.norm(source[3] - source[0]))
            output_width = max(1.0, (top_width + bottom_width) / 2.0)
            output_height = max(1.0, (right_height + left_height) / 2.0)
            self.table_x_min = 0.0
            self.table_y_min = 0.0
            self.table_x_max = output_width - 1.0
            self.table_y_max = output_height - 1.0
            self.x_bounds = (self.table_x_min, self.table_x_max)
            self.y_bounds = (self.table_y_min, self.table_y_max)
            self._table_bounds_explicit = True

        destination = np.asarray(
            [
                [self.table_x_min, self.table_y_min],
                [self.table_x_max, self.table_y_min],
                [self.table_x_max, self.table_y_max],
                [self.table_x_min, self.table_y_max],
            ],
            dtype=np.float32,
        )
        self._perspective_matrix = cv2.getPerspectiveTransform(source, destination)
        self._table_corners = tuple(
            (float(point[0]), float(point[1])) for point in source
        )
        if self._raw_puck is not None:
            self._observation = self._calculate_observation()
            self._publish_latest(self._observation)

    def get_table_corners(
        self,
    ) -> tuple[
        tuple[float, float], tuple[float, float],
        tuple[float, float], tuple[float, float],
    ] | None:
        """Return calibrated source-image corners clockwise from top-left."""
        return self._table_corners

    def reset_table_corners(
        self,
        frame_width: int | None = None,
        frame_height: int | None = None,
    ) -> None:
        """Clear perspective calibration and use full-frame bounds if known."""
        self._perspective_matrix = None
        self._table_corners = None
        self._table_bounds_explicit = False
        if frame_width is not None and frame_height is not None:
            if frame_width <= 1 or frame_height <= 1:
                raise ValueError("frame dimensions must both be greater than one pixel")
            self.table_x_min = 0.0
            self.table_y_min = 0.0
            self.table_x_max = float(frame_width - 1)
            self.table_y_max = float(frame_height - 1)
            self.x_bounds = (self.table_x_min, self.table_x_max)
            self.y_bounds = (self.table_y_min, self.table_y_max)
            self._frame_size = (frame_width, frame_height)
        else:
            self.table_x_min = self.table_y_min = 0.0
            self.table_x_max = self.table_y_max = 1.0
            self.x_bounds = (0.0, 1.0)
            self.y_bounds = (0.0, 1.0)
            self._frame_size = None
        if self._raw_puck is not None:
            self._observation = self._calculate_observation()
            self._publish_latest(self._observation)

    def set_opponent_position(self, x: float | None, y: float | None) -> None:
        """Set the latest opponent position in the same units as table bounds."""
        self._opponent_xy = None if x is None or y is None else (float(x), float(y))

    def set_robot_position(self, x: float | None, y: float | None) -> None:
        """Set the robot position in the same units as table bounds."""
        self._robot_xy = None if x is None or y is None else (float(x), float(y))

    def update_telemetry(
        self,
        robot_position: tuple[float, float] | None = None,
        opponent_position: tuple[float, float] | None = None,
    ) -> np.ndarray:
        """Update detected mallet positions and return the full observation.

        Marker positions are stored in camera pixel coordinates. They are
        transformed through the configured table homography and normalized
        together with the latest puck telemetry. ``None`` leaves a previously
        detected position unchanged, avoiding jittery disappear/reappear
        behavior when an ArUco marker is momentarily occluded.
        """
        if opponent_position is not None:
            now = time.monotonic()
            opponent_normalized = self._normalize_optional(
                (float(opponent_position[0]), float(opponent_position[1]))
            )
            previous = self._opponent_sample
            if previous is not None:
                previous_position, previous_time = previous
                elapsed = now - previous_time
                if elapsed > 1e-6:
                    self._opponent_velocity = (
                        (opponent_normalized[0] - previous_position[0]) / elapsed,
                        (opponent_normalized[1] - previous_position[1]) / elapsed,
                    )
            self._opponent_sample = (opponent_normalized, now)
        if robot_position is not None:
            self._robot_xy = (float(robot_position[0]), float(robot_position[1]))
        if opponent_position is not None:
            self._opponent_xy = (
                float(opponent_position[0]),
                float(opponent_position[1]),
            )
        self._observation = self._calculate_observation()
        self._publish_latest(self._observation)
        return self._observation.copy()

    def get_opponent_velocity(self) -> tuple[float, float]:
        """Return the last normalized opponent velocity in table units/second."""
        return self._opponent_velocity

    def update(
        self,
        puck_x: float,
        puck_y: float,
        puck_vx: float,
        puck_vy: float,
        frame_rate: float,
        opponent_x: float | None = None,
        opponent_y: float | None = None,
        robot_x: float | None = None,
        robot_y: float | None = None,
    ) -> np.ndarray:
        """Normalize one telemetry sample and publish its state vector.

        Positions and velocities use the same source units (pixels or mm).
        ``puck_vx`` and ``puck_vy`` are per camera frame, as in the current
        Orion Kalman engines. Opponent and robot positions are optional; when
        absent, their normalized coordinates are represented by the table
        center (zero).

        Returns:
            A copy of ``[puck_x, puck_y, puck_vx, puck_vy, opp_x, opp_y,
            robot_x, robot_y]`` as a float32 NumPy array. Velocity values are
            normalized displacement per ``sim_dt`` and clipped to ``[-1, 1]``.
        """
        if frame_rate <= 0.0:
            raise ValueError("frame_rate must be greater than zero")

        if opponent_x is not None and opponent_y is not None:
            self._opponent_xy = (float(opponent_x), float(opponent_y))
        if robot_x is not None and robot_y is not None:
            self._robot_xy = (float(robot_x), float(robot_y))

        self._raw_puck = (
            float(puck_x), float(puck_y), float(puck_vx), float(puck_vy), float(frame_rate)
        )
        self._observation = self._calculate_observation()
        self._publish_latest(self._observation)
        return self._observation.copy()

    def _calculate_observation(self) -> np.ndarray:
        """Normalize the latest source-space sample using current bounds."""
        if self._raw_puck is None:
            return self._observation.copy()

        puck_x, puck_y, puck_vx, puck_vy, frame_rate = self._raw_puck
        puck_rectified = self._transform_point(puck_x, puck_y)
        velocity_endpoint = self._transform_point(puck_x + puck_vx, puck_y + puck_vy)
        rectified_vx = velocity_endpoint[0] - puck_rectified[0]
        rectified_vy = velocity_endpoint[1] - puck_rectified[1]

        puck_x_norm = self._normalize(puck_rectified[0], self.x_bounds)
        puck_y_norm = self._normalize(puck_rectified[1], self.y_bounds)
        if self.invert_y:
            puck_y_norm = -puck_y_norm
        opp_x_norm, opp_y_norm = self._normalize_optional(self._opponent_xy)
        robot_x_norm, robot_y_norm = self._normalize_optional(self._robot_xy)

        y_sign = -1.0 if self.invert_y else 1.0
        x_half_range = (self.x_bounds[1] - self.x_bounds[0]) / 2.0
        y_half_range = (self.y_bounds[1] - self.y_bounds[0]) / 2.0
        vx_norm = float(np.clip(rectified_vx * frame_rate * self.sim_dt / x_half_range, -1.0, 1.0))
        vy_norm = float(np.clip(y_sign * rectified_vy * frame_rate * self.sim_dt / y_half_range, -1.0, 1.0))

        return np.asarray(
            [puck_x_norm, puck_y_norm, vx_norm, vy_norm,
             opp_x_norm, opp_y_norm, robot_x_norm, robot_y_norm],
            dtype=np.float32,
        )

    def update_from_engine(
        self,
        result: Mapping[str, Any] | None,
        frame_width: int,
        frame_height: int,
        frame_rate: float,
    ) -> np.ndarray | None:
        """Publish an engine ``last_result`` using frame pixel bounds.

        Returns ``None`` if the engine has not produced a standardized result.
        Worker integration example::

            self.sim_bridge.update_from_engine(
                self.engine.last_result, frame.shape[1], frame.shape[0],
                frame_rate=getattr(self.engine, "ESTIMATED_FPS", 30.0),
            )
        """
        if result is None:
            return None
        if frame_width <= 1 or frame_height <= 1:
            raise ValueError("frame dimensions must both be greater than one pixel")

        if not self._table_bounds_explicit:
            current_size = (frame_width, frame_height)
            if current_size != self._frame_size:
                self.table_x_min = 0.0
                self.table_y_min = 0.0
                self.table_x_max = float(frame_width - 1)
                self.table_y_max = float(frame_height - 1)
                self.x_bounds = (self.table_x_min, self.table_x_max)
                self.y_bounds = (self.table_y_min, self.table_y_max)
                self._frame_size = current_size
        return self.update(
            puck_x=float(result.get("x", 0.0)),
            puck_y=float(result.get("y", 0.0)),
            puck_vx=float(result.get("vx", 0.0)),
            puck_vy=float(result.get("vy", 0.0)),
            frame_rate=frame_rate,
        )

    def _normalize_optional(self, position: tuple[float, float] | None) -> tuple[float, float]:
        if position is None:
            return 0.0, 0.0
        x, y = self._transform_point(*position)
        y_norm = self._normalize(y, self.y_bounds)
        if self.invert_y:
            y_norm = -y_norm
        return self._normalize(x, self.x_bounds), y_norm

    def _transform_point(self, x: float, y: float) -> tuple[float, float]:
        """Transform one camera point into the rectified table coordinate space."""
        if self._perspective_matrix is None:
            return float(x), float(y)
        source = np.asarray([[[x, y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(source, self._perspective_matrix)
        return float(transformed[0, 0, 0]), float(transformed[0, 0, 1])

    def _publish_latest(self, observation: np.ndarray) -> None:
        """Replace any stale queued observation without waiting for a consumer."""
        if self.observations is None:
            return
        try:
            self.observations.put_nowait(observation.copy())
        except Full:
            try:
                self.observations.get_nowait()
            except Empty:
                pass
            try:
                self.observations.put_nowait(observation.copy())
            except Full:
                pass

    def get_observation_vector(self) -> np.ndarray:
        """Return latest telemetry normalized against the current table bounds."""
        if self._raw_puck is not None:
            self._observation = self._calculate_observation()
        return self._observation.copy()

    def predict_puck_trajectory(
        self,
        max_bounces: int = 3,
        time_horizon: float = 2.0,
    ) -> list[tuple[float, float]]:
        """Predict a normalized puck polyline with reflections at side walls.

        The observation stores puck velocity as normalized displacement per
        simulation step. It is converted back to normalized units per second
        with ``sim_dt``. The puck reflects from the top/bottom rails and stops
        when it reaches either goal line, exhausts ``time_horizon``, or reaches
        ``max_bounces``.

        Returns:
            Points in normalized table coordinates, beginning at the current
            puck position and including each wall bounce and final endpoint.
        """
        if max_bounces < 0:
            raise ValueError("max_bounces must be non-negative")
        if time_horizon < 0.0:
            raise ValueError("time_horizon must be non-negative")

        observation = self.get_observation_vector()
        x = float(np.clip(observation[0], -1.0, 1.0))
        y = float(np.clip(observation[1], -1.0, 1.0))
        vx = float(observation[2]) / self.sim_dt
        vy = float(observation[3]) / self.sim_dt
        points = [(x, y)]
        remaining = float(time_horizon)
        bounce_count = 0
        epsilon = 1e-9

        while remaining > epsilon:
            if abs(vx) <= epsilon and abs(vy) <= epsilon:
                break

            time_to_goal = float("inf")
            if vx > epsilon:
                time_to_goal = (1.0 - x) / vx
            elif vx < -epsilon:
                time_to_goal = (-1.0 - x) / vx

            time_to_wall = float("inf")
            wall_y = y
            if vy > epsilon:
                time_to_wall = (1.0 - y) / vy
                wall_y = 1.0
            elif vy < -epsilon:
                time_to_wall = (-1.0 - y) / vy
                wall_y = -1.0

            event_time = min(time_to_goal, time_to_wall)
            if event_time < 0.0:
                event_time = 0.0

            if event_time > remaining or not np.isfinite(event_time):
                x = float(np.clip(x + vx * remaining, -1.0, 1.0))
                y = float(np.clip(y + vy * remaining, -1.0, 1.0))
                points.append((x, y))
                break

            x += vx * event_time
            y += vy * event_time
            remaining -= event_time

            if time_to_goal <= time_to_wall:
                x = -1.0 if vx < 0.0 else 1.0
                points.append((x, float(np.clip(y, -1.0, 1.0))))
                break

            y = wall_y
            points.append((float(np.clip(x, -1.0, 1.0)), y))
            bounce_count += 1
            vy = -vy
            if bounce_count >= max_bounces:
                break

        return points

    def get_latest_queued_observation(self) -> np.ndarray | None:
        """Consume the most recently published queued vector, if one is ready."""
        if self.observations is None:
            return None
        try:
            return self.observations.get_nowait()
        except Empty:
            return None
