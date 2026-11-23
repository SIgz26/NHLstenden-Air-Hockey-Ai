"""Smooth closed-loop defender and controlled counter-pusher (no open-loop strike)."""

from __future__ import annotations

import numpy as np

from core.coordinate_contract import PUCK_RADIUS_M as CONTRACT_PUCK_RADIUS_M

from .base_opponent import BaseOpponent


class StrikerDefenderOpponent(BaseOpponent):
    """Defend the goal with discipline and smoothly push slow pucks back."""

    # Aligned with CODESYS GVL_Kin: rFieldYMin=53.7 mm (0.0 m), rFieldYMax=789.5 mm (0.7358 m),
    # and rEdgeMargin=95.0 mm (zero-speed boundary at 0.095 m front and 0.6408 m back).
    DEFEND_LENGTH_M = 0.58
    MIN_LENGTH_M = 0.14
    MIN_PUSH_LENGTH_M = 0.22
    MAX_LENGTH_M = 0.61
    MAX_WIDTH_M = 0.40
    GOAL_GUARD_WIDTH_M = 0.18
    PUCK_RADIUS_M = CONTRACT_PUCK_RADIUS_M

    # Mode thresholds
    CLEAR_START_MIN_LENGTH_M = 0.06
    CLEAR_START_MAX_LENGTH_M = 0.55
    CLEAR_EXIT_LENGTH_M = 0.02
    CLEAR_MAX_PUCK_SPEED_MPS = 0.45
    INCOMING_VEL_THRESHOLD_MPS = 0.06

    # Smooth motion limits (no 0.0 -> 1.0 m/s step jumps)
    DEFEND_SPEED_MPS = 0.70
    ALIGN_SPEED_MPS = 0.50
    PUSH_SPEED_MPS = 0.65
    MAX_ACCEL_MPS2 = 2.4
    CONTROL_DT_S = 0.01

    DEFEND_KP = 3.6
    ALIGN_KP = 3.4
    PUSH_KP = 4.2
    ALIGN_CLEARANCE_M = 0.09

    def __init__(self) -> None:
        self.mode = "DEFEND"
        self._push_blocked = False
        self._push_target: np.ndarray | None = None
        self._prev_velocity = np.zeros(2, dtype=np.float64)
        self._prev_mallet_pos: np.ndarray | None = None

    def get_action(
        self,
        puck_pos: np.ndarray,
        puck_vel: np.ndarray,
        opponent_pos: np.ndarray,
        field_length: float,
        field_width: float,
        mallet_radius: float,
    ) -> np.ndarray:
        del field_length
        puck_pos = np.asarray(puck_pos, dtype=np.float64)
        puck_vel = np.asarray(puck_vel, dtype=np.float64)
        opponent_pos = np.asarray(opponent_pos, dtype=np.float64)

        puck_length, puck_width = float(puck_pos[0]), float(puck_pos[1])
        length_velocity, width_velocity = float(puck_vel[0]), float(puck_vel[1])
        puck_speed = float(np.hypot(length_velocity, width_velocity))

        width_limit = min(self.MAX_WIDTH_M, field_width / 2.0 - mallet_radius)
        puck_width_limit = max(0.05, field_width / 2.0 - self.PUCK_RADIUS_M)
        mallet_length = float(
            np.clip(opponent_pos[0], self.MIN_LENGTH_M, self.MAX_LENGTH_M)
        )
        mallet_width = float(np.clip(opponent_pos[1], -width_limit, width_limit))
        current = np.asarray([mallet_length, mallet_width], dtype=np.float64)

        # Reset velocity ramp if position jumped (e.g. simulator reset)
        if (
            self._prev_mallet_pos is not None
            and float(np.linalg.norm(current - self._prev_mallet_pos)) > 0.15
        ):
            self._prev_velocity = np.zeros(2, dtype=np.float64)
        self._prev_mallet_pos = current.copy()

        # 1. Mode transitions (DEFEND <-> ALIGN <-> PUSH)
        if puck_length < self.CLEAR_EXIT_LENGTH_M:
            self._reset_to_defend(block_repush=False)
        elif self.mode == "PUSH":
            puck_cleared = length_velocity < -0.25 or puck_length < self.CLEAR_EXIT_LENGTH_M
            reached_push_end = (
                self._push_target is not None
                and float(np.linalg.norm(current - self._push_target)) <= 0.025
            )
            passed_puck = mallet_length < puck_length - 0.03
            if puck_cleared or reached_push_end or passed_puck:
                self._reset_to_defend(block_repush=True)
        elif self.mode == "ALIGN":
            if (
                puck_speed > self.CLEAR_MAX_PUCK_SPEED_MPS * 1.35
                or length_velocity > self.INCOMING_VEL_THRESHOLD_MPS * 1.5
            ):
                self._reset_to_defend(block_repush=False)
        elif self.mode == "DEFEND":
            if (
                self._push_blocked
                and puck_speed <= 0.15
                and self.CLEAR_START_MIN_LENGTH_M <= puck_length <= self.CLEAR_START_MAX_LENGTH_M
            ):
                self._push_blocked = False

            if (
                not self._push_blocked
                and self.CLEAR_START_MIN_LENGTH_M <= puck_length <= self.CLEAR_START_MAX_LENGTH_M
                and puck_speed <= self.CLEAR_MAX_PUCK_SPEED_MPS
                and length_velocity <= self.INCOMING_VEL_THRESHOLD_MPS
            ):
                self.mode = "ALIGN"

        # 2. Compute closed-loop position target and speed cap for active mode
        if self.mode == "ALIGN":
            target = self._alignment_target(
                puck_length,
                puck_width,
                mallet_length,
                mallet_width,
                width_limit,
                mallet_radius,
            )
            required_margin = min(0.05, max(0.03, self.MAX_LENGTH_M - puck_length))
            if (
                mallet_length >= puck_length + required_margin
                and abs(mallet_width - target[1]) <= 0.035
            ):
                self.mode = "PUSH"
                self._push_target = self._compute_push_target(
                    puck_length,
                    puck_width,
                    width_limit,
                )
                target = self._push_target
                kp = self.PUSH_KP
                speed_cap = self.PUSH_SPEED_MPS
            else:
                kp = self.ALIGN_KP
                speed_cap = self.ALIGN_SPEED_MPS
        elif self.mode == "PUSH":
            if self._push_target is None:
                self._push_target = self._compute_push_target(
                    puck_length,
                    puck_width,
                    width_limit,
                )
            target = self._push_target
            kp = self.PUSH_KP
            speed_cap = self.PUSH_SPEED_MPS
        else:
            target = self._defend_target(
                puck_length,
                puck_width,
                length_velocity,
                width_velocity,
                width_limit,
                puck_width_limit,
            )
            kp = self.DEFEND_KP
            speed_cap = self.DEFEND_SPEED_MPS

        desired_velocity = (target - current) * kp
        speed = float(np.linalg.norm(desired_velocity))
        if speed > speed_cap:
            desired_velocity *= speed_cap / speed

        ramped_velocity = self._ramp_velocity(desired_velocity)
        self._prev_velocity = ramped_velocity
        return ramped_velocity

    def _reset_to_defend(self, *, block_repush: bool) -> None:
        self.mode = "DEFEND"
        self._push_blocked = block_repush
        self._push_target = None

    def _ramp_velocity(self, desired_velocity: np.ndarray) -> np.ndarray:
        """Gradually ramp velocity from 0.0 so the H-bot accelerates smoothly without jerk."""
        max_delta = self.MAX_ACCEL_MPS2 * self.CONTROL_DT_S
        delta = desired_velocity - self._prev_velocity
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm > max_delta > 0.0:
            delta *= max_delta / delta_norm
        velocity = self._prev_velocity + delta

        # Respect asymmetric axis caps
        vx = float(np.clip(velocity[0], -self.DEFEND_SPEED_MPS, self.DEFEND_SPEED_MPS))
        vy = float(np.clip(velocity[1], -self.DEFEND_SPEED_MPS, self.DEFEND_SPEED_MPS))
        return np.asarray([vx, vy], dtype=np.float64)

    def _compute_push_target(
        self,
        puck_length: float,
        puck_width: float,
        width_limit: float,
    ) -> np.ndarray:
        """Pick a smooth follow-through position target behind/through the puck."""
        near_side_wall = abs(puck_width) > max(0.05, width_limit - 0.08)
        target_length = float(
            np.clip(puck_length - 0.14, self.MIN_PUSH_LENGTH_M, self.MAX_LENGTH_M)
        )
        if near_side_wall:
            target_width = float(np.clip(puck_width, -width_limit, width_limit))
            return np.asarray([target_length, target_width], dtype=np.float64)

        goal = np.asarray([-0.97, 0.0], dtype=np.float64)
        puck = np.asarray([puck_length, puck_width], dtype=np.float64)
        shot_dir = goal - puck
        if abs(float(shot_dir[0])) > 1e-6:
            slope = float(shot_dir[1] / shot_dir[0])
            target_width = puck_width + slope * (target_length - puck_length)
        else:
            target_width = puck_width
        target_width = float(np.clip(target_width, -width_limit, width_limit))
        return np.asarray([target_length, target_width], dtype=np.float64)

    def _alignment_target(
        self,
        puck_length: float,
        puck_width: float,
        mallet_length: float,
        mallet_width: float,
        width_limit: float,
        mallet_radius: float,
    ) -> np.ndarray:
        near_side_wall = abs(puck_width) > max(0.05, width_limit - 0.08)
        available_back_room = max(0.04, self.MAX_LENGTH_M - puck_length)
        clearance_behind = min(self.ALIGN_CLEARANCE_M, available_back_room)

        if near_side_wall:
            behind_puck = np.asarray(
                [
                    np.clip(puck_length + clearance_behind, self.MIN_LENGTH_M, self.MAX_LENGTH_M),
                    np.clip(puck_width, -width_limit, width_limit),
                ],
                dtype=np.float64,
            )
        else:
            goal = np.asarray([-0.97, 0.0], dtype=np.float64)
            puck = np.asarray([puck_length, puck_width], dtype=np.float64)
            shot_direction = goal - puck
            shot_direction /= max(float(np.linalg.norm(shot_direction)), 1e-6)
            behind_puck = puck - shot_direction * clearance_behind
            behind_puck[0] = float(
                np.clip(behind_puck[0], self.MIN_LENGTH_M, self.MAX_LENGTH_M)
            )
            behind_puck[1] = float(np.clip(behind_puck[1], -width_limit, width_limit))

        # Route safely around the side if mallet is currently in front of the puck
        if mallet_length < puck_length + 0.05:
            if mallet_length > puck_length:
                return behind_puck
            side_clearance = mallet_radius + self.PUCK_RADIUS_M + 0.03
            side = float(np.sign(mallet_width - puck_width))
            if side == 0.0:
                side = 1.0 if (width_limit - puck_width) >= (width_limit + puck_width) else -1.0
            staging_width = float(
                np.clip(puck_width + side * side_clearance, -width_limit, width_limit)
            )
            staging_length = float(
                np.clip(
                    puck_length - side_clearance,
                    self.MIN_LENGTH_M,
                    self.MAX_LENGTH_M,
                )
            )
            at_staging_point = (
                abs(mallet_length - staging_length) < 0.035
                and abs(mallet_width - staging_width) < 0.035
            )
            if not at_staging_point:
                return np.asarray([staging_length, staging_width], dtype=np.float64)
            return np.asarray([behind_puck[0], staging_width], dtype=np.float64)

        return behind_puck

    def _defend_target(
        self,
        puck_length: float,
        puck_width: float,
        length_velocity: float,
        width_velocity: float,
        width_limit: float,
        puck_width_limit: float,
    ) -> np.ndarray:
        guard_limit = min(self.GOAL_GUARD_WIDTH_M, width_limit)
        if puck_length < 0.0 and length_velocity <= 0.05:
            target_width = 0.0
        else:
            target_width = puck_width
            if length_velocity > 0.05:
                time_to_line = (self.DEFEND_LENGTH_M - puck_length) / length_velocity
                if time_to_line >= 0.0:
                    target_width += width_velocity * time_to_line
                    if puck_width_limit > 0.0:
                        while abs(target_width) > puck_width_limit:
                            if target_width > puck_width_limit:
                                target_width = 2.0 * puck_width_limit - target_width
                            else:
                                target_width = -2.0 * puck_width_limit - target_width
            target_width = float(np.clip(target_width, -guard_limit, guard_limit))
        return np.asarray([self.DEFEND_LENGTH_M, target_width], dtype=np.float64)