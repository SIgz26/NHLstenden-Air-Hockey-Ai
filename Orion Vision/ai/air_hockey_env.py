"""Gymnasium air-hockey simulation with SI-unit rigid-body dynamics."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ai.opponents.attacker import AttackerOpponent, ComplexOpponent


class StaticOpponent:
    """Simple non-moving opponent placeholder for curriculum stages."""

    def get_action(
        self,
        puck_pos: np.ndarray,
        puck_vel: np.ndarray,
        opponent_pos: np.ndarray,
        field_length: float,
        field_width: float,
        mallet_radius: float,
    ) -> np.ndarray:
        return np.zeros(2, dtype=np.float64)


class AirHockeyGymEnv(gym.Env[np.ndarray, np.ndarray]):
    """Air-hockey physics with an 8D normalized SimBridge observation.

    State order is ``[puck_x, puck_y, puck_vx, puck_vy, opponent_x,
    opponent_y, robot_x, robot_y]``. Internally, positions use meters and
    velocities use m/s. Observation positions are normalized by field
    half-dimensions and puck speeds by ``MAX_PUCK_SPEED_MPS``. Actions request
    normalized target velocity for the robot mallet.
    """

    REWARD_MODES = ("aggressive", "balanced", "defensive")
    metadata = {"render_modes": []}

    FIELD_LENGTH_M = 1.95
    FIELD_WIDTH_M = 0.98
    PUCK_RADIUS_M = 0.031
    PUCK_MASS_KG = 0.020
    MALLET_RADIUS_M = 0.048
    MALLET_MASS_KG = 0.110
    RESTITUTION = 0.85
    WALL_DAMPING = 0.90
    MAX_MALLET_ACCELERATION_MPS2 = 12.0
    MAX_MALLET_SPEED_MPS = 2.5
    MAX_PUCK_SPEED_MPS = 5.0
    PUCK_AIR_FRICTION = 0.005
    GOAL_WIDTH_M = 0.30
    DT = 1.0 / 120.0
    CONTACT_SLOP_M = 1e-5

    VALID_OPPONENT_TYPES = ("none", "static", "complex")

    def __init__(
        self,
        max_episode_steps: int = 1000,
        seed: int | None = None,
        reward_mode: str = "aggressive",
        opponent_type: str = "complex",
    ) -> None:
        super().__init__()
        if max_episode_steps < 1:
            raise ValueError("max_episode_steps must be at least 1")
        self.reward_mode = self._resolve_reward_mode(reward_mode)
        self.opponent_type = self._resolve_opponent_type(opponent_type)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.max_episode_steps = int(max_episode_steps)
        self._rng = np.random.default_rng(seed)
        self._puck_position = np.zeros(2, dtype=np.float64)
        self._puck_velocity = np.zeros(2, dtype=np.float64)
        self._robot_position = np.zeros(2, dtype=np.float64)
        self._robot_velocity = np.zeros(2, dtype=np.float64)
        self._opponent_position = np.zeros(2, dtype=np.float64)
        self._opponent_velocity = np.zeros(2, dtype=np.float64)
        self._steps = 0
        self._episode_return = 0.0
        self._previous_robot_puck_distance = 0.0
        self._observation = np.zeros(8, dtype=np.float32)
        self.opponent = self._build_opponent(self.opponent_type)

    @staticmethod
    def _resolve_reward_mode(reward_mode: str) -> str:
        mode = str(reward_mode).strip().lower()
        if mode not in AirHockeyGymEnv.REWARD_MODES:
            valid_modes = ", ".join(AirHockeyGymEnv.REWARD_MODES)
            raise ValueError(f"reward_mode must be one of: {valid_modes}")
        return mode

    @staticmethod
    def _resolve_opponent_type(opponent_type: str) -> str:
        value = str(opponent_type).strip().lower()
        if value not in AirHockeyGymEnv.VALID_OPPONENT_TYPES:
            valid_types = ", ".join(AirHockeyGymEnv.VALID_OPPONENT_TYPES)
            raise ValueError(f"opponent_type must be one of: {valid_types}")
        return value

    @staticmethod
    def _build_opponent(opponent_type: str):
        if opponent_type == "none":
            return None
        if opponent_type == "static":
            return StaticOpponent()
        return ComplexOpponent()

    @property
    def observation(self) -> np.ndarray:
        """Return a copy of the current normalized observation."""
        return self._observation.copy()

    @property
    def puck_position_m(self) -> np.ndarray:
        """Current puck center in meters."""
        return self._puck_position.copy()

    @property
    def puck_velocity_mps(self) -> np.ndarray:
        """Current puck velocity in m/s."""
        return self._puck_velocity.copy()

    @property
    def robot_position_m(self) -> np.ndarray:
        """Current robot mallet center in meters."""
        return self._robot_position.copy()

    @property
    def robot_velocity_mps(self) -> np.ndarray:
        """Current robot mallet velocity in m/s."""
        return self._robot_velocity.copy()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        half_length = self.FIELD_LENGTH_M / 2.0
        self._puck_position[:] = (
            self._rng.uniform(0.15, max(0.25, half_length * 0.70)),
            self._rng.uniform(-0.22, 0.22),
        )
        speed = self._rng.uniform(0.35, 1.2)
        angle = self._rng.uniform(-np.pi / 2.0, np.pi / 2.0)
        self._puck_velocity[:] = (
            -speed * np.cos(angle),
            speed * np.sin(angle),
        )
        self._robot_position[:] = (-half_length * 0.72, self._rng.uniform(-0.15, 0.15))
        self._robot_velocity[:] = 0.0
        if self.opponent_type == "static":
            self._opponent_position[:] = (half_length * 0.72, 0.0)
        else:
            self._opponent_position[:] = (half_length * 0.72, self._rng.uniform(-0.15, 0.15))
        self._opponent_velocity[:] = 0.0
        self._steps = 0
        self._episode_return = 0.0
        self._previous_robot_puck_distance = float(
            np.linalg.norm(self._puck_position - self._robot_position)
        )
        self._observation = self._make_observation()
        return self._observation.copy(), {}

    def step(self, action: np.ndarray):
        target_velocity = np.asarray(action, dtype=np.float64).reshape(2)
        if not np.isfinite(target_velocity).all():
            raise ValueError("action must contain finite values")
        target_velocity = np.clip(target_velocity, -1.0, 1.0) * self.MAX_MALLET_SPEED_MPS
        
        self._steps += 1
        previous_puck_x = float(self._puck_position[0])
        previous_robot_puck_distance = self._previous_robot_puck_distance
        half_length = self.FIELD_LENGTH_M / 2.0
        half_width = self.FIELD_WIDTH_M / 2.0
        max_y = half_width - self.MALLET_RADIUS_M

        # 1. Verplaats mallets
        self._advance_mallet(self._robot_position, self._robot_velocity, target_velocity,
                             (-half_length + self.MALLET_RADIUS_M, -0.15))

        if self.opponent_type == "complex" and self.opponent is not None:
            desired_opponent_vel = self.opponent.get_action(
                self._puck_position, self._puck_velocity, self._opponent_position,
                self.FIELD_LENGTH_M, self.FIELD_WIDTH_M, self.MALLET_RADIUS_M
            )
            self._advance_mallet(self._opponent_position, self._opponent_velocity, desired_opponent_vel,
                                 (self.MALLET_RADIUS_M, half_length - self.MALLET_RADIUS_M))
        elif self.opponent_type == "static":
            self._opponent_position[:] = (half_length * 0.72, 0.0)
            self._opponent_velocity[:] = 0.0

        # 2. Physics update
        self._puck_position += self._puck_velocity * self.DT
        self._resolve_puck_walls()
        self._puck_velocity *= 1.0 - self.PUCK_AIR_FRICTION * self.DT

        # Bewaar relatieve snelheid voor impact berekening
        rel_vel_before = np.linalg.norm(self._robot_velocity - self._puck_velocity)

        robot_contact = self._resolve_mallet_puck_collision(self._robot_position, self._robot_velocity)
        opponent_contact = self._resolve_mallet_puck_collision(self._opponent_position, self._opponent_velocity)
        self._resolve_puck_walls()

        # 3. Status updates
        goal_opening_half = self.GOAL_WIDTH_M / 2.0 - self.PUCK_RADIUS_M
        scored_right = (self._puck_position[0] >= half_length - self.PUCK_RADIUS_M and abs(self._puck_position[1]) <= goal_opening_half)
        scored_left = (self._puck_position[0] <= -half_length + self.PUCK_RADIUS_M and abs(self._puck_position[1]) <= goal_opening_half)
        current_robot_puck_distance = float(np.linalg.norm(self._puck_position - self._robot_position))

        # ----------------------------------------------------------------------
        # VERBETERDE REWARD PROFIELEN (Sessie 03)
        # ----------------------------------------------------------------------
        reward_profile = {
            "aggressive": {
                "base": -0.01,
                "clear_bonus": 2.0,
                "robot_contact": 1.8,
                "goal_for": 25.0,
                "goal_against": -25.0,
                "wall_penalty": 0.05,
                "defensive_alignment": 0.08,
                "opponent_contact_penalty": 0.10,
            },
            "balanced": {
                "base": -0.01,
                "clear_bonus": 2.0,
                "robot_contact": 1.6,
                "goal_for": 25.0,
                "goal_against": -25.0,
                "wall_penalty": 0.05,
                "defensive_alignment": 0.06,
                "opponent_contact_penalty": 0.10,
            },
            "defensive": {
                "base": -0.01,
                "clear_bonus": 2.0,
                "robot_contact": 1.4,
                "goal_for": 25.0,
                "goal_against": -25.0,
                "wall_penalty": 0.05,
                "defensive_alignment": 0.10,
                "opponent_contact_penalty": 0.10,
            },
        }[self.reward_mode]

        reward = float(reward_profile["base"])

        active_clear = robot_contact or self._puck_velocity[0] > 0.5
        if previous_puck_x < 0.0 and self._puck_position[0] >= 0.0 and active_clear:
            reward += reward_profile["clear_bonus"]

        if self._puck_position[0] < 0.0:
            y_diff = abs(float(self._robot_position[1] - self._puck_position[1]))
            alignment = max(0.0, 1.0 - (y_diff / half_width))
            reward += reward_profile["defensive_alignment"] * alignment

        attack_distance = float(np.linalg.norm(self._puck_position - self._robot_position))
        if (
            self._puck_position[0] < 0.0
            and self._robot_position[0] > self._puck_position[0]
            and self._puck_velocity[0] > 0.8
            and attack_distance < 0.40
            and target_velocity[0] > 0.5
        ):
            reward += 2.5 * min(1.0, self._puck_velocity[0] / 1.5)

        current_y_abs = abs(float(self._robot_position[1]))
        wall_threshold = 0.05
        if current_y_abs > (max_y - wall_threshold):
            proximity = (current_y_abs - (max_y - wall_threshold)) / wall_threshold
            reward -= reward_profile["wall_penalty"] * max(0.0, proximity)

        if robot_contact and rel_vel_before > 0.8:
            impact_strength = 0.5 + min(1.0, rel_vel_before / 1.5)
            reward += reward_profile["robot_contact"] * impact_strength

        if self._puck_position[0] >= 0.0 and self._robot_position[0] > -0.35:
            reward -= 0.02

        near_back_wall = abs(float(self._robot_position[0] + half_length)) < 0.05
        near_side_wall = abs(float(self._robot_position[1])) > (max_y - 0.05)
        if (near_back_wall or near_side_wall) and np.linalg.norm(self._robot_velocity) < 0.05:
            if previous_robot_puck_distance <= current_robot_puck_distance:
                reward -= 0.02

        if opponent_contact:
            reward -= reward_profile["opponent_contact_penalty"]

        if scored_right:
            reward += reward_profile["goal_for"]
        elif scored_left:
            reward += reward_profile["goal_against"]

        # Afsluiting step
        self._previous_robot_puck_distance = current_robot_puck_distance
        self._episode_return += reward
        self._observation = self._make_observation()
        
        terminated = bool(scored_right or scored_left)
        truncated = self._steps >= self.max_episode_steps

        info = {
            "goal_scored": bool(scored_right or scored_left),
            "scored_for": "robot" if scored_right else "opponent" if scored_left else None,
            "robot_contact": robot_contact,
            "opponent_contact": opponent_contact,
            "episode_return": self._episode_return,
        }

        return self._observation.copy(), float(reward), terminated, truncated, info




    def _advance_mallet(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        target_velocity: np.ndarray,
        x_limits: tuple[float, float],
    ) -> None:
        """Apply acceleration-limited motor response and integrate mallet."""
        velocity_delta = target_velocity - velocity
        delta_speed = float(np.linalg.norm(velocity_delta))
        max_delta_speed = self.MAX_MALLET_ACCELERATION_MPS2 * self.DT
        if delta_speed > max_delta_speed:
            velocity_delta *= max_delta_speed / delta_speed
        velocity += velocity_delta
        speed = float(np.linalg.norm(velocity))
        if speed > self.MAX_MALLET_SPEED_MPS:
            velocity *= self.MAX_MALLET_SPEED_MPS / speed

        position += velocity * self.DT
        position[0] = float(np.clip(position[0], *x_limits))
        y_limit = self.FIELD_WIDTH_M / 2.0 - self.MALLET_RADIUS_M
        if position[1] < -y_limit or position[1] > y_limit:
            position[1] = float(np.clip(position[1], -y_limit, y_limit))
            velocity[1] = 0.0

    def _resolve_puck_walls(self) -> None:
        """Reflect the puck center at radius-adjusted wall positions."""
        y_limit = self.FIELD_WIDTH_M / 2.0 - self.PUCK_RADIUS_M
        if self._puck_position[1] > y_limit:
            self._puck_position[1] = y_limit
            self._puck_velocity[1] = -abs(self._puck_velocity[1]) * self.WALL_DAMPING
        elif self._puck_position[1] < -y_limit:
            self._puck_position[1] = -y_limit
            self._puck_velocity[1] = abs(self._puck_velocity[1]) * self.WALL_DAMPING

        half_length = self.FIELD_LENGTH_M / 2.0
        x_limit = half_length - self.PUCK_RADIUS_M
        inside_goal = abs(float(self._puck_position[1])) <= self.GOAL_WIDTH_M / 2.0 - self.PUCK_RADIUS_M
        if not inside_goal:
            if self._puck_position[0] > x_limit:
                self._puck_position[0] = x_limit
                self._puck_velocity[0] = -abs(self._puck_velocity[0]) * self.WALL_DAMPING
            elif self._puck_position[0] < -x_limit:
                self._puck_position[0] = -x_limit
                self._puck_velocity[0] = abs(self._puck_velocity[0]) * self.WALL_DAMPING

    def _resolve_mallet_puck_collision(
        self,
        mallet_position: np.ndarray,
        mallet_velocity: np.ndarray,
    ) -> bool:
        """Resolve an elastic circle impact and remove puck/mallet overlap."""
        displacement = self._puck_position - mallet_position
        distance = float(np.linalg.norm(displacement))
        combined_radius = self.PUCK_RADIUS_M + self.MALLET_RADIUS_M
        if distance > combined_radius:
            return False

        if distance > 1e-12:
            normal = displacement / distance
        else:
            relative = self._puck_velocity - mallet_velocity
            relative_speed = float(np.linalg.norm(relative))
            normal = relative / relative_speed if relative_speed > 1e-12 else np.asarray((1.0, 0.0))

        overlap = combined_radius - distance
        if overlap > 0.0:
            self._puck_position += normal * (overlap + self.CONTACT_SLOP_M)

        relative_velocity = self._puck_velocity - mallet_velocity
        normal_speed = float(np.dot(relative_velocity, normal))
        if normal_speed >= 0.0:
            return True

        inverse_mass_sum = 1.0 / self.PUCK_MASS_KG + 1.0 / self.MALLET_MASS_KG
        impulse_magnitude = -(1.0 + self.RESTITUTION) * normal_speed / inverse_mass_sum
        impulse = impulse_magnitude * normal
        self._puck_velocity += impulse / self.PUCK_MASS_KG
        mallet_velocity -= impulse / self.MALLET_MASS_KG
        puck_speed = float(np.linalg.norm(self._puck_velocity))
        if puck_speed > self.MAX_PUCK_SPEED_MPS:
            self._puck_velocity *= self.MAX_PUCK_SPEED_MPS / puck_speed
        return True

    def _make_observation(self) -> np.ndarray:
        """Normalize SI state to the SimBridge-compatible [-1, 1] vector."""
        half_length = self.FIELD_LENGTH_M / 2.0
        half_width = self.FIELD_WIDTH_M / 2.0
        observation = np.asarray(
            [
                self._puck_position[0] / half_length,
                self._puck_position[1] / half_width,
                self._puck_velocity[0] / self.MAX_PUCK_SPEED_MPS,
                self._puck_velocity[1] / self.MAX_PUCK_SPEED_MPS,
                self._opponent_position[0] / half_length,
                self._opponent_position[1] / half_width,
                self._robot_position[0] / half_length,
                self._robot_position[1] / half_width,
            ],
            dtype=np.float32,
        )
        return np.clip(observation, -1.0, 1.0)