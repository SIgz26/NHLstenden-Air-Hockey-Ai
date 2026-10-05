"""Gymnasium air-hockey simulation with SI-unit rigid-body dynamics."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


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

    def __init__(
        self,
        max_episode_steps: int = 1000,
        seed: int | None = None,
        reward_mode: str = "aggressive",
    ) -> None:
        super().__init__()
        if max_episode_steps < 1:
            raise ValueError("max_episode_steps must be at least 1")
        self.reward_mode = self._resolve_reward_mode(reward_mode)
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

    @staticmethod
    def _resolve_reward_mode(reward_mode: str) -> str:
        mode = str(reward_mode).strip().lower()
        if mode not in AirHockeyGymEnv.REWARD_MODES:
            valid_modes = ", ".join(AirHockeyGymEnv.REWARD_MODES)
            raise ValueError(f"reward_mode must be one of: {valid_modes}")
        return mode

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
        self._advance_mallet(
            self._robot_position,
            self._robot_velocity,
            target_velocity,
            (-half_length + self.MALLET_RADIUS_M, -self.MALLET_RADIUS_M),
        )
        self._advance_opponent()

        self._puck_position += self._puck_velocity * self.DT
        self._resolve_puck_walls()
        self._puck_velocity *= 1.0 - self.PUCK_AIR_FRICTION * self.DT

        robot_contact = self._resolve_mallet_puck_collision(
            self._robot_position, self._robot_velocity
        )
        opponent_contact = self._resolve_mallet_puck_collision(
            self._opponent_position, self._opponent_velocity
        )
        self._resolve_puck_walls()

        goal_opening_half = self.GOAL_WIDTH_M / 2.0 - self.PUCK_RADIUS_M
        scored_right = (
            self._puck_position[0] >= half_length - self.PUCK_RADIUS_M
            and abs(float(self._puck_position[1])) <= goal_opening_half
        )
        scored_left = (
            self._puck_position[0] <= -half_length + self.PUCK_RADIUS_M
            and abs(float(self._puck_position[1])) <= goal_opening_half
        )
        current_robot_puck_distance = float(np.linalg.norm(self._puck_position - self._robot_position))

        reward_profile = {
            "aggressive": {
                "base": -0.001,
                "puck_progress": 0.12,
                "attack_distance": 0.18,
                "stuck_penalty": 0.02,
                "robot_contact": 3.4,
                "impact_scale": 1.0,
                "opponent_contact_penalty": 0.08,
                "alignment_bonus": 0.04,
                "goal_for": 10.0,
                "goal_against": 3.0,
            },
            "balanced": {
                "base": -0.001,
                "puck_progress": 0.10,
                "attack_distance": 0.15,
                "stuck_penalty": 0.02,
                "robot_contact": 3.0,
                "impact_scale": 1.0,
                "opponent_contact_penalty": 0.08,
                "alignment_bonus": 0.05,
                "goal_for": 10.0,
                "goal_against": 3.0,
            },
            "defensive": {
                "base": -0.001,
                "puck_progress": 0.08,
                "attack_distance": 0.10,
                "stuck_penalty": 0.04,
                "robot_contact": 2.5,
                "impact_scale": 0.9,
                "opponent_contact_penalty": 0.06,
                "alignment_bonus": 0.08,
                "goal_for": 9.0,
                "goal_against": 2.5,
            },
        }[self.reward_mode]

        reward = reward_profile["base"] + reward_profile["puck_progress"] * (
            float(self._puck_position[0]) - previous_puck_x
        )

        if self._puck_position[0] < 0.0:
            reward += reward_profile["attack_distance"] * max(
                0.0, previous_robot_puck_distance - current_robot_puck_distance
            )
        # --- NIEUW / AANGEPAST: 1. Straf voor te dicht bij de zijkanten (Muren) ---
        # y_limit voor de mallet is FIELD_WIDTH_M / 2 - MALLET_RADIUS_M
        max_y = self.FIELD_WIDTH_M / 2.0 - self.MALLET_RADIUS_M
        wall_distance_threshold = 0.05  # Binnen 5 cm van de muur
        
        current_y_abs = abs(float(self._robot_position[1]))
        if current_y_abs > (max_y - wall_distance_threshold):
            # Proportionele straf: hoe dichter bij de muur, hoe hoger de straf
            proximity = (current_y_abs - (max_y - wall_distance_threshold)) / wall_distance_threshold
            reward -= 0.05 * proximity  # Instelbare factor (bijv. -0.05 max per stap)

        # --- NIEUW: 2. Pluspunt voor verdedigende positie voor het eigen doel ---
        # Eigen doel ligt aan de linkerkant (x < 0)
        # Check of robot op eigen helft staat én voor de opening van het doel (GOAL_WIDTH_M)
        in_defensive_x = -half_length < self._robot_position[0] < -half_length * 0.5
        in_goal_y_range = abs(self._robot_position[1]) <= (self.GOAL_WIDTH_M / 2.0)

        if in_defensive_x and in_goal_y_range:
            reward += 0.02  # Kleine constante bonus voor afdekken van het doel
            if self._puck_position[0] < 0.0:
                y_diff = abs(self._robot_position[1] - self._puck_position[1])
                alignment = max(0.0, 1.0 - (y_diff / (self.FIELD_WIDTH_M / 2.0)))
                reward += 0.03 * alignment

        near_back_wall = abs(float(self._robot_position[0] + half_length)) < 0.05
        near_side_wall = abs(float(self._robot_position[1])) > (self.FIELD_WIDTH_M / 2.0 - 0.05)
        is_stuck_in_wall = (
            (near_back_wall or near_side_wall)
            and float(np.linalg.norm(self._robot_velocity)) < 0.05
            and previous_robot_puck_distance - current_robot_puck_distance <= 0.0
        )

        if is_stuck_in_wall:
            reward -= reward_profile["stuck_penalty"]

        if robot_contact:
            reward += reward_profile["robot_contact"]
            if self._puck_velocity[0] > 0.0:
                impact_bonus = float(np.clip(1.0 + 1.0 * self._puck_velocity[0], 1.0, 4.0))
                reward += impact_bonus * reward_profile["impact_scale"]

        if opponent_contact:
            reward -= reward_profile["opponent_contact_penalty"]

        if self._puck_position[0] < 0.0:
            lateral_error = abs(float(self._robot_position[1] - self._puck_position[1]))
            reward += reward_profile["alignment_bonus"] * max(
                0.0, 1.0 - lateral_error / (self.FIELD_WIDTH_M / 2.0)
            )

        if scored_right:
            reward += reward_profile["goal_for"]
        elif scored_left:
            reward -= reward_profile["goal_against"]

        self._previous_robot_puck_distance = current_robot_puck_distance
        terminated = bool(scored_right or scored_left)
        truncated = self._steps >= self.max_episode_steps
        self._episode_return += reward
        self._observation = self._make_observation()
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

    def _advance_opponent(self) -> None:
        """Simple defender tracks puck Y with a bounded motor response."""
        half_length = self.FIELD_LENGTH_M / 2.0
        target = np.asarray(
            [half_length * 0.72, float(np.clip(self._puck_position[1], -0.28, 0.28))],
            dtype=np.float64,
        )
        desired_velocity = np.clip((target - self._opponent_position) / 0.25, -1.5, 1.5)
        self._advance_mallet(
            self._opponent_position,
            self._opponent_velocity,
            desired_velocity,
            (self.MALLET_RADIUS_M, half_length - self.MALLET_RADIUS_M),
        )

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
