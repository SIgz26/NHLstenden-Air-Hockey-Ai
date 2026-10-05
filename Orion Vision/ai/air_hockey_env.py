"""Gymnasium air-hockey simulation with SI-unit rigid-body dynamics."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ai.opponents.attacker import AttackerOpponent, ComplexOpponent


class AirHockeyGymEnv(gym.Env[np.ndarray, np.ndarray]):
    """Air-hockey physics with an 8D normalized SimBridge observation.

    State order is ``[puck_x, puck_y, puck_vx, puck_vy, opponent_x,
    opponent_y, robot_x, robot_y]``. Internally, positions use meters and
    velocities use m/s. Observation positions are normalized by field
    half-dimensions and puck speeds by ``MAX_PUCK_SPEED_MPS``. Actions request
    normalized target velocity for the robot mallet.
    """

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

    def __init__(self, max_episode_steps: int = 1000, seed: int | None = None) -> None:
        super().__init__()
        if max_episode_steps < 1:
            raise ValueError("max_episode_steps must be at least 1")
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
        self._observation = np.zeros(8, dtype=np.float32)

        # Maak een instantie aan van AttackerOpponent
        self.opponent = ComplexOpponent()

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
            self._rng.uniform(-0.15, 0.15),
            self._rng.uniform(-0.20, 0.20),
        )
        self._puck_velocity[:] = self._rng.uniform(-0.8, 0.8, size=2)
        self._robot_position[:] = (-half_length * 0.72, self._rng.uniform(-0.15, 0.15))
        self._robot_velocity[:] = 0.0
        self._opponent_position[:] = (half_length * 0.72, self._rng.uniform(-0.15, 0.15))
        self._opponent_velocity[:] = 0.0
        self._steps = 0
        self._episode_return = 0.0
        self._observation = self._make_observation()
        return self._observation.copy(), {}

    def step(self, action: np.ndarray):
        target_velocity = np.asarray(action, dtype=np.float64).reshape(2)
        if not np.isfinite(target_velocity).all():
            raise ValueError("action must contain finite values")
        target_velocity = np.clip(target_velocity, -1.0, 1.0) * self.MAX_MALLET_SPEED_MPS
        
        self._steps += 1
        previous_puck_x = float(self._puck_position[0])
        half_length = self.FIELD_LENGTH_M / 2.0
        half_width = self.FIELD_WIDTH_M / 2.0
        y_limit_mallet = half_width - self.MALLET_RADIUS_M

        # 1. Verplaats mallets
        self._advance_mallet(
            self._robot_position,
            self._robot_velocity,
            target_velocity,
            (-half_length + self.MALLET_RADIUS_M, -self.MALLET_RADIUS_M),
        )

        desired_opponent_vel = self.opponent.get_action(
            self._puck_position,
            self._puck_velocity,
            self._opponent_position,
            self.FIELD_LENGTH_M,
            self.FIELD_WIDTH_M,
            self.MALLET_RADIUS_M,
        )
        self._advance_mallet(
            self._opponent_position,
            self._opponent_velocity,
            desired_opponent_vel,
            (self.MALLET_RADIUS_M, half_length - self.MALLET_RADIUS_M),
        )

        # 2. Physics update
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

        # 3. Goal detectie
        goal_opening_half = self.GOAL_WIDTH_M / 2.0 - self.PUCK_RADIUS_M
        scored_right = (
            self._puck_position[0] >= half_length - self.PUCK_RADIUS_M
            and abs(float(self._puck_position[1])) <= goal_opening_half
        )
        scored_left = (
            self._puck_position[0] <= -half_length + self.PUCK_RADIUS_M
            and abs(float(self._puck_position[1])) <= goal_opening_half
        )

        # ----------------------------------------------------------------------
        # REWARD LOGICA (Sessie 03 Verbeterd)
        # ----------------------------------------------------------------------
        reward = -0.005  # Basis time penalty om actie te stimuleren

        # --- 1. IMPACT REWARD (vervangt simpele contact reward) ---
        if robot_contact:
            # Bereken relatieve snelheid op het moment van impact
            relative_vel = np.linalg.norm(self._robot_velocity - self._puck_velocity)
            # Beloon alleen echte slagen (voorkomt 375 schoten/min 'contact farming')
            if relative_vel > 0.5: 
                reward += 0.5 * relative_vel 
            else:
                reward += 0.05 

        if opponent_contact:
            reward -= 0.1

        # --- 2. ANTI-CAMPING & WALL PENALTY ---
        # Straf voor het 'plakken' aan de achterwand (x_min ≈ -0.927)
        if self._robot_position[0] < (-half_length + self.MALLET_RADIUS_M + 0.05):
            puck_dist_x = abs(self._puck_position[0] - self._robot_position[0])
            if puck_dist_x > 0.4:
                reward -= 0.01  # Straf voor te passief wachten in de hoek
        
        # Straf voor te dicht bij de zijmuren (Wall Proximity Penalty uit logboek)
        dist_to_wall = half_width - abs(self._robot_position[1])
        if dist_to_wall < 0.05:
            proximity = (0.05 - dist_to_wall) / 0.05
            reward -= 0.05 * proximity

        # --- 3. DYNAMISCHE CLEARANCE & PROGRESSIE ---
        # Beloon het verplaatsen van de puck van eigen naar vijandelijke helft
        if previous_puck_x < 0 and self._puck_position[0] >= 0:
            reward += 1.0 
        
        # Progressie reward (alleen als de puck niet al bij de tegenstander ligt)
        if self._puck_position[0] < 0.5:
            reward += 0.02 * (self._puck_position[0] - previous_puck_x)

        # --- 4. DEFENSIVE ALIGNMENT BONUS ---
        if self._puck_position[0] < 0.0:  # Alleen als puck op onze helft is
            # Bonus voor het staan tussen de puck en het doel
            lateral_error = abs(self._robot_position[1] - self._puck_position[1])
            reward += 0.03 * max(0.0, 1.0 - lateral_error / half_width)
            
            # Extra bonus voor positie voor het eigen doel (Defensive zone)
            if self._robot_position[0] < -half_length / 2.0:
                if abs(self._robot_position[1]) < self.GOAL_WIDTH_M / 2.0:
                    reward += 0.02

        # --- 5. SPARSE GOAL REWARDS ---
        if scored_right:
            reward += 15.0
        elif scored_left:
            reward -= 15.0

        # ----------------------------------------------------------------------
        # EINDE REWARD LOGICA
        # ----------------------------------------------------------------------

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