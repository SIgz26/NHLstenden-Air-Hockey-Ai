"""Interactive air-hockey match simulation using trained SB3 policies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO, SAC

from ai.air_hockey_env import AirHockeyGymEnv

HUMAN_VS_AI = "human_vs_ai"
AI_VS_AI = "ai_vs_ai"
AI_VS_SCRIPTED = "ai_vs_scripted"
MATCH_MODES = (HUMAN_VS_AI, AI_VS_AI, AI_VS_SCRIPTED)


def mirror_observation_for_right_player(observation: np.ndarray) -> np.ndarray:
    """Mirror an observation so a left-side policy can control the right side."""
    state = np.asarray(observation, dtype=np.float32).reshape(8)
    return np.asarray(
        [
            -state[0], state[1], -state[2], state[3],
            -state[6], state[7], -state[4], state[5],
        ],
        dtype=np.float32,
    )


def mirror_action_for_right_player(action: np.ndarray) -> np.ndarray:
    """Mirror a normalized left-side action back into right-side coordinates."""
    result = np.asarray(action, dtype=np.float32).reshape(2).copy()
    result[0] *= -1.0
    return result


def mouse_position_to_world(
    mouse_x: float,
    mouse_y: float,
    field_left: float,
    field_top: float,
    field_width: float,
    field_height: float,
    field_length_m: float = AirHockeyGymEnv.FIELD_LENGTH_M,
    field_width_m: float = AirHockeyGymEnv.FIELD_WIDTH_M,
    min_x_m: float = -0.925,
    max_x_m: float = -0.15,
    mallet_radius_m: float = AirHockeyGymEnv.MALLET_RADIUS_M,
) -> tuple[float, float]:
    """Convert a screen point into a legal left-side mallet target in meters."""
    if field_width <= 0.0 or field_height <= 0.0:
        raise ValueError("field dimensions must be positive")
    x_fraction = np.clip((mouse_x - field_left) / field_width, 0.0, 1.0)
    y_fraction = np.clip((mouse_y - field_top) / field_height, 0.0, 1.0)
    x_m = (x_fraction - 0.5) * field_length_m
    y_m = (0.5 - y_fraction) * field_width_m
    legal_min_x = -field_length_m / 2.0 + mallet_radius_m
    legal_min_y = -field_width_m / 2.0 + mallet_radius_m
    legal_max_y = field_width_m / 2.0 - mallet_radius_m
    return (
        float(np.clip(x_m, max(min_x_m, legal_min_x), min(max_x_m, -mallet_radius_m))),
        float(np.clip(y_m, legal_min_y, legal_max_y)),
    )


def load_policy(model_path: str | Path) -> Any:
    """Load an SB3 PPO or SAC archive without requiring a separate selector."""
    path = Path(model_path).expanduser()
    try:
        return PPO.load(str(path), device="auto")
    except Exception as ppo_error:
        try:
            return SAC.load(str(path), device="auto")
        except Exception as sac_error:
            raise ValueError(
                f"Could not load model archive '{path}' as PPO or SAC: {sac_error}"
            ) from ppo_error


class MatchSimulator:
    """Step a Gymnasium air-hockey match with human, learned, or scripted play."""

    def __init__(
        self,
        mode: str,
        model_a_path: str | Path | None = None,
        model_b_path: str | Path | None = None,
        seed: int | None = None,
    ) -> None:
        if mode not in MATCH_MODES:
            raise ValueError(f"mode must be one of: {', '.join(MATCH_MODES)}")
        self.mode = mode
        self.model_a_path = Path(model_a_path) if model_a_path else None
        self.model_b_path = Path(model_b_path) if model_b_path else None
        self.model_a = None
        self.model_b = None

        if mode in (AI_VS_AI, AI_VS_SCRIPTED):
            if self.model_a_path is None:
                raise ValueError("Model A is required for this match mode")
            self.model_a = load_policy(self.model_a_path)
        if mode in (HUMAN_VS_AI, AI_VS_AI):
            if self.model_b_path is None:
                raise ValueError("Model B is required for this match mode")
            self.model_b = load_policy(self.model_b_path)

        opponent_type = "complex" if mode == AI_VS_SCRIPTED else "none"
        self.env = AirHockeyGymEnv(seed=seed, opponent_type=opponent_type)
        self.observation, _ = self.env.reset(seed=seed)
        self.mouse_target_m = self.env.robot_position_m.copy()
        self.elapsed_steps = 0

    @staticmethod
    def _predict(model: Any, observation: np.ndarray) -> np.ndarray:
        action, _ = model.predict(observation, deterministic=True)
        return np.clip(np.asarray(action, dtype=np.float32).reshape(2), -1.0, 1.0)

    def set_mouse_target(self, x_m: float, y_m: float) -> None:
        half_width = self.env.FIELD_WIDTH_M / 2.0 - self.env.MALLET_RADIUS_M
        self.mouse_target_m = np.asarray(
            [
                np.clip(x_m, -self.env.FIELD_LENGTH_M / 2.0 + self.env.MALLET_RADIUS_M, -0.15),
                np.clip(y_m, -half_width, half_width),
            ],
            dtype=np.float64,
        )

    def _human_action(self) -> np.ndarray:
        delta = self.mouse_target_m - self.env.robot_position_m
        target_speed = np.clip(delta * 8.0, -self.env.MAX_MALLET_SPEED_MPS, self.env.MAX_MALLET_SPEED_MPS)
        return np.asarray(target_speed / self.env.MAX_MALLET_SPEED_MPS, dtype=np.float32)

    def _advance_model_opponent(self, action: np.ndarray) -> None:
        target_velocity = np.clip(action, -1.0, 1.0) * self.env.MAX_MALLET_SPEED_MPS
        half_length = self.env.FIELD_LENGTH_M / 2.0
        self.env._advance_mallet(
            self.env._opponent_position,
            self.env._opponent_velocity,
            target_velocity,
            (self.env.MALLET_RADIUS_M, half_length - self.env.MALLET_RADIUS_M),
        )

    def step(self):
        """Advance one 120 Hz simulation frame and cache the new state."""
        observation = self.env.observation
        if self.mode == HUMAN_VS_AI:
            left_action = self._human_action()
            right_action = mirror_action_for_right_player(
                self._predict(self.model_b, mirror_observation_for_right_player(observation))
            )
            self._advance_model_opponent(right_action)
        elif self.mode == AI_VS_AI:
            left_action = self._predict(self.model_a, observation)
            right_action = mirror_action_for_right_player(
                self._predict(self.model_b, mirror_observation_for_right_player(observation))
            )
            self._advance_model_opponent(right_action)
        else:
            left_action = self._predict(self.model_a, observation)

        self.observation, reward, terminated, truncated, info = self.env.step(left_action)
        self.elapsed_steps += 1
        return self.observation.copy(), reward, terminated, truncated, info

    def reset_after_goal(self, scored_for: str | None = None) -> None:
        """Reset positions and serve from center while preserving match clock."""
        self.observation, _ = self.env.reset()
        serve_direction = 1.0 if scored_for == "robot" else -1.0
        self.env._puck_position[:] = (0.0, 0.0)
        self.env._puck_velocity[:] = (serve_direction * 0.8, 0.0)
        self.env._previous_robot_puck_distance = float(
            np.linalg.norm(self.env._puck_position - self.env._robot_position)
        )
        self.observation = self.env._make_observation()

    def reset_puck(self) -> None:
        """Place the puck at center and stop it without resetting the score."""
        self.env._puck_position[:] = (0.0, 0.0)
        self.env._puck_velocity[:] = (0.0, 0.0)
        self.env._previous_robot_puck_distance = float(
            np.linalg.norm(self.env._puck_position - self.env._robot_position)
        )
        self.observation = self.env._make_observation()

    @property
    def puck_speed_mps(self) -> float:
        return float(np.linalg.norm(self.env.puck_velocity_mps))

    @property
    def elapsed_seconds(self) -> float:
        return self.elapsed_steps * self.env.DT

    def close(self) -> None:
        self.env.close()
