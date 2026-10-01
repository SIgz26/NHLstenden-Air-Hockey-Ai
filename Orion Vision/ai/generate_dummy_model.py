"""Generate an untrained Stable-Baselines3 SAC archive for smoke testing."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import SAC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "dummy_sac_model.zip"


class DummyAirHockeyEnv(gym.Env):
    """Minimal Gymnasium environment matching Orion's observation/action sizes."""

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(8,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )
        self._observation = np.zeros(8, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._observation = self.observation_space.sample()
        return self._observation.copy(), {}

    def step(self, action: np.ndarray):
        self._observation = self.observation_space.sample()
        return self._observation.copy(), 0.0, False, False, {}


def generate_dummy_model(output_path: str | Path = DEFAULT_MODEL_PATH) -> Path:
    """Create and save a fresh untrained SAC model to ``output_path``."""
    path = Path(output_path).expanduser().resolve()
    if path.suffix.lower() != ".zip":
        path = path.with_suffix(".zip")
    path.parent.mkdir(parents=True, exist_ok=True)

    environment = DummyAirHockeyEnv()
    model = SAC(
        "MlpPolicy",
        environment,
        verbose=0,
        device="cpu",
        seed=0,
    )
    model.save(str(path.with_suffix("")))
    environment.close()

    if not path.is_file():
        raise RuntimeError(f"SB3 did not create the expected model archive: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an untrained SAC model for Orion Vision smoke tests."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Destination .zip archive (default: {DEFAULT_MODEL_PATH})",
    )
    args = parser.parse_args()
    model_path = generate_dummy_model(args.output)
    print(f"Dummy SAC model created successfully: {model_path}")


if __name__ == "__main__":
    main()
