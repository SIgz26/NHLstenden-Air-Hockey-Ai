"""Utilities for analyzing SAC training runs and benchmarking air-hockey models."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from stable_baselines3 import SAC
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from ai.air_hockey_env import AirHockeyGymEnv

CURRENT_DIR = Path.cwd()

if CURRENT_DIR.name == "research":
    PROJECT_ROOT = CURRENT_DIR.parent
else:
    PROJECT_ROOT = CURRENT_DIR

MODEL_ROOT = PROJECT_ROOT / "models/Sessie02"
TB_LOG_ROOT = PROJECT_ROOT / "sac_air_hockey_tensorboard" 

#PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "research" else Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MODEL_ROOT = PROJECT_ROOT / "models/Sessie02"
TB_LOG_ROOT = PROJECT_ROOT / "sac_air_hockey_tensorboard"


def find_tensorboard_runs(root: str | Path = TB_LOG_ROOT) -> list[Path]:
    """Return unique run directories containing TensorBoard event files."""
    root_path = Path(root)
    if not root_path.exists():
        return []
    return sorted({path.parent for path in root_path.rglob("event*") if path.is_file()})


def load_tensorboard_metrics(root: str | Path = TB_LOG_ROOT) -> dict[str, pd.DataFrame]:
    """Load scalar metrics from TensorBoard event files into per-tag DataFrames."""
    run_paths = find_tensorboard_runs(root)
    if not run_paths:
        return {}

    metrics: dict[str, pd.DataFrame] = {}
    for run_path in run_paths:
        event_acc = EventAccumulator(str(run_path))
        event_acc.Reload()
        for tag in event_acc.Tags().get("scalars", []):
            values = event_acc.Scalars(tag)
            frame = pd.DataFrame(
                {
                    "step": [entry.step for entry in values],
                    "value": [float(entry.value) for entry in values],
                    "tag": tag,
                    "run": run_path.name,
                }
            )
            metrics.setdefault(tag, []).append(frame)

    return {
        tag: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["step", "value", "tag", "run"])
        for tag, frames in metrics.items()
    }


def load_model(model_path: str | Path) -> SAC:
    """Load a trained SB3 SAC model from disk."""
    model_file = Path(model_path)
    if not model_file.is_absolute():
        model_file = PROJECT_ROOT / model_file
    return SAC.load(str(model_file), env=AirHockeyGymEnv(reward_mode="aggressive"))


def evaluate_model(
    model_path: str | Path,
    episodes: int = 100,
    reward_mode: str = "aggressive",
    max_steps: int = 1000,
    seed: int | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run deterministic episodes and summarize goals, shots, speed, and response latency.

    Response latency is measured from the puck entering the robot's half while
    moving toward its goal until the mallet starts moving toward the puck.
    """
    if episodes < 1:
        raise ValueError("episodes must be at least 1")

    env = AirHockeyGymEnv(reward_mode=reward_mode, max_episode_steps=max_steps, seed=seed)
    model_file = Path(model_path).expanduser()
    if not model_file.is_absolute():
        model_file = PROJECT_ROOT / model_file
    model = SAC.load(str(model_file), env=env)

    rows: list[dict[str, float | int]] = []
    total_goals_for = 0
    total_goals_against = 0
    total_wins = 0
    total_draws = 0

    for episode_idx in range(episodes):
        observation, _ = env.reset(seed=(seed + episode_idx) if seed is not None else None)
        episode_steps = 0
        episode_goals_for = 0
        episode_goals_against = 0
        episode_shots = 0
        episode_return = 0.0
        episode_mallet_speeds: list[float] = []
        reaction_start_step: int | None = None
        reaction_time_s: float | None = None
        done = False

        while not done:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_steps += 1
            episode_return += float(reward)
            robot_velocity = env.robot_velocity_mps
            velocity = float(np.linalg.norm(robot_velocity))
            episode_mallet_speeds.append(velocity)

            if info.get("robot_contact") and env.puck_velocity_mps[0] > 0.0:
                episode_shots += 1

            puck_position = env.puck_position_m
            puck_velocity = env.puck_velocity_mps
            if reaction_start_step is None and puck_position[0] < 0.0 and puck_velocity[0] < 0.0:
                reaction_start_step = episode_steps
            if reaction_start_step is not None and reaction_time_s is None and velocity > 0.05:
                toward_puck = puck_position - env.robot_position_m
                if float(np.dot(robot_velocity, toward_puck)) > 0.0:
                    reaction_time_s = (episode_steps - reaction_start_step) * env.DT

            if info.get("goal_scored"):
                if info.get("scored_for") == "robot":
                    episode_goals_for += 1
                elif info.get("scored_for") == "opponent":
                    episode_goals_against += 1

            done = terminated or truncated

        total_goals_for += episode_goals_for
        total_goals_against += episode_goals_against
        won = episode_goals_for > episode_goals_against
        draw = episode_goals_for == episode_goals_against
        total_wins += int(won)
        total_draws += int(draw)
        episode_duration_s = episode_steps * env.DT

        rows.append(
            {
                "episode": episode_idx + 1,
                "goals_for": episode_goals_for,
                "goals_against": episode_goals_against,
                "won": won,
                "draw": draw,
                "episode_length": episode_steps,
                "episode_duration_s": episode_duration_s,
                "episode_return": episode_return,
                "shots": episode_shots,
                "shots_per_minute": episode_shots / max(episode_duration_s / 60.0, env.DT / 60.0),
                "avg_mallet_speed": float(np.mean(episode_mallet_speeds)) if episode_mallet_speeds else 0.0,
                "reaction_time_s": reaction_time_s,
            }
        )

    results = pd.DataFrame(rows)
    total_goals = total_goals_for + total_goals_against
    summary = {
        "episodes": episodes,
        "goals_for": total_goals_for,
        "goals_against": total_goals_against,
        "wins": total_wins,
        "draws": total_draws,
        "win_rate": (total_wins / episodes) * 100.0,
        "goal_ratio": (total_goals_for / total_goals) * 100.0 if total_goals else 0.0,
        "goals_against_ratio": (total_goals_against / total_goals) * 100.0 if total_goals else 0.0,
        "avg_episode_length": float(results["episode_length"].mean()),
        "avg_episode_duration_s": float(results["episode_duration_s"].mean()),
        "avg_episode_return": float(results["episode_return"].mean()),
        "avg_shot_count": float(results["shots"].mean()),
        "avg_shots_per_minute": float(results["shots_per_minute"].mean()),
        "avg_mallet_speed": float(results["avg_mallet_speed"].mean()),
        "avg_reaction_time_s": float(results["reaction_time_s"].mean()),
    }
    env.close()
    return results, summary


def plot_tensorboard_metrics(metrics: dict[str, pd.DataFrame], selected_tags: list[str] | None = None) -> dict[str, Any]:
    """Plot scalar metrics for one or more tags and return the used figure objects."""
    selected = selected_tags or ["ep_rew_mean", "train/critic_loss", "train/actor_loss", "train/ent_coef"]
    figures: dict[str, Any] = {}

    for tag in selected:
        if tag not in metrics:
            continue
        frame = metrics[tag].copy()
        if frame.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=frame, x="step", y="value", hue="run", ax=ax, estimator="mean")
        ax.set_title(tag)
        ax.set_xlabel("Training step")
        ax.set_ylabel(tag)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        figures[tag] = fig
    return figures


def compare_reward_profiles(model_paths: dict[str, str | Path], episodes: int = 50) -> pd.DataFrame:
    """Evaluate available reward-profile models on matching seeded episodes."""
    rows: list[dict[str, float | str]] = []
    for profile_name, model_path in model_paths.items():
        if model_path is None or not Path(model_path).expanduser().exists():
            continue
        _, summary = evaluate_model(model_path=model_path, episodes=episodes, reward_mode=profile_name, seed=7)
        rows.append({
            "profile": profile_name,
            "win_rate": summary["win_rate"],
            "goal_ratio": summary["goal_ratio"],
            "goals_against_ratio": summary["goals_against_ratio"],
            "avg_episode_length": summary["avg_episode_length"],
            "avg_shot_count": summary["avg_shot_count"],
            "avg_shots_per_minute": summary["avg_shots_per_minute"],
            "avg_mallet_speed": summary["avg_mallet_speed"],
            "avg_reaction_time_s": summary["avg_reaction_time_s"],
        })
    return pd.DataFrame(rows)


def plot_reward_profile_comparison(profile_df: pd.DataFrame) -> Any:
    """Render a summary bar chart for reward-profile comparison."""
    metrics = [
        "win_rate",
        "goal_ratio",
        "goals_against_ratio",
        "avg_shots_per_minute",
        "avg_mallet_speed",
        "avg_reaction_time_s",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for axis, metric in zip(axes, metrics):
        sns.barplot(data=profile_df, x="profile", y=metric, ax=axis, palette="magma")
        axis.set_title(f"{metric.replace('_', ' ').title()}")
        axis.set_xlabel("Reward profile")
        axis.set_ylabel(metric.replace("_", " ").title())
        axis.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


def heatmap_robot_positions(episode_positions: list[np.ndarray]) -> Any:
    """Create a 2D occupation heatmap from all robot positions seen during evaluation."""
    non_empty_positions = [np.asarray(positions) for positions in episode_positions if len(positions)]
    if not non_empty_positions:
        raise ValueError("No robot positions available for heatmap generation.")
    flat_positions = np.vstack(non_empty_positions)

    x_min, x_max = -AirHockeyGymEnv.FIELD_LENGTH_M / 2.0, AirHockeyGymEnv.FIELD_LENGTH_M / 2.0
    y_min, y_max = -AirHockeyGymEnv.FIELD_WIDTH_M / 2.0, AirHockeyGymEnv.FIELD_WIDTH_M / 2.0

    fig, ax = plt.subplots(figsize=(8, 5))
    _, _, _, image = ax.hist2d(
        flat_positions[:, 0],
        flat_positions[:, 1],
        bins=28,
        range=[[x_min, x_max], [y_min, y_max]],
        cmap="Blues",
    )
    ax.set_title("Robot Position Heatmap")
    ax.set_xlabel("Field X (m)")
    ax.set_ylabel("Field Y (m)")
    ax.set_aspect("equal")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Visit count")
    fig.tight_layout()
    return fig


def collect_robot_positions(
    model_path: str | Path,
    episodes: int = 50,
    reward_mode: str = "aggressive",
    max_steps: int = 1000,
    seed: int = 7,
) -> list[np.ndarray]:
    """Collect robot positions across evaluation episodes for heatmap analysis."""
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    env = AirHockeyGymEnv(reward_mode=reward_mode, max_episode_steps=max_steps, seed=seed)
    model_file = Path(model_path).expanduser()
    if not model_file.is_absolute():
        model_file = PROJECT_ROOT / model_file
    model = SAC.load(str(model_file), env=env)
    positions: list[np.ndarray] = []

    try:
        for episode_idx in range(episodes):
            observation, _ = env.reset(seed=seed + episode_idx)
            done = False
            while not done:
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, _ = env.step(action)
                positions.append(env.robot_position_m.copy())
                done = terminated or truncated
    finally:
        env.close()
    return positions
