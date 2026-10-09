"""Three-stage PPO curriculum for the Air Hockey training environment."""

from __future__ import annotations

from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from ai.air_hockey_env import AirHockeyGymEnv

ROOT_DIR = Path(__file__).resolve().parents[1]
TENSORBOARD_LOG = str(ROOT_DIR / "sac_air_hockey_tensorboard")


def make_env(opponent_type: str, seed: int = 0, reward_mode: str = "aggressive"):
    """Build a monitored single-agent env for a curriculum stage."""

    def _factory():
        env = AirHockeyGymEnv(
            max_episode_steps=1000,
            seed=seed,
            reward_mode=reward_mode,
            opponent_type=opponent_type,
        )
        log_dir = ROOT_DIR / "training_logs" / opponent_type
        log_dir.mkdir(parents=True, exist_ok=True)
        return Monitor(env, filename=str(log_dir / f"{opponent_type}_{seed}"), allow_early_resets=True)

    return DummyVecEnv([_factory])


def train_stage(
    opponent_type: str,
    total_timesteps: int,
    save_name: str | Path,
    previous_model_path: str | None = None,
    stage_name: str = "stage",
    seed: int = 0,
    reward_mode: str = "aggressive",
    callback=None,
):
    """Train or continue PPO for one curriculum stage."""
    env = make_env(opponent_type=opponent_type, seed=seed, reward_mode=reward_mode)

    model_path = Path(save_name).expanduser()
    if not model_path.is_absolute():
        model_path = ROOT_DIR / model_path
    if previous_model_path and Path(previous_model_path).exists():
        model = PPO.load(str(previous_model_path), env=env)
        model.set_env(env)
        model.tensorboard_log = TENSORBOARD_LOG
    else:
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=2048,
            batch_size=64,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.02,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=TENSORBOARD_LOG,
            device="auto",
        )

    try:
        model.learn(
            total_timesteps=total_timesteps,
            reset_num_timesteps=False,
            tb_log_name=stage_name,
            callback=callback,
        )
        model.save(str(model_path))
    finally:
        env.close()
    return model


def main():
    stage1_path = ROOT_DIR / "stage1_ppo.zip"
    stage2_path = ROOT_DIR / "stage2_ppo.zip"
    final_path = ROOT_DIR / "final_curriculum_ppo.zip"

    stage1_model = train_stage(
        opponent_type="none",
        total_timesteps=300_000,
        save_name="stage1_ppo.zip",
        stage_name="Stage1_Puck_Raken",
        seed=42,
    )
    stage1_model.save(str(stage1_path))

    stage2_model = train_stage(
        opponent_type="static",
        total_timesteps=300_000,
        save_name="stage2_ppo.zip",
        previous_model_path=str(stage1_path),
        stage_name="Stage2_Static_Opponent",
        seed=42,
        reward_mode="defensive",
    )
    stage2_model.save(str(stage2_path))

    final_model = train_stage(
        opponent_type="complex",
        total_timesteps=500_000,
        save_name="final_curriculum_ppo.zip",
        previous_model_path=str(stage2_path),
        stage_name="Stage3_Complex_Opponent",
        seed=42,
        reward_mode="defensive",
    )
    final_model.save(str(final_path))


if __name__ == "__main__":
    main()
