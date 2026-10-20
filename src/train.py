import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Type

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import A2C, DQN, PPO, SAC, TD3
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

# Import the custom traffic environment. Adjust the import path if the environment
# resides elsewhere in the project.
try:
    from src.envs.traffic_env import TrafficEnv  # type: ignore
except ImportError as e:
    raise ImportError(
        "Could not import TrafficEnv. Ensure that src/envs/traffic_env.py exists "
        "and defines a class named TrafficEnv that follows the Gymnasium API."
    ) from e


# Mapping from string identifiers to Stable‑Baselines3 algorithm classes.
ALGORITHM_REGISTRY = {
    "ppo": PPO,
    "a2c": A2C,
    "dqn": DQN,
    "sac": SAC,
    "td3": TD3,
}


def make_env(
    env_id: str,
    seed: Optional[int] = None,
    rank: int = 0,
    log_dir: Optional[Path] = None,
) -> Callable[[], gym.Env]:
    """
    Returns a callable that creates a new instance of the environment.

    Parameters
    ----------
    env_id : str
        Identifier for the environment. Currently only ``"traffic"`` is supported.
    seed : Optional[int]
        Random seed for reproducibility.
    rank : int
        Index of the subprocess (used only for logging).
    log_dir : Optional[Path]
        Directory where per‑environment logs are stored (if any).

    Returns
    -------
    Callable[[], gym.Env]
        A function that creates the environment when called.
    """

    def _init() -> gym.Env:
        if env_id == "traffic":
            env = TrafficEnv()
        else:
            raise ValueError(f"Unsupported env_id '{env_id}'.")
        if seed is not None:
            env.reset(seed=seed + rank)
            env.action_space.seed(seed + rank)
            env.observation_space.seed(seed + rank)
        if log_dir is not None:
            # Example of attaching a logger to the env; actual implementation
            # depends on the env's API.
            env.logger = logging.getLogger(f"env_{rank}")
        return env

    return _init


def get_vec_env(
    env_id: str,
    n_envs: int,
    seed: Optional[int],
    log_dir: Optional[Path],
    use_subproc: bool = True,
) -> VecEnv:
    """
    Create a vectorized environment.

    Parameters
    ----------
    env_id : str
        Identifier for the environment.
    n_envs : int
        Number of parallel environments.
    seed : Optional[int]
        Random seed.
    log_dir : Optional[Path]
        Directory for environment logs.
    use_subproc : bool
        Whether to use subprocesses (``True``) or a dummy vectorized env
        (``False``).

    Returns
    -------
    VecEnv
        A vectorized environment compatible with Stable‑Baselines3.
    """
    env_fns = [
        make_env(env_id=env_id, seed=seed, rank=i, log_dir=log_dir) for i in range(n_envs)
    ]
    if use_subproc and n_envs > 1:
        return SubprocVecEnv(env_fns)
    else:
        return DummyVecEnv(env_fns)


def parse_args() -> argparse.Namespace:
    """
    Parse command‑line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train a multi‑agent RL controller for dynamic traffic lights."
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="ppo",
        choices=sorted(ALGORITHM_REGISTRY.keys()),
        help="RL algorithm to use (default: ppo).",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1_000_000,
        help="Total number of training timesteps (default: 1_000_000).",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default="traffic",
        help="Identifier of the environment to train on (default: traffic).",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=8,
        help="Number of parallel environments (default: 8).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory where TensorBoard logs and checkpoints are saved.",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=100_000,
        help="Save a checkpoint every N timesteps (default: 100000).",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=50_000,
        help="Run evaluation every N timesteps (default: 50000).",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of episodes per evaluation (default: 10).",
    )
    parser.add_argument(
        "--reward-threshold",
        type=float,
        default=None,
        help="Stop training early if the evaluation reward exceeds this threshold.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for torch (cpu, cuda, auto).",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Enable TensorBoard logging.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress INFO level logging; only warnings and errors are shown.",
    )
    return parser.parse_args()


def configure_logging(log_dir: Path, quiet: bool) -> None:
    """
    Configure the root logger.

    Parameters
    ----------
    log_dir : Path
        Directory where the log file will be stored.
    quiet : bool
        If True, set level to WARNING; otherwise INFO.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "train.log"

    level = logging.WARNING if quiet else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.getLogger("gymnasium").setLevel(logging.ERROR)  # suppress gym noise


def main() -> None:
    """
    Entry point for training.
    """
    args = parse_args()

    # Resolve paths
    log_dir = Path(args.log_dir).expanduser().resolve()
    configure_logging(log_dir, args.quiet)
    logger = logging.getLogger(__name__)
    logger.info("Starting training with configuration: %s", args)

    # Set seeds for reproducibility
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # Prepare vectorized environment
    vec_env = get_vec_env(
        env_id=args.env_id,
        n_envs=args.n_envs,
        seed=args.seed,
        log_dir=log_dir,
        use_subproc=True,
    )

    # Choose algorithm class
    algo_cls: Type = ALGORITHM_REGISTRY[args.algo.lower()]
    logger.info("Selected algorithm: %s", algo_cls.__name__)

    # Callback for checkpointing
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=log_dir / "checkpoints",
        name_prefix=f"{args.algo}_checkpoint",
        save_replay_buffer=False,
        save_vecnorm=False,
        verbose=1,
    )

    # Callback for evaluation
    eval_env = get_vec_env(
        env_id=args.env_id,
        n_envs=1,
        seed=args.seed,
        log_dir=None,
        use_subproc=False,
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir / "best_model",
        log_path=log_dir / "eval",
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
    )

    # Early stopping based on reward threshold
    callbacks: List[BaseCallback] = [checkpoint_callback, eval_callback]
    if args.reward_threshold is not None:
        stop_callback = StopTrainingOnRewardThreshold(
            reward_threshold=args.reward_threshold,
            verbose=1,
        )
        eval_callback.stop_training_on_success = stop_callback
        callbacks.append(stop_callback)

    # TensorBoard logging
    tensorboard_log = str(log_dir / "tensorboard") if args.tensorboard else None

    # Device handling
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create the model
    model = algo_cls(
        "MlpPolicy",
        vec_env,
        seed=args.seed,
        verbose=1,
        tensorboard_log=tensorboard_log,
        device=device,
    )

    # Train
    logger.info("Commencing training for %d timesteps.", args.total_timesteps)
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        log_interval=10,
        tb_log_name=args.algo,
    )

    # Save final model
    final_path = log_dir / f"{args.algo}_final"
    final_path.mkdir(parents=True, exist_ok=True)
    model_path = final_path / "model.zip"
    model.save(model_path)
    logger.info("Training completed. Final model saved to %s", model_path)

    # Optional final evaluation
    logger.info("Running final evaluation on %d episodes.", args.eval_episodes)
    mean_reward, std_reward = eval_callback.evaluate_policy(
        model,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )
    logger.info(
        "Final evaluation results – Mean reward: %.2f ± %.2f",
        mean_reward,
        std_reward,
    )

    # Clean up
    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
