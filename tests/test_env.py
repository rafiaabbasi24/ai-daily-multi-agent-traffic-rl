import importlib
import sys
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO


def _import_env_class() -> Any:
    """
    Attempt to import the traffic light environment class from a few
    likely locations. Raises ImportError if none are found.
    """
    possible_paths = [
        "envs.traffic_env:TrafficLightEnv",
        "traffic_env:TrafficLightEnv",
        "traffic_light_env:TrafficLightEnv",
        "dynamic_traffic_light.env:TrafficLightEnv",
    ]

    for path in possible_paths:
        module_path, _, class_name = path.partition(":")
        try:
            module = importlib.import_module(module_path)
            env_class = getattr(module, class_name)
            return env_class
        except (ImportError, AttributeError):
            continue

    raise ImportError(
        "Could not locate TrafficLightEnv. Tried the following import paths: "
        + ", ".join(p.split(":")[0] for p in possible_paths)
    )


# Retrieve the environment class once for all tests
TrafficLightEnv = _import_env_class()


@pytest.fixture(scope="function")
def env():
    """
    Pytest fixture that creates a fresh instance of the traffic light environment
    for each test and ensures it is properly closed after use.
    """
    env_instance = TrafficLightEnv()
    # Gymnasium environments may require a seed for reproducibility
    env_instance.reset(seed=42)
    yield env_instance
    env_instance.close()


def test_env_creation():
    """
    Verify that the environment can be instantiated and that its
    observation and action spaces are gymnasium spaces.
    """
    env = TrafficLightEnv()
    assert isinstance(env, gym.Env), "Environment should inherit from gym.Env"
    assert isinstance(env.observation_space, gym.Space), "observation_space must be a gym Space"
    assert isinstance(env.action_space, gym.Space), "action_space must be a gym Space"
    env.close()


def test_reset_returns_valid_observation(env):
    """
    Ensure that ``reset`` returns an observation that conforms to the declared
    observation space.
    """
    observation, info = env.reset()
    # ``observation`` may be a NumPy array or a dict of arrays.
    if isinstance(observation, dict):
        for key, value in observation.items():
            space = env.observation_space.spaces[key]  # type: ignore[attr-defined]
            assert space.contains(value), f"Observation key '{key}' is out of bounds"
    else:
        assert env.observation_space.contains(observation), "Reset observation out of bounds"


def test_step_returns_valid_tuple(env):
    """
    Verify that ``step`` returns a tuple with the correct types and that the
    observation respects the observation space.
    """
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    # Types
    assert isinstance(reward, (float, np.floating, int, np.integer)), "Reward must be numeric"
    assert isinstance(terminated, bool), "terminated flag must be bool"
    assert isinstance(truncated, bool), "truncated flag must be bool"
    assert isinstance(info, dict), "info must be a dict"

    # Observation validation
    if isinstance(observation, dict):
        for key, value in observation.items():
            space = env.observation_space.spaces[key]  # type: ignore[attr-defined]
            assert space.contains(value), f"Observation key '{key}' out of bounds after step"
    else:
        assert env.observation_space.contains(observation), "Step observation out of bounds"


def test_episode_termination(env):
    """
    Run a short episode to ensure that the environment eventually returns a
    terminal state (either ``terminated`` or ``truncated``). The test also
    checks that the total number of steps does not exceed a reasonable bound.
    """
    max_steps = getattr(env, "max_steps", 1000)
    steps = 0
    observation, _ = env.reset()
    while True:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        steps += 1
        if terminated or truncated:
            break
        if steps > max_steps:
            pytest.fail(f"Episode did not terminate after {max_steps} steps")
    assert steps > 0, "Episode should contain at least one step"


def test_stable_baselines_compatibility():
    """
    Instantiate a PPO agent from Stable‑Baselines3 with the traffic‑light
    environment and perform a very short learning run. This checks that the
    environment adheres to the Gymnasium API expected by SB3.
    """
    # Create a fresh environment instance for SB3
    env = TrafficLightEnv()
    # Wrap the environment if it does not already provide a ``render_mode`` attribute
    if not hasattr(env, "render_mode"):
        env = gym.make("GymnasiumEnvWrapper", env=env)  # type: ignore[arg-type]

    # Use a deterministic seed for reproducibility
    env.reset(seed=0)

    # Initialise PPO with a tiny network to keep the test fast
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-3,
        n_steps=32,
        batch_size=16,
        n_epochs=1,
        gamma=0.99,
        gae_lambda=0.95,
        verbose=0,
        device="cpu",
    )

    # Train for a minimal number of timesteps; this should finish quickly
    model.learn(total_timesteps=64)

    # Perform a single rollout to ensure the trained policy can act
    obs, _ = env.reset()
    action, _states = model.predict(obs, deterministic=True)
    assert env.action_space.contains(action), "Predicted action is out of bounds"

    env.close()
    model.env.close()
