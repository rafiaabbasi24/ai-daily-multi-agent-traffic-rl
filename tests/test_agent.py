import pytest
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO

# Attempt to import the agent from the most likely locations.
# Adjust the import path according to the actual project structure.
try:
    # Common layout: src/agent.py
    from src.agent import TrafficLightAgent  # type: ignore
except ImportError:
    try:
        # Alternative layout: agents.py at project root
        from agents import TrafficLightAgent  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Could not import TrafficLightAgent. Ensure the project provides "
            "TrafficLightAgent in either 'src/agent.py' or 'agents.py'."
        ) from exc


def _make_dummy_env():
    """
    Create a lightweight Gymnasium environment suitable for quick tests.
    The environment must expose ``observation_space`` and ``action_space``
    compatible with the TrafficLightAgent implementation.
    """
    # Using CartPole as a generic placeholder; replace with the actual
    # traffic environment if available.
    env = gym.make("CartPole-v1", render_mode=None)
    # Reset the environment to obtain the initial observation.
    obs, _ = env.reset()
    return env, obs


def test_agent_initialization():
    """
    Verify that the TrafficLightAgent can be instantiated with a Gymnasium
    environment and that its internal policy matches the environment's spaces.
    """
    env, _ = _make_dummy_env()
    agent = TrafficLightAgent(env)

    # The agent should expose a ``policy`` attribute that is a Stable‑Baselines3 model.
    assert isinstance(agent.policy, PPO), "Agent policy should be an instance of PPO"

    # The policy's observation and action spaces must align with the environment.
    assert agent.policy.observation_space == env.observation_space, (
        "Policy observation space does not match environment observation space"
    )
    assert agent.policy.action_space == env.action_space, (
        "Policy action space does not match environment action space"
    )


def test_agent_predict_action():
    """
    Ensure that the agent can produce a valid action given a single observation.
    """
    env, obs = _make_dummy_env()
    agent = TrafficLightAgent(env)

    # ``predict`` should return a NumPy array (or scalar) compatible with the env's action space.
    action, _states = agent.predict(obs, deterministic=True)

    # For discrete action spaces, the action should be an integer within the valid range.
    if isinstance(env.action_space, gym.spaces.Discrete):
        assert isinstance(action, (int, np.integer)), "Action should be an integer for Discrete spaces"
        assert 0 <= int(action) < env.action_space.n, "Action out of bounds for Discrete space"
    else:
        # For continuous spaces, ensure shape and dtype match.
        assert isinstance(action, np.ndarray), "Action should be a NumPy array for continuous spaces"
        assert action.shape == env.action_space.shape, "Action shape mismatch"
        assert np.all(action >= env.action_space.low) and np.all(
            action <= env.action_space.high
        ), "Action values out of bounds"


def test_agent_training_updates_policy():
    """
    Train the agent for a very small number of timesteps and verify that the
    policy network parameters have been updated.
    """
    env, _ = _make_dummy_env()
    agent = TrafficLightAgent(env)

    # Capture a snapshot of the initial parameters.
    initial_state_dict = {k: v.clone() for k, v in agent.policy.policy.state_dict().items()}

    # Train for a minimal number of timesteps to keep the test fast.
    agent.learn(total_timesteps=200, log_interval=0)

    # Capture the parameters after training.
    updated_state_dict = agent.policy.policy.state_dict()

    # At least one parameter tensor should have changed.
    changed = False
    for key in initial_state_dict:
        if not torch.equal(initial_state_dict[key], updated_state_dict[key]):
            changed = True
            break

    assert changed, "Policy parameters did not change after training"


def test_agent_save_and_load(tmp_path):
    """
    Test that the agent can be saved to disk and reloaded without errors,
    and that the loaded agent produces the same actions as the original.
    """
    env, obs = _make_dummy_env()
    agent = TrafficLightAgent(env)

    # Train briefly to ensure the model has non‑default weights.
    agent.learn(total_timesteps=100, log_interval=0)

    # Save the model.
    save_path = tmp_path / "traffic_agent.zip"
    agent.save(str(save_path))

    # Load a new instance from the saved file.
    loaded_agent = TrafficLightAgent.load(str(save_path), env=env)

    # Predict actions with both agents and compare.
    action_original, _ = agent.predict(obs, deterministic=True)
    action_loaded, _ = loaded_agent.predict(obs, deterministic=True)

    # For deterministic policies, the actions should be identical.
    if isinstance(action_original, np.ndarray):
        np.testing.assert_allclose(action_original, action_loaded, err_msg="Loaded agent action differs")
    else:
        assert action_original == action_loaded, "Loaded agent action differs"
