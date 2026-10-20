import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv


class MultiAgentPPO:
    """
    Multi‑agent wrapper around Stable‑Baselines3 PPO.

    Each traffic‑light controller (agent) receives its own independent PPO
    instance.  The class abstracts the bookkeeping required to train,
    evaluate, save and load a collection of agents.

    Parameters
    ----------
    env_fns : Dict[str, Callable[[], gym.Env]]
        Mapping from an agent identifier to a callable that creates a *single‑agent*
        environment for that identifier.  The callable must return a fresh
        ``gym.Env`` instance each time it is invoked.
    policy : str, optional
        Policy architecture to use for all agents (default ``"MlpPolicy"``).
    ppo_kwargs : dict, optional
        Additional keyword arguments passed to the underlying ``stable_baselines3.PPO``
        constructor for each agent.
    """

    def __init__(
        self,
        env_fns: Dict[str, Callable[[], gym.Env]],
        policy: str = "MlpPolicy",
        **ppo_kwargs: Any,
    ) -> None:
        self.agent_ids: List[str] = list(env_fns.keys())
        self.env_fns: Dict[str, Callable[[], gym.Env]] = env_fns
        self.policy: str = policy
        self.ppo_kwargs: Dict[str, Any] = ppo_kwargs

        # Create a PPO instance per agent.  Environments are wrapped in a
        # ``DummyVecEnv`` because SB3 expects a vectorised environment.
        self.agents: Dict[str, PPO] = {}
        for aid in self.agent_ids:
            env = DummyVecEnv([env_fns[aid]])  # type: ignore[arg-type]
            self.agents[aid] = PPO(policy, env, **ppo_kwargs)

    # --------------------------------------------------------------------- #
    # Training utilities
    # --------------------------------------------------------------------- #
    def train(
        self,
        total_timesteps: int,
        log_interval: int = 10,
        callback: Optional[Callable[[str, int, PPO], None]] = None,
    ) -> None:
        """
        Train all agents sequentially for the given number of timesteps.

        The training loop is simple: each agent is trained for
        ``total_timesteps // n_agents`` timesteps.  This deterministic
        allocation keeps the overall wall‑clock time predictable and avoids
        inter‑agent interference.

        Parameters
        ----------
        total_timesteps : int
            Total number of environment steps to allocate across all agents.
        log_interval : int, optional
            How often (in training updates) each PPO instance prints its
            progress.  Passed directly to ``PPO.learn``.
        callback : Callable[[str, int, PPO], None], optional
            Optional user‑supplied callback invoked after each agent finishes
            its allocated training budget.  Arguments are ``agent_id``,
            ``timesteps_used`` and the trained ``PPO`` object.
        """
        if total_timesteps <= 0:
            raise ValueError("total_timesteps must be a positive integer")

        n_agents = len(self.agent_ids)
        if n_agents == 0:
            raise RuntimeError("No agents were provided to MultiAgentPPO")

        # Allocate timesteps as evenly as possible.
        base_steps = total_timesteps // n_agents
        remainder = total_timesteps % n_agents

        for idx, aid in enumerate(self.agent_ids):
            steps = base_steps + (1 if idx < remainder else 0)
            if steps == 0:
                continue
            self.agents[aid].learn(total_timesteps=steps, log_interval=log_interval)
            if callback is not None:
                callback(aid, steps, self.agents[aid])

    # --------------------------------------------------------------------- #
    # Inference utilities
    # --------------------------------------------------------------------- #
    def act(
        self,
        observations: Dict[str, np.ndarray],
        deterministic: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Compute actions for all agents given a dictionary of observations.

        Parameters
        ----------
        observations : Dict[str, np.ndarray]
            Mapping from agent identifier to its current observation.
        deterministic : bool, optional
            Whether to use deterministic actions (default ``False``).

        Returns
        -------
        Dict[str, np.ndarray]
            Mapping from agent identifier to the selected action.
        """
        actions: Dict[str, np.ndarray] = {}
        for aid, obs in observations.items():
            if aid not in self.agents:
                raise KeyError(f"Agent id '{aid}' not recognised by MultiAgentPPO")
            # ``predict`` expects a batched observation; we add a dummy batch axis.
            action, _ = self.agents[aid].predict(obs, deterministic=deterministic)
            actions[aid] = np.asarray(action)
        return actions

    # --------------------------------------------------------------------- #
    # Persistence utilities
    # --------------------------------------------------------------------- #
    def save(self, save_dir: str) -> None:
        """
        Save each agent's policy to ``save_dir/<agent_id>.zip``.

        Parameters
        ----------
        save_dir : str
            Directory where the models will be stored.  It will be created if
            it does not already exist.
        """
        os.makedirs(save_dir, exist_ok=True)
        for aid, model in self.agents.items():
            path = os.path.join(save_dir, f"{aid}.zip")
            model.save(path)

    def load(self, load_dir: str) -> None:
        """
        Load agents' policies from ``load_dir/<agent_id>.zip``.

        Parameters
        ----------
        load_dir : str
            Directory containing the saved models.  Only agents whose files
            exist will be re‑loaded; missing files raise a ``FileNotFoundError``.
        """
        for aid in self.agent_ids:
            path = os.path.join(load_dir, f"{aid}.zip")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Model file for agent '{aid}' not found at {path}")
            # Re‑create the environment to satisfy the PPO constructor.
            env = DummyVecEnv([self.env_fns[aid]])  # type: ignore[arg-type]
            self.agents[aid] = PPO.load(path, env=env)

    # --------------------------------------------------------------------- #
    # Evaluation utilities
    # --------------------------------------------------------------------- #
    def evaluate(
        self,
        n_episodes: int = 5,
        deterministic: bool = True,
        render: bool = False,
    ) -> Dict[str, List[float]]:
        """
        Run a simple evaluation loop for each agent independently.

        The method returns a dictionary mapping each agent identifier to a list
        of episode returns.

        Parameters
        ----------
        n_episodes : int, optional
            Number of episodes to run per agent.
        deterministic : bool, optional
            Whether to use deterministic actions during evaluation.
        render : bool, optional
            If ``True`` and the underlying environment supports rendering,
            ``env.render()`` is called at each step.

        Returns
        -------
        Dict[str, List[float]]
            Episode returns per agent.
        """
        if n_episodes <= 0:
            raise ValueError("n_episodes must be a positive integer")

        results: Dict[str, List[float]] = {aid: [] for aid in self.agent_ids}
        for aid in self.agent_ids:
            env = self.env_fns[aid]()
            for _ in range(n_episodes):
                obs, _ = env.reset()
                done = False
                total_reward = 0.0
                while not done:
                    action, _ = self.agents[aid].predict(obs, deterministic=deterministic)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    total_reward += float(reward)
                    if render:
                        env.render()
                results[aid].append(total_reward)
                env.close()
        return results

    # --------------------------------------------------------------------- #
    # Helper utilities
    # --------------------------------------------------------------------- #
    def get_agent(self, agent_id: str) -> PPO:
        """
        Retrieve the underlying PPO instance for a given agent.

        Parameters
        ----------
        agent_id : str
            Identifier of the agent.

        Returns
        -------
        PPO
            The PPO model associated with ``agent_id``.
        """
        if agent_id not in self.agents:
            raise KeyError(f"Agent id '{agent_id}' not found")
        return self.agents[agent_id]

    def list_agents(self) -> List[str]:
        """
        Return the list of agent identifiers managed by this wrapper.

        Returns
        -------
        List[str]
            Agent IDs.
        """
        return self.agent_ids.copy()