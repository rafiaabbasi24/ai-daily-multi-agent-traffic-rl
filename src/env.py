"""
src/env.py
~~~~~~~~~~

Custom Gymnasium environment for a single intersection traffic signal control
problem. The environment models vehicle arrivals as a Poisson process and
departures according to the currently active traffic phase. It is designed to
be lightweight, deterministic given a seed, and suitable for training
reinforcement learning agents with libraries such as Stable‑Baselines3.

Typical usage
-------------
>>> from src.env import TrafficEnv
>>> env = TrafficEnv()
>>> obs, info = env.reset()
>>> for _ in range(100):
...     action = env.action_space.sample()
...     obs, reward, terminated, truncated, info = env.step(action)
...     if terminated or truncated:
...         break
"""

from __future__ import annotations

import random
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np

# --------------------------------------------------------------------------- #
# Environment definition
# --------------------------------------------------------------------------- #


class TrafficEnv(gym.Env):
    """
    A simple traffic signal control environment.

    The intersection has four incoming approaches (North, South, East, West).
    At each step a traffic phase is selected:

    * 0 – Green for North‑South, red for East‑West.
    * 1 – Green for East‑West, red for North‑South.

    Vehicles arrive on each approach according to independent Poisson processes.
    During a green phase a fixed number of vehicles can depart from each green
    lane (the *service rate*). The state consists of the current queue lengths
    for the four approaches.

    The episode ends after ``max_steps`` steps. The reward is the negative sum
    of queue lengths (i.e. we minimise total waiting vehicles).

    Parameters
    ----------
    arrival_rate : float, optional
        Expected number of vehicle arrivals per step on each approach.
        Must be non‑negative. Default is 0.2.
    service_rate : int, optional
        Maximum number of vehicles that can depart from a green lane per step.
        Must be positive. Default is 5.
    max_steps : int, optional
        Maximum number of steps per episode. Default is 200.
    seed : int | None, optional
        Random seed for reproducibility. If ``None`` the environment is not
        seeded.

    Notes
    -----
    The environment follows the Gymnasium API (observation and action spaces,
    ``reset`` and ``step`` signatures). It can be wrapped by ``gymnasium.wrappers``
    or used directly with Stable‑Baselines3 agents.
    """

    metadata = {"render_modes": ["human"], "render_fps": 2}

    # --------------------------------------------------------------------- #
    # Construction
    # --------------------------------------------------------------------- #

    def __init__(
        self,
        arrival_rate: float = 0.2,
        service_rate: int = 5,
        max_steps: int = 200,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        if arrival_rate < 0:
            raise ValueError("arrival_rate must be non‑negative")
        if service_rate <= 0:
            raise ValueError("service_rate must be positive")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")

        self.arrival_rate = float(arrival_rate)
        self.service_rate = int(service_rate)
        self.max_steps = int(max_steps)

        # Action space: 0 = NS green, 1 = EW green
        self.action_space = gym.spaces.Discrete(2)

        # Observation space: queue length for each of the 4 approaches.
        # We cap the queue length at a large number (e.g., 1000) to keep the
        # space bounded.
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1000,
            shape=(4,),
            dtype=np.int32,
        )

        # Internal state
        self._queues: np.ndarray = np.zeros(4, dtype=np.int32)  # N, S, E, W
        self._step_count: int = 0
        self._rng = np.random.default_rng(seed)
        self._seed = seed

    # --------------------------------------------------------------------- #
    # Helper methods
    # --------------------------------------------------------------------- #

    def _sample_arrivals(self) -> np.ndarray:
        """
        Sample vehicle arrivals for each approach using a Poisson distribution.

        Returns
        -------
        np.ndarray
            Array of shape (4,) containing integer arrivals for N, S, E, W.
        """
        # np.random.poisson works with the global RNG; we use the instance RNG.
        arrivals = self._rng.poisson(self.arrival_rate, size=4)
        return arrivals.astype(np.int32)

    def _apply_action(self, action: int) -> None:
        """
        Update queues based on the selected traffic phase.

        Parameters
        ----------
        action : int
            0 for NS green, 1 for EW green.
        """
        # Determine which indices are green.
        if action == 0:  # NS green
            green_idxs = (0, 1)  # N, S
        else:  # EW green
            green_idxs = (2, 3)  # E, W

        # Vehicles depart from green lanes, limited by service_rate and queue size.
        for idx in green_idxs:
            depart = min(self.service_rate, self._queues[idx])
            self._queues[idx] -= depart

    # --------------------------------------------------------------------- #
    # Gymnasium API
    # --------------------------------------------------------------------- #

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to an initial state.

        Parameters
        ----------
        seed : int | None
            Optional seed to re‑initialise the RNG.
        options : dict | None
            Unused, present for API compatibility.

        Returns
        -------
        observation : np.ndarray
            Initial queue lengths (all zeros).
        info : dict
            Empty dictionary (placeholder for additional info).
        """
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._seed = seed

        self._queues.fill(0)
        self._step_count = 0
        return self._queues.copy(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one time step.

        Parameters
        ----------
        action : int
            Traffic phase selected by the agent.

        Returns
        -------
        observation : np.ndarray
            Updated queue lengths.
        reward : float
            Negative total queue length.
        terminated : bool
            ``True`` when ``max_steps`` is reached.
        truncated : bool
            Always ``False`` (no external truncation condition).
        info : dict
            Contains ``step`` (current step count) and ``arrivals`` (sampled arrivals).
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action!r}")

        # 1. Vehicles arrive.
        arrivals = self._sample_arrivals()
        self._queues += arrivals

        # 2. Apply the chosen traffic phase (departures).
        self._apply_action(action)

        # 3. Increment step counter.
        self._step_count += 1

        # 4. Compute reward.
        total_queue = int(self._queues.sum())
        reward = -float(total_queue)

        # 5. Determine termination.
        terminated = self._step_count >= self.max_steps
        truncated = False  # No external truncation logic.

        info = {
            "step": self._step_count,
            "arrivals": arrivals,
            "queues": self._queues.copy(),
        }

        return self._queues.copy(), reward, terminated, truncated, info

    def render(self, mode: str = "human") -> None:
        """
        Render the current state to the console.

        Parameters
        ----------
        mode : str
            Only ``"human"`` is supported; other values raise ``NotImplementedError``.
        """
        if mode != "human":
            raise NotImplementedError(f"Render mode {mode!r} is not supported.")
        n, s, e, w = self._queues
        print(f"Step {self._step_count:3d} | N:{n:3d} S:{s:3d} | E:{e:3d} W:{w:3d}")

    def close(self) -> None:
        """Clean up resources (none needed for this environment)."""
        pass

    # --------------------------------------------------------------------- #
    # Optional: seed method for older Gym compatibility
    # --------------------------------------------------------------------- #

    def seed(self, seed: int | None = None) -> list[int]:
        """
        Set the environment's random seed.

        Parameters
        ----------
        seed : int | None
            Seed value.

        Returns
        -------
        list[int]
            List containing the seed used.
        """
        self.reset(seed=seed)
        return [self._seed] if self._seed is not None else []