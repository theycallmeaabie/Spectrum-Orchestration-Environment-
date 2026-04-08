"""
Spectrum Orchestration Environment — OpenEnv Server
===================================================
Wraps the SpectrumOrchestrationEnv Gymnasium environment in the OpenEnv
MCPEnvironment base class so it can be served via FastAPI + WebSocket and
deployed to a Hugging Face Space.

Tools exposed via MCP:
  - assign_spectrum(channels, powers)  — take a spectrum management step
  - get_state()                        — return current observation dict
  - get_score()                        — return current score breakdown
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Action, Observation, State
from fastmcp import FastMCP

from src.environment import SpectrumOrchestrationEnv


class SpectrumOrchestrationEnvironment(MCPEnvironment):
    """
    OpenEnv-compliant server for the Spectrum Orchestration RL environment.

    Exposes the environment as MCP tools so LLM agents can interact with it.
    """

    def __init__(self, difficulty: str = "easy"):
        mcp = FastMCP("spectrum_orchestration_env")

        # We need a reference to self before tools are registered
        # so we use a container trick
        env_container: dict = {}

        @mcp.tool
        def assign_spectrum(channels: list, powers: list) -> dict:
            """
            Assign channel and power levels to all devices.

            Args:
                channels: List of floats in [0, 1] — one per device.
                          Mapped to discrete channel indices internally.
                powers:   List of floats in [0, 1] — one per device.
                          Mapped to dBm range [0, 23] internally.

            Returns:
                dict with reward, score breakdown, mean SINR (dB),
                mean throughput (Mbps), done flag, and step number.
            """
            env = env_container["env"]
            import torch, numpy as np

            N = env.num_devices
            ch = (list(channels) + [0.5] * N)[:N]
            pw = (list(powers) + [0.5] * N)[:N]
            action = np.array(ch + pw, dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(action)
            env_container["last_obs"] = obs.tolist()
            env_container["last_info"] = info
            env_container["done"] = terminated or truncated

            return {
                "reward": round(reward, 4),
                "done": bool(terminated or truncated),
                **{k: round(v, 4) if isinstance(v, float) else v
                   for k, v in info.items()},
            }

        @mcp.tool
        def get_state() -> dict:
            """
            Return the current observation vector as a named dict.

            Returns:
                dict containing step count and observation summary.
            """
            env = env_container.get("env")
            if env is None:
                return {"error": "Call reset first."}
            obs = env_container.get("last_obs", [])
            N = env.num_devices
            return {
                "step": env_container.get("last_info", {}).get("step", 0),
                "num_devices": N,
                "obs_length": len(obs),
                "obs_preview": obs[:10],
            }

        @mcp.tool
        def get_score() -> dict:
            """
            Return the latest score breakdown from the scoring engine.

            Returns:
                dict with total_score, throughput_score, interference_score,
                fairness_score, power_score.
            """
            info = env_container.get("last_info", {})
            keys = ["total_score", "throughput_score", "interference_score",
                    "fairness_score", "power_score"]
            return {k: round(info.get(k, 0.0), 4) for k in keys}

        super().__init__(mcp)

        self._env_container = env_container
        self._difficulty = difficulty
        self._state = State(episode_id=str(uuid4()), step_count=0)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        env = SpectrumOrchestrationEnv(difficulty=self._difficulty, seed=seed)
        self._env_container["env"] = env
        obs, info = env.reset(seed=seed)
        self._env_container["last_obs"] = obs.tolist()
        self._env_container["last_info"] = info
        self._env_container["done"] = False

        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )

        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "status": "ready",
                "difficulty": self._difficulty,
                "num_devices": env.num_devices,
                "num_channels": env.num_channels,
                "message": (
                    f"Spectrum Orchestration Env ready. "
                    f"{env.num_devices} devices, {env.num_channels} channels. "
                    f"Use assign_spectrum(channels, powers) to take steps."
                ),
            },
        )

    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        return Observation(
            done=False,
            reward=0.0,
            metadata={"error": f"Unknown action type: {type(action).__name__}. Use MCP tools."},
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        self._state.step_count += 1
        return super().step(action, timeout_s=timeout_s, **kwargs)

    async def step_async(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        self._state.step_count += 1
        return await super().step_async(action, timeout_s=timeout_s, **kwargs)

    @property
    def state(self) -> State:
        return self._state
