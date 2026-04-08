"""
Spectrum Orchestration RL Environment (PyTorch Version)
=====================================================
A Gymnasium-compatible environment for 5G/6G dynamic spectrum management,
using PyTorch for all internal radio simulations to avoid numpy requirements.
"""

from __future__ import annotations

import gymnasium as gym
import torch
from gymnasium import spaces
from enum import Enum
from typing import Optional

try:
    from src.scoring import SpectrumScorer
except ImportError:
    from scoring import SpectrumScorer

class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

DIFFICULTY_CONFIGS = {
    Difficulty.EASY: dict(
        num_devices=10, num_channels=4, area_size=200.0, num_towers=1,
        mobility=False, max_speed=0.0, shadow_fading_std=0.0, rayleigh=False, max_steps=100,
    ),
    Difficulty.MEDIUM: dict(
        num_devices=50, num_channels=8, area_size=500.0, num_towers=1,
        mobility=True, max_speed=5.0, shadow_fading_std=4.0, rayleigh=False, max_steps=200,
    ),
    Difficulty.HARD: dict(
        num_devices=200, num_channels=16, area_size=1000.0, num_towers=3,
        mobility=True, max_speed=30.0, shadow_fading_std=8.0, rayleigh=True, max_steps=300,
    ),
}

def _log_distance_path_loss(
    distance: torch.Tensor, d0: float = 1.0, pl_d0: float = 30.0, n: float = 3.5,
    shadow_std: float = 0.0, rng: torch.Generator | None = None
) -> torch.Tensor:
    distance = torch.clamp(distance, min=d0)
    pl = pl_d0 + 10.0 * n * torch.log10(distance / d0)
    if shadow_std > 0 and rng is not None:
        pl += torch.normal(0, shadow_std, size=pl.shape, generator=rng)
    return pl

def _db_to_linear(db: torch.Tensor) -> torch.Tensor:
    return 10.0 ** (db / 10.0)

class SpectrumOrchestrationEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"]}
    P_MIN, P_MAX = 0.0, 23.0
    NOISE_FLOOR, BANDWIDTH = -104.0, 10e6

    def __init__(self, difficulty: str | Difficulty = Difficulty.MEDIUM, render_mode: Optional[str] = None, seed: Optional[int] = None):
        super().__init__()
        self.difficulty = Difficulty(difficulty) if isinstance(difficulty, str) else difficulty
        cfg = DIFFICULTY_CONFIGS[self.difficulty]
        self.cfg, self.render_mode = cfg, render_mode

        self.num_devices, self.num_channels = cfg["num_devices"], cfg["num_channels"]
        self.area_size, self.num_towers = cfg["area_size"], cfg["num_towers"]
        self.mobility, self.max_speed = cfg["mobility"], cfg["max_speed"]
        self.shadow_std, self.use_rayleigh = cfg["shadow_fading_std"], cfg["rayleigh"]
        self.max_steps = cfg["max_steps"]

        N, C = self.num_devices, self.num_channels
        import numpy as np # ONLY for Gym spaces typing requirements, not for physics loop. Gym inherently requires numpy bounds mapping for continuous specs
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6 * N,), dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2 * N,), dtype=np.float32)

        self.scorer = SpectrumScorer()
        self._rng = torch.Generator()
        if seed is not None:
            self._rng.manual_seed(seed)
            
        self._step_count = 0

    def reset(self, *, seed=None, options=None):
        if seed is not None: self._rng.manual_seed(seed)
        N = self.num_devices

        if self.num_towers == 1:
            self._tower_pos = torch.tensor([[self.area_size / 2, self.area_size / 2]])
        else:
            angles = torch.linspace(0, 2 * 3.14159, self.num_towers + 1)[:-1]
            r = self.area_size * 0.3
            cx, cy = self.area_size / 2, self.area_size / 2
            self._tower_pos = torch.stack([cx + r * torch.cos(angles), cy + r * torch.sin(angles)], dim=1)

        self._device_pos = torch.rand((N, 2), generator=self._rng) * self.area_size
        if self.mobility:
            speed = torch.rand(N, generator=self._rng) * self.max_speed
            angle = torch.rand(N, generator=self._rng) * 2 * 3.14159
            self._device_vel = torch.stack([speed * torch.cos(angle), speed * torch.sin(angle)], dim=1)
        else:
            self._device_vel = torch.zeros((N, 2))

        self._device_demands = torch.rand(N, generator=self._rng) * 0.9 + 0.1
        self._channel_allocs = torch.randint(0, self.num_channels, (N,), generator=self._rng, dtype=torch.float32)
        self._power_allocs = torch.full((N,), (self.P_MIN + self.P_MAX) / 2)
        self._interference = torch.zeros(N)
        self._step_count = 0

        # We return numpy *only* for the Gym interface border, physics stay PyTorch
        return self._get_obs().numpy(), self._get_info()

    def step(self, action):
        action_t = torch.as_tensor(action, dtype=torch.float32).flatten()
        N = self.num_devices

        channel_raw = action_t[:N]
        power_raw = action_t[N:]

        self._channel_allocs = torch.clamp(torch.floor(channel_raw * self.num_channels), 0, self.num_channels - 1)
        self._power_allocs = torch.clamp(self.P_MIN + power_raw * (self.P_MAX - self.P_MIN), self.P_MIN, self.P_MAX)

        if self.mobility:
            self._device_pos += self._device_vel
            for dim in range(2):
                below = self._device_pos[:, dim] < 0
                above = self._device_pos[:, dim] > self.area_size
                self._device_pos[below, dim] *= -1
                self._device_pos[above, dim] = 2 * self.area_size - self._device_pos[above, dim]
                self._device_vel[below, dim] *= -1
                self._device_vel[above, dim] *= -1

        gains, interf, sinr, throughputs = self._compute_radio()
        self._interference = interf

        score_info = self.scorer.score(throughputs, interf, self._power_allocs, self._device_demands, self.P_MAX)
        reward = score_info["total_score"]

        self._step_count += 1
        terminated = False
        truncated = self._step_count >= self.max_steps

        info = self._get_info()
        info.update(score_info)
        info["mean_sinr_db"] = float(torch.mean(10 * torch.log10(torch.clamp(sinr, min=1e-10))))
        info["mean_throughput_mbps"] = float(torch.mean(throughputs) / 1e6)

        return self._get_obs().numpy(), float(reward), terminated, truncated, info

    def _compute_radio(self):
        N = self.num_devices
        diff = self._device_pos.unsqueeze(1) - self._tower_pos.unsqueeze(0)
        dists_to_towers = torch.linalg.norm(diff, dim=2)
        dist_to_serving, nearest_tower = torch.min(dists_to_towers, dim=1)

        pl = _log_distance_path_loss(dist_to_serving, shadow_std=self.shadow_std, rng=self._rng)
        
        h2 = torch.ones(N)
        if self.use_rayleigh:
            h2 = -torch.log(torch.rand(N, generator=self._rng))

        gain_linear = _db_to_linear(-pl) * h2
        power_linear = _db_to_linear(self._power_allocs)
        signal = power_linear * gain_linear
        noise_linear = _db_to_linear(torch.tensor(self.NOISE_FLOOR))

        interference = torch.zeros(N)
        for c in range(self.num_channels):
            mask = self._channel_allocs == c
            if mask.sum() <= 1: continue
            indices = torch.where(mask)[0]
            for i_idx in range(len(indices)):
                i = indices[i_idx]
                for j_idx in range(i_idx + 1, len(indices)):
                    j = indices[j_idx]
                    d_j_to_tower_i = dists_to_towers[j, nearest_tower[i]]
                    pl_ji = _log_distance_path_loss(d_j_to_tower_i.unsqueeze(0))[0]
                    g_ji = _db_to_linear(-pl_ji)
                    interference[i] += power_linear[j] * g_ji

                    d_i_to_tower_j = dists_to_towers[i, nearest_tower[j]]
                    pl_ij = _log_distance_path_loss(d_i_to_tower_j.unsqueeze(0))[0]
                    g_ij = _db_to_linear(-pl_ij)
                    interference[j] += power_linear[i] * g_ij

        sinr = signal / (interference + noise_linear)
        throughput = self.BANDWIDTH * torch.log2(1.0 + sinr)
        return gain_linear, interference, sinr, throughput

    def _get_obs(self) -> torch.Tensor:
        N = self.num_devices
        pos_norm = self._device_pos / self.area_size
        diff = self._device_pos.unsqueeze(1) - self._tower_pos.unsqueeze(0)
        dists = torch.linalg.norm(diff, dim=2)
        dist_serving = torch.min(dists, dim=1)[0]
        pl = _log_distance_path_loss(dist_serving)
        gains = _db_to_linear(-pl)
        gains_log = torch.log10(torch.clamp(gains, min=1e-15))
        ch_norm = self._channel_allocs / max(self.num_channels - 1, 1)
        interf_log = torch.log10(torch.clamp(self._interference, min=1e-15))

        return torch.cat([pos_norm[:, 0], pos_norm[:, 1], self._device_demands, gains_log, ch_norm, interf_log])

    def _get_info(self) -> dict: return {"step": self._step_count}
