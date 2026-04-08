"""
Tests for the Spectrum Orchestration RL Environment (PyTorch)
"""

import sys
import os
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from environment import SpectrumOrchestrationEnv, Difficulty
from scoring import SpectrumScorer

class TestEnvironmentBasics:
    @pytest.fixture
    def env_easy(self):
        return SpectrumOrchestrationEnv(difficulty="easy", seed=42)

    def test_reset_returns_obs_and_info(self, env_easy):
        obs, info = env_easy.reset()
        assert isinstance(info, dict)
        assert len(obs.shape) == 1

    def test_step_returns_correct_tuple(self, env_easy):
        obs, _ = env_easy.reset()
        action = env_easy.action_space.sample()
        result = env_easy.step(action)
        assert len(result) == 5

    def test_episode_terminates(self, env_easy):
        obs, _ = env_easy.reset()
        done, steps = False, 0
        while not done:
            obs, reward, terminated, truncated, info = env_easy.step(env_easy.action_space.sample())
            done = terminated or truncated
            steps += 1
        assert steps == env_easy.max_steps

    def test_reward_is_bounded(self, env_easy):
        obs, _ = env_easy.reset()
        for _ in range(10):
            obs, reward, _, _, _ = env_easy.step(env_easy.action_space.sample())
            assert 0.0 <= reward <= 1.0

class TestInterference:
    def test_same_channel_causes_more_interference(self):
        env = SpectrumOrchestrationEnv(difficulty="easy", seed=42)
        env.reset()
        N = env.num_devices
        action_same = torch.zeros(2 * N, dtype=torch.float32)
        action_same[:N] = 0.0
        action_same[N:] = 1.0
        _, _, _, _, info_same = env.step(action_same)

        env.reset(seed=42)
        action_spread = torch.zeros(2 * N, dtype=torch.float32)
        action_spread[:N] = torch.linspace(0, 0.99, N)
        action_spread[N:] = 1.0
        _, _, _, _, info_spread = env.step(action_spread)
        assert info_same["interference_score"] <= info_spread["interference_score"]

    def test_lower_power_reduces_interference(self):
        env = SpectrumOrchestrationEnv(difficulty="easy", seed=42)
        env.reset()
        N = env.num_devices
        action_high = torch.zeros(2 * N, dtype=torch.float32)
        action_high[:N] = 0.0
        action_high[N:] = 1.0
        _, _, _, _, info_high = env.step(action_high)

        env.reset(seed=42)
        action_low = torch.zeros(2 * N, dtype=torch.float32)
        action_low[:N] = 0.0
        action_low[N:] = 0.1
        _, _, _, _, info_low = env.step(action_low)
        assert info_low["interference_score"] >= info_high["interference_score"]

class TestScoring:
    @pytest.fixture
    def scorer(self):
        return SpectrumScorer()

    def test_score_range(self, scorer):
        N = 10
        throughputs = torch.rand(N) * 1e7
        interference = torch.rand(N) * 1e-8
        powers = torch.rand(N) * 23
        demands = torch.rand(N) * 0.9 + 0.1
        result = scorer.score(throughputs, interference, powers, demands)
        assert 0.0 <= result["total_score"] <= 1.0

    def test_perfect_fairness(self, scorer):
        N = 10
        throughputs = torch.full((N,), 5e6)
        interference = torch.zeros(N)
        powers = torch.full((N,), 10.0)
        demands = torch.full((N,), 0.5)
        result = scorer.score(throughputs, interference, powers, demands)
        assert result["fairness_score"] > 0.99

class TestMediumDifficulty:
    def test_positions_change_with_mobility(self):
        env = SpectrumOrchestrationEnv(difficulty="medium", seed=42)
        obs1, _ = env.reset()
        N = env.num_devices
        pos_before = torch.tensor(obs1[:N])
        obs2, _, _, _, _ = env.step(env.action_space.sample())
        pos_after = torch.tensor(obs2[:N])
        assert not torch.allclose(pos_before, pos_after)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
