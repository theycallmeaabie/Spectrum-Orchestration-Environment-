"""
Baseline PPO Agent for Spectrum Orchestration
==============================================
A PyTorch implementation of Proximal Policy Optimization (PPO) to demonstrate
that the environment is learnable. This is a self-contained training script
with no external RL library dependencies beyond PyTorch.
"""

from __future__ import annotations
import argparse
import sys
import os
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from environment import SpectrumOrchestrationEnv, Difficulty

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.actor_mean = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, act_dim), nn.Sigmoid())
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))
        self.critic = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))

    def forward(self, obs: torch.Tensor):
        features = self.shared(obs)
        return self.actor_mean(features), torch.exp(self.actor_log_std.clamp(-5, 2)), self.critic(features).squeeze(-1)

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        mean, std, value = self.forward(obs)
        if deterministic: return mean, torch.zeros(1), value
        dist = Normal(mean, std)
        action = dist.sample().clamp(0, 1)
        return action, dist.log_prob(action).sum(-1), value

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor):
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std)
        return dist.log_prob(action).sum(-1), dist.entropy().sum(-1), value

class PPOTrainer:
    def __init__(self, env: SpectrumOrchestrationEnv, lr: float = 3e-4, gamma: float = 0.99, gae_lambda: float = 0.95, clip_eps: float = 0.2, entropy_coeff: float = 0.01, value_coeff: float = 0.5, ppo_epochs: int = 4, batch_size: int = 64, device: str = "cpu"):
        self.env, self.gamma, self.gae_lambda, self.clip_eps, self.entropy_coeff, self.value_coeff, self.ppo_epochs, self.batch_size, self.device = env, gamma, gae_lambda, clip_eps, entropy_coeff, value_coeff, ppo_epochs, batch_size, torch.device(device)
        self.model = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0]).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def collect_rollout(self) -> dict:
        obs_list, act_list, rew_list, val_list, logp_list, done_list = [], [], [], [], [], []
        obs, _ = self.env.reset()
        done = False
        while not done:
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            with torch.no_grad(): action, log_prob, value = self.model.get_action(obs_t)
            action_np = action.cpu().numpy().flatten()
            next_obs, reward, terminated, truncated, info = self.env.step(action_np)
            done = terminated or truncated
            obs_list.append(obs_t.squeeze(0))
            act_list.append(action.squeeze(0))
            rew_list.append(reward)
            val_list.append(value.item())
            logp_list.append(log_prob.item())
            done_list.append(done)
            obs = next_obs
        returns, advantages = self._compute_gae(rew_list, val_list, done_list)
        return {"obs": torch.stack(obs_list), "actions": torch.stack(act_list), "returns": returns, "advantages": advantages, "log_probs": torch.tensor(logp_list, dtype=torch.float32), "rewards": torch.tensor(rew_list, dtype=torch.float32), "info": info}

    def _compute_gae(self, rewards, values, dones):
        T = len(rewards)
        advantages = torch.zeros(T, dtype=torch.float32)
        last_gae = 0.0
        for t in reversed(range(T)):
            next_value = 0.0 if t == T - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
        returns = advantages + torch.tensor(values, dtype=torch.float32)
        return returns, (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    def update(self, rollout: dict):
        obs_t, act_t, ret_t, adv_t, old_logp_t = rollout["obs"].to(self.device), rollout["actions"].to(self.device), rollout["returns"].to(self.device), rollout["advantages"].to(self.device), rollout["log_probs"].to(self.device)
        T, total_loss = len(obs_t), 0.0
        for _ in range(self.ppo_epochs):
            indices = torch.randperm(T)
            for start in range(0, T, self.batch_size):
                idx = indices[start:min(start + self.batch_size, T)]
                new_logp, entropy, values = self.model.evaluate(obs_t[idx], act_t[idx])
                ratio = torch.exp(new_logp - old_logp_t[idx])
                surr1, surr2 = ratio * adv_t[idx], torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_t[idx]
                loss = -torch.min(surr1, surr2).mean() + self.value_coeff * nn.functional.mse_loss(values, ret_t[idx]) - self.entropy_coeff * entropy.mean()
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                total_loss += loss.item()
        return total_loss

    def train(self, num_episodes: int = 500, log_interval: int = 10, results_dir: Path | None = None):
        history = {"episode": [], "reward": [], "total_score": [], "throughput_score": [], "interference_score": [], "fairness_score": [], "power_score": [], "loss": []}
        best_score = -float("inf")
        print(f"Device: {self.device}")
        for ep in range(1, num_episodes + 1):
            rollout = self.collect_rollout()
            loss = self.update(rollout)
            ep_reward = float(rollout["rewards"].sum())
            info = rollout["info"]
            total_score = info.get("total_score", 0.0)
            history["episode"].append(ep)
            history["reward"].append(ep_reward)
            history["total_score"].append(total_score)
            history["throughput_score"].append(info.get("throughput_score", 0.0))
            history["interference_score"].append(info.get("interference_score", 0.0))
            history["fairness_score"].append(info.get("fairness_score", 0.0))
            history["power_score"].append(info.get("power_score", 0.0))
            history["loss"].append(loss)
            if total_score > best_score:
                best_score = total_score
                torch.save(self.model.state_dict(), Path(__file__).parent.parent / "checkpoints" / "best_model.pt")
            if ep % log_interval == 0:
                print(f"Episode {ep:>5d} | Reward {ep_reward:>7.3f} | Score {total_score:.3f} | Loss {loss:.4f}")
        if results_dir is not None:
            results_dir.mkdir(parents=True, exist_ok=True)
            out_path = results_dir / "training_history.json"
            with open(out_path, "w") as f:
                json.dump(history, f, indent=2)
            print(f"Training history saved to {out_path}")
        return history

def run_random_baseline(env: SpectrumOrchestrationEnv, num_episodes: int = 50):
    scores = []
    for _ in range(num_episodes):
        env.reset()
        done = False
        while not done:
            _, _, terminated, truncated, info = env.step(env.action_space.sample())
            done = terminated or truncated
        scores.append(info.get("total_score", 0.0))
    mean, std = torch.tensor(scores).mean().item(), torch.tensor(scores).std().item()
    print(f"Random baseline: Score = {mean:.4f} \u00B1 {std:.4f}")
    return mean, std

def main():
    parser = argparse.ArgumentParser(description="Train a PPO agent on Spectrum Orchestration.")
    parser.add_argument("--difficulty", type=str, default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--random-baseline", action="store_true", help="Run random baseline before PPO training.")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = SpectrumOrchestrationEnv(difficulty=args.difficulty, seed=42)
    (Path(__file__).parent.parent / "checkpoints").mkdir(exist_ok=True)
    results_dir = Path(__file__).parent.parent / "results"
    if args.random_baseline:
        print("--- Running random baseline ---")
        run_random_baseline(env, num_episodes=50)
        env.reset()  # fresh reset before PPO
    print(f"--- Training PPO ({args.difficulty}, {args.episodes} episodes) ---")
    trainer = PPOTrainer(env=env, device=device)
    trainer.train(num_episodes=args.episodes, results_dir=results_dir)

if __name__ == "__main__":
    main()
