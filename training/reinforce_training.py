import os
import argparse
import json
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from environment.custom_env import AdaptiveLearningEnv


class PolicyNet(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes=(128, 64)):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, action_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def ensure_dirs():
    os.makedirs('models/reinforce', exist_ok=True)
    os.makedirs('results/reinforce', exist_ok=True)


def discount_rewards(rewards, gamma):
    discounted = []
    r = 0
    for reward in reversed(rewards):
        r = reward + gamma * r
        discounted.append(r)
    return list(reversed(discounted))


def train_reinforce(env, total_episodes=2000, gamma=0.99, lr=1e-3, batch_size=5, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = PolicyNet(obs_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    all_rewards = []
    all_lengths = []

    for ep in range(1, total_episodes + 1):
        obs = env.reset()
        done = False
        rewards = []
        log_probs = []
        length = 0

        while not done:
            obs_v = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits = policy(obs_v)
            probs = torch.softmax(logits, dim=-1)
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            logp = m.log_prob(action)

            obs, reward, done, info = env.step(int(action.item()))
            rewards.append(reward)
            log_probs.append(logp)
            length += 1

        all_rewards.append(sum(rewards))
        all_lengths.append(length)

        if ep % batch_size == 0:
            returns = discount_rewards(rewards, gamma)
            returns_v = torch.tensor(returns, dtype=torch.float32)
            returns_v = (returns_v - returns_v.mean()) / (returns_v.std() + 1e-8)

            policy_loss = []
            for lp, R in zip(log_probs, returns_v):
                policy_loss.append(-lp * R)
            loss = torch.stack(policy_loss).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if ep % 50 == 0:
            avg_reward = np.mean(all_rewards[-50:])
            print(f"Episode {ep} | AvgReward(50): {avg_reward:.2f} | AvgLen(50): {np.mean(all_lengths[-50:]):.2f}")
            torch.save(policy.state_dict(), f"models/reinforce/policy_ep{ep}.pt")

    torch.save(policy.state_dict(), "models/reinforce/policy_final.pt")

    with open('results/reinforce/summary.json', 'w') as f:
        json.dump({
            'avg_reward': float(np.mean(all_rewards)),
            'avg_length': float(np.mean(all_lengths)),
        }, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    ensure_dirs()
    env = AdaptiveLearningEnv(max_steps=50, seed=args.seed)
    train_reinforce(env, total_episodes=args.episodes, gamma=args.gamma, lr=args.lr, batch_size=args.batch_size, seed=args.seed)
    env.close()


if __name__ == '__main__':
    main()
