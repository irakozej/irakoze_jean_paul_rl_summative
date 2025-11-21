# === evaluation/evaluate_agent.py ===
"""
Evaluate a single trained agent (SB3 DQN/PPO/A2C or PyTorch REINFORCE) on AdaptiveLearningEnv.
Saves per-episode metrics to a CSV file in results/evaluation/<model_name>/

Usage examples:
  python evaluation/evaluate_agent.py --model-path models/ppo/run_01/final_model.zip --algo ppo --episodes 100
  python evaluation/evaluate_agent.py --model-path models/reinforce/policy_final.pt --algo reinforce --episodes 100

Note: Make sure project root is in PYTHONPATH when running.
"""

import argparse
import os
import csv
import importlib
import numpy as np

from environment.custom_env import AdaptiveLearningEnv


def evaluate_sb3(model_path, algo, episodes=100, seed=0):
    # Lazy imports to avoid heavy deps if not needed
    from stable_baselines3.common.vec_env import DummyVecEnv
    if algo.lower() == 'dqn':
        from stable_baselines3 import DQN as SB3Model
    elif algo.lower() == 'ppo':
        from stable_baselines3 import PPO as SB3Model
    elif algo.lower() == 'a2c':
        from stable_baselines3 import A2C as SB3Model
    else:
        raise ValueError('Unsupported SB3 algo')

    env = AdaptiveLearningEnv(max_steps=50, seed=seed)
    vec_env = DummyVecEnv([lambda: env])
    # load model
    model = SB3Model.load(model_path, env=vec_env)

    results = []
    for ep in range(episodes):
        obs = vec_env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            total_reward += reward[0]  # reward is array
            steps += 1
            done = done[0]  # done is array
        results.append({'episode': ep, 'reward': total_reward, 'steps': steps, 'outcome': info[0].get('outcome', '')})
    vec_env.close()
    return results


def evaluate_reinforce(model_path, episodes=100, seed=0):
    import torch
    from training.reinforce_training import PolicyNet

    env = AdaptiveLearningEnv(max_steps=50, seed=seed)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = PolicyNet(obs_dim, action_dim)
    policy.load_state_dict(torch.load(model_path, map_location='cpu'))
    policy.eval()

    results = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        while not done:
            with torch.no_grad():
                obs_v = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                logits = policy(obs_v)
                probs = torch.softmax(logits, dim=-1).numpy().squeeze(0)
            action = int(np.argmax(probs))  # deterministic: choose argmax
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
        results.append({'episode': ep, 'reward': total_reward, 'steps': steps, 'outcome': info.get('outcome', '')})
    env.close()
    return results


def save_results_csv(results, out_dir, model_tag):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{model_tag}_eval.csv')
    keys = ['episode', 'reward', 'steps', 'outcome']
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"Saved evaluation CSV to {out_path}")
    # summary
    rewards = [r['reward'] for r in results]
    print(f"Mean reward: {np.mean(rewards):.3f} | Std: {np.std(rewards):.3f} | Median: {np.median(rewards):.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--algo', required=True, choices=['dqn', 'ppo', 'a2c', 'reinforce'])
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out-dir', default='results/evaluation')
    args = parser.parse_args()

    model_tag = os.path.splitext(os.path.basename(args.model_path))[0]

    if args.algo in ['dqn', 'ppo', 'a2c']:
        results = evaluate_sb3(args.model_path, args.algo, episodes=args.episodes, seed=args.seed)
    else:
        results = evaluate_reinforce(args.model_path, episodes=args.episodes, seed=args.seed)

    save_results_csv(results, args.out_dir, f"{args.algo}_{model_tag}")


if __name__ == '__main__':
    main()

