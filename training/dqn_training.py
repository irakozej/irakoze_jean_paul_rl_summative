import argparse
import os
import json
from itertools import product
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.custom_env import AdaptiveLearningEnv



def make_env(seed=None):
    def _init():
        env = AdaptiveLearningEnv(max_steps=50, seed=seed)
        env = Monitor(env)
        return env
    return _init


def safe_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def ensure_dirs():
    safe_mkdir("models")
    safe_mkdir("models/dqn")
    safe_mkdir("results")
    safe_mkdir("results/logs")


def get_hyperparam_grid():
    learning_rates = [1e-4, 5e-4, 1e-3]
    buffer_sizes = [10000, 30000, 50000]
    batch_sizes = [32, 64]
    tau_vals = [1.0, 0.005] 
    target_update_intervals = [500, 1000]

    combos = []
    for lr, buf, batch, tgt in product(learning_rates, buffer_sizes, batch_sizes, target_update_intervals):
        combos.append({
            'learning_rate': lr,
            'buffer_size': buf,
            'batch_size': batch,
            'target_update_interval': tgt,
        })
    return combos[:12]


def train_one(run_id: int, params: dict, total_timesteps: int, seed: int = 0):
    run_tag = f"run_{run_id:02d}"
    model_dir = os.path.join('models', 'dqn', run_tag)
    log_dir = os.path.join('results', 'dqn', run_tag)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    new_logger = configure(log_dir, ['stdout', 'csv', 'tensorboard'])

    env = DummyVecEnv([make_env(seed)])
    eval_env = DummyVecEnv([make_env(seed + 1000)])

    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=50, verbose=1)
    eval_callback = EvalCallback(eval_env, best_model_save_path=model_dir,
                                 log_path=log_dir, eval_freq=5000,
                                 deterministic=True, render=False, callback_after_eval=stop_callback)

    model = DQN(
        'MlpPolicy',
        env,
        learning_rate=params['learning_rate'],
        buffer_size=params['buffer_size'],
        batch_size=params['batch_size'],
        target_update_interval=params['target_update_interval'],
        verbose=1,
        seed=seed,
    )
    model.set_logger(new_logger)

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

  
    model_path = os.path.join(model_dir, 'final_model.zip')
    model.save(model_path)

    env.close()
    eval_env.close()

    with open(os.path.join(log_dir, 'train_done.txt'), 'w') as f:
        f.write(f"Training finished for {run_tag}\n")

    return model_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=200000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    ensure_dirs()
    grid = get_hyperparam_grid()
    total_timesteps = args.timesteps

    print(f"Starting DQN hyperparam search with {len(grid)} runs, {total_timesteps} timesteps each")
    for i, params in enumerate(grid, start=1):
        print(f"Starting run {i}/{len(grid)}: {params}")
        train_one(i, params, total_timesteps, seed=args.seed + i)

    print("All DQN runs complete.")


if __name__ == '__main__':
    main()
