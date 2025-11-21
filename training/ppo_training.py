import argparse
import os
import json
from itertools import product
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

from environment.custom_env import AdaptiveLearningEnv


def make_env(seed=None):
    def _init():
        env = AdaptiveLearningEnv(max_steps=50, seed=seed)
        env = Monitor(env)
        return env
    return _init


def ensure_dirs():
    os.makedirs('models/ppo', exist_ok=True)
    os.makedirs('results/ppo', exist_ok=True)


def get_hyperparam_grid():
    learning_rates = [3e-4, 1e-4]
    n_steps = [128, 512]
    batch_sizes = [64, 128]
    n_epochs = [3, 10]
    clip_ranges = [0.1, 0.2]

    combos = []
    for lr, ns, bs, ne, cr in product(learning_rates, n_steps, batch_sizes, n_epochs, clip_ranges):
        combos.append({
            'learning_rate': lr,
            'n_steps': ns,
            'batch_size': bs,
            'n_epochs': ne,
            'clip_range': cr,
        })
    return combos[:12]


def train_one(run_id: int, params: dict, total_timesteps: int, seed: int = 0):
    run_tag = f"run_{run_id:02d}"
    model_dir = os.path.join('models', 'ppo', run_tag)
    log_dir = os.path.join('results', 'ppo', run_tag)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    new_logger = configure(log_dir, ['stdout', 'csv', 'tensorboard'])

    env = DummyVecEnv([make_env(seed)])
    eval_env = DummyVecEnv([make_env(seed + 1000)])

    with open(os.path.join(log_dir, 'hyperparams.json'), 'w') as f:
        json.dump(params, f, indent=2)

    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=50, verbose=1)
    eval_callback = EvalCallback(eval_env, best_model_save_path=model_dir,
                                 log_path=log_dir, eval_freq=5000,
                                 deterministic=True, render=False, callback_after_eval=stop_callback)

    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=params['learning_rate'],
        n_steps=params['n_steps'],
        batch_size=params['batch_size'],
        n_epochs=params['n_epochs'],
        clip_range=params['clip_range'],
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

    print(f"Starting PPO hyperparam search with {len(grid)} runs, {total_timesteps} timesteps each")
    for i, params in enumerate(grid, start=1):
        print(f"Starting run {i}/{len(grid)}: {params}")
        train_one(i, params, total_timesteps, seed=args.seed + i)

    print("All PPO runs complete.")


if __name__ == '__main__':
    main()


