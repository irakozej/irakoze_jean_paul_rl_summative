# ==============================
# environment/custom_env.py (GUI-ready modifications)
# ==============================
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .rendering import EnvRenderer

class AdaptiveLearningEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode=None):
        super().__init__()
        
        self.render_mode = render_mode
        self.renderer = EnvRenderer() if render_mode == "human" else None

        self.action_space = spaces.Discrete(4)  # 0=easy, 1=medium, 2=hard, 3=review
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(11,), dtype=np.float32
        )

        self.mastery = 0.2
        self.difficulty = 0.3
        self.max_steps = 50
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.mastery = 0.2
        self.difficulty = 0.3
        self.step_count = 0
        state = np.array([self.mastery, self.difficulty] + [0.0]*9, dtype=np.float32)
        return state, {}

    def step(self, action):
        self.step_count += 1

        # Difficulty change
        if action == 0:   # easy
            self.difficulty = max(0.0, self.difficulty - 0.05)
            reward = 0.1
        elif action == 1: # medium
            reward = 0.3
        elif action == 2: # hard
            self.difficulty = min(1.0, self.difficulty + 0.05)
            reward = 0.6
        else:             # review
            reward = 0.2

        self.mastery = min(1.0, self.mastery + reward * 0.1)

        state = np.array([self.mastery, self.difficulty] + [0.0]*9, dtype=np.float32)
        terminated = self.mastery >= 0.95
        truncated = self.step_count >= self.max_steps

        if self.render_mode == "human":
            self.renderer.render(
                state=state,
                action=action,
                reward=reward,
                episode=0,
                step=self.step_count,
            )

        return state, reward, terminated, truncated, {}

    def render(self):
        pass


# ==============================
# simulation/run_dqn_gui.py
# ==============================
import time
from stable_baselines3 import DQN
from environment.custom_env import AdaptiveLearningEnv

def run_gui(model_path="models/dqn/best_model.zip"):
    env = AdaptiveLearningEnv(render_mode="human")
    model = DQN.load(model_path)

    obs, _ = env.reset()
    episode_reward = 0
    step = 0

    while True:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        step += 1
        time.sleep(0.25)

        if terminated or truncated:
            print("Episode complete. Reward =", episode_reward)
            time.sleep(1)
            obs, _ = env.reset()
            episode_reward = 0
            step = 0

if __name__ == "__main__":
    run_gui()

