import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from stable_baselines3 import DQN
from environment.custom_env import AdaptiveLearningEnv

def main():
    # Load environment with render_mode="human" for GUI
    env = AdaptiveLearningEnv(render_mode="human")
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "dqn", "run_01", "best_model")

    print(f"Loading DQN model from: {model_path}")
    model = DQN.load(model_path)

    obs, _ = env.reset()
    done = False

    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render GUI
        env.render()

        if terminated or truncated:
            print("Episode ended. Resetting environment...")
            obs, _ = env.reset()

if __name__ == "__main__":
    main()

