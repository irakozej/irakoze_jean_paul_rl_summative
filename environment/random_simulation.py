import time
from environment.custom_env import AdaptiveLearningEnv


def run_random_demo(episodes=3, seed=0):
    env = AdaptiveLearningEnv(max_steps=40, seed=seed)
    for ep in range(episodes):
        obs = env.reset()
        done = False
        print(f"Starting random demo episode {ep}")
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            env.render(mode='human')
            
            time.sleep(0.08)
        print(f"Episode {ep} finished. Info: {info}")
    env.close()


if __name__ == '__main__':
    run_random_demo()