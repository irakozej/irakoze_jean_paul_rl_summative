import matplotlib.pyplot as plt
import matplotlib.animation as animation

class EnvRenderer:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.ax.set_title("Adaptive Learning Environment – Live Simulation")
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)

        self.text_state = self.ax.text(0.05, 0.85, '', fontsize=12, va='top')
        self.text_action = self.ax.text(0.05, 0.65, '', fontsize=12, va='top')
        self.text_reward = self.ax.text(0.05, 0.45, '', fontsize=12, va='top')
        self.text_episode = self.ax.text(0.05, 0.25, '', fontsize=12, va='top')

        plt.tight_layout()
        plt.ion()
        plt.show()

    def render(self, state, action, reward, episode, step):
        mastery, difficulty = state[0], state[1]

        self.text_state.set_text(f"State:\nMastery: {mastery:.3f}\nDifficulty: {difficulty:.3f}")
        self.text_action.set_text(f"Action Taken: {action}")
        self.text_reward.set_text(f"Reward: {reward:.3f}")
        self.text_episode.set_text(f"Episode: {episode} | Step: {step}")

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


