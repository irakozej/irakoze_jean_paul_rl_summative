import gymnasium as gym
import numpy as np
from gymnasium import spaces
import math


class AdaptiveLearningEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    ACTIONS = [
        "deliver_beginner",
        "deliver_intermediate",
        "deliver_advanced",
        "give_practice",
        "give_remediation",
        "recommend_break",
        "assess_learning",
    ]

    RESOURCE_ONE_HOT = {
        "deliver_beginner": 0,
        "deliver_intermediate": 1,
        "deliver_advanced": 2,
        "give_practice": 3,
        "give_remediation": 4,
        "recommend_break": 5,
        "assess_learning": 6,
    }

    def __init__(self, max_steps: int = 50, seed: int | None = None):
        super().__init__()
        self.max_steps = max_steps
        self.current_step = 0
        self.seed(seed)

        # Observations:
        obs_low = np.array([0.0, 0.0, 0.0] + [0.0] * 7 + [0.0], dtype=np.float32)
        obs_high = np.array([1.0, 1.0, 1.0] + [1.0] * 7 + [1.0], dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        self.action_space = spaces.Discrete(len(self.ACTIONS))

        # internal state
        self.mastery = None
        self.engagement = None
        self.fatigue = None
        self.last_action = None

        # rendering helper
        self.renderer = None
        self.screen = None
        self.clock = None

        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self.seed(seed)
        self.current_step = 0
        self.mastery = float(self.np_random.uniform(0.0, 0.4))
        # engagement: how engaged at start
        self.engagement = float(self.np_random.uniform(0.4, 0.9))
        self.fatigue = float(self.np_random.uniform(0.0, 0.2))
        self.last_action = -1

        return self._get_obs(), {}

    def _get_obs(self):
        one_hot = [0.0] * len(self.ACTIONS)
        if self.last_action is not None and self.last_action >= 0:
            one_hot[self.last_action] = 1.0
        steps_remaining_norm = max(0.0, (self.max_steps - self.current_step) / self.max_steps)
        obs = [self.mastery, self.engagement, self.fatigue] + one_hot + [steps_remaining_norm]
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action {action}"
        self.current_step += 1
        self.last_action = int(action)

        action_name = self.ACTIONS[action]

        reward = 0.0
        info = {"action_name": action_name}
        
        # Convenience
        m = self.mastery
        e = self.engagement
        f = self.fatigue

        # small random factor
        noise = float(self.np_random.normal(0, 0.01))

        if action_name == "deliver_beginner":
            # best when mastery < 0.4
            if m < 0.4:
                delta = 0.05 + 0.05 * (0.4 - m) + noise
                reward += 1.0 * min(0.15, delta)
                self.mastery = min(1.0, m + delta)
                self.engagement = min(1.0, e + 0.02)
            else:
                # too easy — small penalty
                reward -= 0.2
                self.engagement = max(0.0, e - 0.05)
                self.fatigue = min(1.0, f + 0.02)

        elif action_name == "deliver_intermediate":
            if 0.3 <= m < 0.7:
                delta = 0.06 + 0.04 * (0.5 - abs(0.5 - m)) + noise
                reward += 1.2 * min(0.18, delta)
                self.mastery = min(1.0, m + delta)
                self.engagement = min(1.0, e + 0.03)
            else:
                reward -= 0.25
                self.engagement = max(0.0, e - 0.04)
                self.fatigue = min(1.0, f + 0.03)

        elif action_name == "deliver_advanced":
            if m >= 0.6:
                delta = 0.07 + 0.03 * (m - 0.6) + noise
                reward += 1.5 * min(0.2, delta)
                self.mastery = min(1.0, m + delta)
                self.engagement = min(1.0, e + 0.02)
            else:
                reward -= 0.35
                self.engagement = max(0.0, e - 0.06)
                self.fatigue = min(1.0, f + 0.05)

        elif action_name == "give_practice":
            delta = 0.04 + noise
            reward += 0.8 * delta
            self.mastery = min(1.0, m + delta)
            self.engagement = min(1.0, e + 0.01)
            self.fatigue = min(1.0, f + 0.01)

        elif action_name == "give_remediation":
            # helps low-mastery students more
            if m < 0.5:
                delta = 0.08 + 0.05 * (0.5 - m) + noise
                reward += 1.2 * min(0.25, delta)
                self.mastery = min(1.0, m + delta)
                self.engagement = min(1.0, e + 0.04)
            else:
                reward -= 0.15
                self.engagement = max(0.0, e - 0.03)
                self.fatigue = min(1.0, f + 0.02)

        elif action_name == "recommend_break":
            reward -= 0.02
            self.fatigue = max(0.0, f - 0.2)
            self.engagement = min(1.0, e + 0.05)

        elif action_name == "assess_learning":
            reward += 0.05
            self.engagement = max(0.0, min(1.0, e + noise))
            self.fatigue = min(1.0, f + 0.01)
        if self.fatigue > 0.7:
            penalty = (self.fatigue - 0.7) * 0.5
            reward -= penalty

        # engagement drop if many poor choices
        if self.engagement < 0.2:
            reward -= 0.1

        terminated = False
        truncated = False
        info.update({"mastery": self.mastery, "engagement": self.engagement, "fatigue": self.fatigue})

        # Terminal on mastery threshold
        if self.mastery >= 0.95:
            terminated = True
            reward += 5.0
            info["outcome"] = "mastery_reached"

        if self.current_step >= self.max_steps:
            truncated = True
            info.setdefault("outcome", "max_steps_reached")

        obs = self._get_obs()
        return obs, float(reward), terminated, truncated, info

    def render(self, mode="human"):
        try:
            from environment import rendering
        except Exception:
            import rendering
        return rendering.render_env(self, mode=mode)

    def close(self):
        try:
            from environment import rendering
        except Exception:
            import rendering
        rendering.cleanup()

# adding a comit here to test git changes
