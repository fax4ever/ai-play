from collections import defaultdict
import gymnasium as gym
from gymnasium import Env
import numpy as np
import random

class RandomPolicy:
    def __init__(self, n_actions):
        self.n_actions = n_actions
    def __call__(self, obs) -> int:
        return random.randint(0, self.n_actions-1)
    
class GreedyPolicy:
    def __init__(self, Q):
        self.Q = Q
    def __call__(self, obs) -> int:
        return np.argmax(self.Q[obs])
    
class EpsGreedyPolicy:
    def __init__(self, Q):
        self.Q = Q
        self.n_actions = len(Q[0])
    def __call__(self, obs, eps: float) -> int:
        greedy = random.random > eps
        if greedy:
            return np.argmax(self.Q[obs])
        else:
            return random.randint(0, self.n_actions-1)
        
def qlearn(
        env: gym.Env,
        alpha0: float,
        gamma: float,
        max_steps: int,
):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = EpsGreedyPolicy(Q)

    done = True
    for step in range(max_steps):
        if done:
            obs = env.reset()

        eps = (max_steps - step) / max_steps
        action = policy(obs, eps)

        obs2, rew, done, info = env.step(action)
