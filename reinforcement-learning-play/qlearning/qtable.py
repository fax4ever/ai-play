from collections import defaultdict
import gymnasium as gym
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
        greedy = random.random() > eps
        if greedy:
            # greedy in the limit
            return np.argmax(self.Q[obs])
        else:
            # random at the beginning
            return random.randint(0, self.n_actions-1)
        
def qlearn(
        env: gym.Env,
        alpha0: float,      # learning rate (in this case is contant - in general it should decay)
        gamma: float,       # discounting factor
        max_steps: int,     # duration of the learning
):
    # at the beginning all the values are zero
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = EpsGreedyPolicy(Q)

    done = True
    for step in range(max_steps):
        if done:
            # start of new episode from episode 0 (state S0)
            # episods may have different steps
            obs = env.reset()[0]

        # progress of learning process: 1=>start 0=>finish
        # linear decay from 1 to 0
        eps = (max_steps - step) / max_steps
        action = policy(obs, eps)

        # get the new observation (that is next state --- full observability)
        # and the revenue we get performing the step
        obs2, rew, d1, d2, info = env.step(action)
        done = d1 | d2

        # Q formulas for the indeterministic case
        Q[obs][action] += alpha0 * (rew + gamma * np.max(Q[obs2]) - Q[obs][action])
        obs = obs2

    return Q

def rollouts(
    env: gym.Env,
    policy,
    gamma: float,
    n_episodes: int,
    render: bool        
) -> float:
    sum_returns = 0.0

    # Init
    done = False
    obs = env.reset()[0]
    disconting = 1
    ep = 0
    if render:
        env.render()

    # Iterate steps
    while True:
        if done:
            if render:
                print("New episode")
            obs = env.reset()[0]
            disconting = 1
            ep += 1
            if ep > n_episodes:
                break
        
        action = policy(obs)
        #observation, reward, terminated, truncated, info
        obs, rew, d1, d2, info = env.step(action)
        done = d1 | d2

        sum_returns += rew * disconting
        disconting *= gamma

        if render:
            env.render()
    return sum_returns / n_episodes
