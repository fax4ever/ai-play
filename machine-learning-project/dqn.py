import random
import numpy as np
import gymnasium as gym
from qnetwork import QNetwork

class DQN:
    def __init__(self, env: gym.Env):
        self.learning_rate = 0.1 # (alpha0)
        self.discount_factor = 0.95 # (gamma)
        self.learning_steps = 200000
        self.env = env
        self.trained = False
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        print("observation space", self.env.observation_space)
        print("action space", self.env.action_space)

    def learningAction(self, obs, epsilon_greedy: float) -> int:
        greedy = random.random() > epsilon_greedy
        if greedy:
            # greedy in the limit
            return self.greedyAction(obs)
        else:
            # random at the beginning
            return self.randomAction()    

    def rolloutsAction(self, obs) -> int:
        if self.trained:
            # greedy if we learned
            return self.greedyAction(obs)
        else:
            # random otherwise
            return self.randomAction()   

    def randomAction(self) -> int:
        return random.randint(0, self.env.action_space.n - 1)
    
    def greedyAction(self, obs) -> int:
        return self.qtable.action(obs)    

    def rollouts(self, n_episodes: int) -> float:
        sum_returns = 0.0
        done = False
        obs = self.env.reset()[0]
        disconting = 1
        ep = 0

        while True:
            if done:
                obs = self.env.reset()[0]
                disconting = 1
                ep += 1
                if ep > n_episodes:
                    break
            
            action = self.rolloutsAction(obs)
            #observation, reward, terminated, truncated, info
            obs, rew, d1, d2, info = self.env.step(action)
            done = d1 | d2
            sum_returns += rew * disconting
            disconting *= self.discount_factor

        return sum_returns / n_episodes

    def qlearn(self):
        self.qtable = QNetwork(self.state_size, self.action_size, self.discount_factor)
        num_episodes = 10
        for episode in range(num_episodes):
            state = self.env.reset()[0]
            state = np.reshape(state, [1, self.state_size])
            for t in range(500):
                self.env.render()
                eps = (num_episodes - episode) / num_episodes
                action = self.learningAction(state, eps)
                next_state, reward, d1, d2, _ = self.env.step(action)
                done = d1 or d2
                next_state = np.reshape(next_state, [1, self.state_size])

                self.qtable.save(state, action, reward, next_state, done)
                state = next_state
                if done:
                    break
