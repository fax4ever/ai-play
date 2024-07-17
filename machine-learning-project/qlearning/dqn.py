import random
import numpy as np
import gymnasium as gym
from qlearning.dqn_params import DQN_PARAMS
from qlearning.qnetwork import QNetwork

class DQN:
    def __init__(self, env: gym.Env, params: DQN_PARAMS):
        self.params = params
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
        return random.randint(0, self.action_size - 1)
    
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
            obs, rew, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            sum_returns += rew * disconting
            disconting *= self.params.discount_factor

        return sum_returns / n_episodes

    def qlearn(self):
        self.qtable = QNetwork(self.state_size, self.action_size, self.params)
        self.rewards = []  # List to store rewards per episode

        for episode in range(self.params.train_episodes):
            done = False
            terminated = False
            truncated = False
            state = self.env.reset()[0]
            state = np.reshape(state, [1, self.state_size])
            total_reward = 0
            
            while not done:
                self.env.render()
                eps = (self.params.train_episodes - episode) / self.params.train_episodes
                action = self.learningAction(state, eps)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                total_reward += reward
                done = terminated or truncated
                next_state = np.reshape(next_state, [1, self.state_size])
                self.qtable.save(state, action, reward, next_state, done)
                state = next_state

            self.rewards.append(total_reward)
            print('Total training rewards: {} for episod number = {} with final reward = {} terminated = {} truncated = {}'
                  .format(total_reward, episode, reward, terminated, truncated))
            