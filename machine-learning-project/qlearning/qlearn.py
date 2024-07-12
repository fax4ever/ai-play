import random
import gymnasium as gym
from qlearning.qdictionary import QDictionary

class QLearn:
    def __init__(self, env: gym.Env):
        self.learning_rate = 0.1 # (alpha0)
        self.discount_factor = 0.95 # (gamma)
        self.learning_steps = 200000
        self.env = env
        self.trained = False
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
            done = d1 or d2
            sum_returns += rew * disconting
            disconting *= self.discount_factor

        return sum_returns / n_episodes
    
    def qlearn(self) -> QDictionary:
        self.qtable = QDictionary(self.env.action_space.n)

        done = True
        for step in range(self.learning_steps):
            if done:
                # start of new episode from episode 0 (state S0)
                # episods may have different steps
                obs = self.env.reset()[0]

            # progress of learning process: 1=>start 0=>finish
            # linear decay of epsilon from 1 to 0
            eps = (self.learning_steps - step) / self.learning_steps
            action = self.learningAction(obs, eps)

            # get the new observation (that is next state --- full observability)
            # and the revenue we get performing the step
            obs2, rew, d1, d2, info = self.env.step(action)
            done = d1 or d2

            # Q(obs,a)
            oldQ = self.qtable.get(obs, action)
            # r + gamma * maxQ(obs2,<any>)
            newQ = rew + self.discount_factor * self.qtable.getMax(obs2)
            # Q formulas for the indeterministic case
            averagedQ = (1 - self.learning_rate) * oldQ + self.learning_rate * newQ
            self.qtable.set(obs, action, averagedQ)
            obs = obs2
        self.trained = True
        return self.qtable
    
    def qtable(self, qtable: QDictionary):
        self.qtable = qtable
        self.trained = True