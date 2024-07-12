import gymnasium as gym
from qlearning.dqn import DQN

def main():
    dqn = DQN(gym.make('CartPole-v1'))

    avg_return = dqn.rollouts(5)
    print("avg return before learning", avg_return)

    dqn.qlearn()

    avg_return = dqn.rollouts(20)
    print("avg return after learning", avg_return)

if __name__ == "__main__":
    main()