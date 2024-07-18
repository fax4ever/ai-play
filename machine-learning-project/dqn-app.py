import gymnasium as gym
from qlearning.dqn import DQN
from qlearning.dqn_params import DQN_PARAMS
import matplotlib.pyplot as plt  # Library for plotting

def main():
    params = DQN_PARAMS()
    env = gym.make('CartPole-v1', max_episode_steps=params.max_episode_steps)
    dqn = DQN(env, params)

    avg_return = dqn.rollouts(5)
    print("avg return before learning", avg_return)

    dqn.qlearn()

    avg_return = dqn.rollouts(20)
    print("avg return after learning", avg_return)

    env.close()

    # Plotting the rewards
    plt.figure(figsize=(10,6))  # Set the figure size
    plt.plot(dqn.rewards, label='Q-learning Train')  # Plot Q-learning training rewards
    plt.xlabel('Episode')  # Label x-axis
    plt.ylabel('Total Reward')  # Label y-axis
    plt.title('Q-Learning (Episode vs Rewards)')
    plt.legend()  # Display legend
    plt.show()  # Show the plot

if __name__ == "__main__":
    main()