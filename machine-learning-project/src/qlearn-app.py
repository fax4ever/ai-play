import gymnasium as gym
from qlearn import QLearn
import matplotlib.pyplot as plt  # Library for plotting

def main():
    taxi_env = gym.make('Taxi-v3', render_mode="rgb_array", max_episode_steps=500)
    qlearn = QLearn(taxi_env)

    avg_return = qlearn.rollouts(5)
    print("avg return before learning", avg_return)

    qlearn.qlearn()

    avg_return = qlearn.rollouts(20)
    print("avg return after learning", avg_return)

    # Plotting the rewards
    plt.figure(figsize=(10,6))  # Set the figure size
    plt.plot(qlearn.rewards, label='Q-learning Train')  # Plot Q-learning training rewards
    plt.xlabel('Episode')  # Label x-axis
    plt.ylabel('Total Reward')  # Label y-axis
    plt.title('Q-Learning (Episode vs Rewards)')
    plt.legend()  # Display legend
    plt.show()  # Show the plot

if __name__ == "__main__":
    main()