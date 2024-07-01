import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from qlearn import QLearn

def main():
    taxi_env = gym.make('Taxi-v3', render_mode="rgb_array", max_episode_steps=500)
    qlearn = QLearn(taxi_env)

    avg_return = qlearn.rollouts(5)
    print("avg return before learning", avg_return)

    qlearn.qlearn()

    avg_return = qlearn.rollouts(20)
    print("avg return after learning", avg_return)

if __name__ == "__main__":
    main()