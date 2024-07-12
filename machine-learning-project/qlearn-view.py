import gymnasium as gym
from qlearning.qlearn import QLearn

def main():
    qlearn = QLearn(gym.make('FrozenLake-v1', max_episode_steps=500))

    avg_return = qlearn.rollouts(5)
    print("avg return before learning", avg_return)

    qtable = qlearn.qlearn()

    avg_return = qlearn.rollouts(20)
    print("avg return after learning", avg_return)

    qlearn2 = QLearn(gym.make('FrozenLake-v1', max_episode_steps=500, render_mode="human"))
    qlearn2.qtable(qtable)

    avg_return = qlearn2.rollouts(5)
    print("avg return after learning", avg_return)

if __name__ == "__main__":
    main()