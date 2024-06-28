import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from top import rollouts, qlearn, RandomPolicy

#taxi_env = gym.make('Taxi-v3', render_mode="human")
taxi_env = gym.make('Taxi-v3')
taxi_env = TimeLimit(taxi_env, max_episode_steps=50)

# Markof Decison Process (finite states (observations) - finite actions)
print("Observation space", taxi_env.observation_space)
print("Action space", taxi_env.action_space)

avg_return = rollouts(
    env=taxi_env,
    policy=RandomPolicy(taxi_env.action_space.n),
    gamma=0.95,
    n_episodes=5,
    render=False,
)

print("Random avg return", avg_return)

qtable = qlearn(env=taxi_env, alpha0=0.1, gamma=0.95, max_steps=200000)

print("Learning finished")