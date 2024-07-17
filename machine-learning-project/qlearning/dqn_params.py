class DQN_PARAMS:
    def __init__(self):
        self.learning_rate = 0.1 # (alpha0)
        self.discount_factor = 0.95 # (gamma)
        self.replay_buffer_size = 50_000
        self.min_replay_buffer_size = 1_000 # keep this value grater or equal to mini_batch_size
        self.mini_batch_size = 128
        self.train_episodes = 300
        self.steps_to_update_policy_model = 4
        self.steps_to_update_target_model = 100
        self.network_learning_rate = 0.001 # (for the gradient descent)
        self.max_episode_steps = 500