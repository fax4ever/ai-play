from collections import deque
import random
import numpy as np
from keras import layers, models, optimizers, losses
from qlearning.dqn_params import DQN_PARAMS

# the neural network:
# it is designed to have one input for each state value (in the input layer)
# and to have one output (linear - returning any real value) for each action
# we use the relu activation function to denote non-linear behaviours
def network(input_size: int, output_size: int, params: DQN_PARAMS):
    input = layers.Input([input_size])
    x = layers.Dense(24, activation="relu")(input)
    x = layers.Dense(24, activation="relu")(x)
    x = layers.Dense(output_size)(x)
    network = models.Model(inputs=input, outputs=x, name="regression_fc1")

    # if it does not perform very well, we can reduce the **learning_rate (learning step)** 
    # of the stocastic gradient descent here:
    network.compile(optimizer=optimizers.Adam(learning_rate=params.network_learning_rate), loss=losses.MeanSquaredError())
    network.summary()
    return network

class QNetwork:
    def __init__(self, input_size: int, output_size: int, params: DQN_PARAMS):
        self.params = params
        self.input_size = input_size
        self.replay_buffer = deque(maxlen=params.min_replay_buffer_size)
        self.network = network(input_size, output_size, params)
        self.total_steps = 0

    def action(self, state) -> int:
        q_values = self.network.predict(np.reshape(state, [1, self.input_size]))
        return np.argmax(q_values[0])

    def save(self, state, action, reward, next_state, done):
        self.replay_buffer.append([state, action, reward, next_state, done])
        self.total_steps += 1
        if (len(self.replay_buffer) < self.params.mini_batch_size or len(self.replay_buffer) < self.params.min_replay_buffer_size):
            return
        if (self.total_steps % self.params.steps_to_update_policy_model == 0 or done):
            self.__replay()

    def __replay(self):
        # the minibatch strategy supports the stocastic (for-non-linear) gradient descend
        minibatch = np.asarray(random.sample(self.replay_buffer, self.params.mini_batch_size), dtype="object")
        # S
        current_states = np.array([element[0] for element in minibatch])
        # Q(S) => [a1, a2, ...]
        current_qs_list = self.network.predict(current_states)
        # S'
        new_current_states = np.array([element[3] for element in minibatch])
        # Q(S') => [a1', a2', ...]
        future_qs_list = self.network.predict(new_current_states)

        X = []
        Y = []

        for index, (state, action, reward, _, done) in enumerate(minibatch):
            if not done:
                # new Q(obs,a) = r + gamma * maxQ(obs2,<any>)
                max_future_q = reward + self.params.discount_factor * np.amax(future_qs_list[index])
            else:
                # new Q(obs,a) = r
                max_future_q = reward
            current_qs = current_qs_list[index]
            current_qs[action] = (1 - self.params.learning_rate) * current_qs[action] + self.params.learning_rate * max_future_q
            X.append(state)
            Y.append(current_qs)

        self.network.fit(np.array(X), np.array(Y), batch_size=self.params.mini_batch_size, verbose=0, shuffle=True)

