from collections import deque
import random
import numpy as np
from keras import layers, models, optimizers, losses
from qlearning.dqn_params import DQN_PARAMS

# the neural network:
# it is designed to have one input for each state value (in the input layer)
# and to have one output (linear - returning any real value) for each action
# we use the relu activation function to denote non-linear behaviours
def network(input_size: int, output_size: int):
    input = layers.Input([input_size])
    x = layers.Dense(24, activation="relu")(input)
    x = layers.Dense(24, activation="relu")(x)
    x = layers.Dense(output_size)(x)
    network = models.Model(inputs=input, outputs=x, name="regression_fc1")

    # if it does not perform very well, we can reduce the **learning_rate (learning step)** 
    # of the stocastic gradient descent here:
    network.compile(optimizer=optimizers.Adam(), loss=losses.MeanSquaredError())
    network.summary()
    return network

class QNetwork:
    def __init__(self, input_size: int, output_size: int, params: DQN_PARAMS):
        self.params = params
        # gamma is the discount_factor, provided by the problem using this network
        self.gamma = params.discount_factor
        # we will have minibatch with size batch_size, selected among our replay_buffer
        self.batch_size = 128 
        self.replay_buffer = deque(maxlen=50000)
        self.network = network(input_size, output_size)

    def action(self, state) -> int:
        q_values = self.network.predict(state)
        return np.argmax(q_values[0])

    def save(self, state, action, reward, next_state, done):
        self.replay_buffer.append([state, action, reward, next_state, done])
        if (not done and len(self.replay_buffer) > self.batch_size):
            self.__replay()

    def __replay(self):
        # the minibatch strategy supports the stocastic (for-non-linear) gradient descend
        minibatch = np.asarray(random.sample(self.replay_buffer, self.batch_size), dtype="object")
        for state, action, reward, next_state, done in minibatch:
            newQ = reward
            if not done:
                # new Q(obs,a) = r + gamma * maxQ(obs2,<any>)
                newQ += self.gamma * np.amax(self.network.predict(next_state))
            # Update the Q(obs,a)    
            Q = self.network.predict(state)
            Q[0][action] = newQ
            # provide this new output (Q) to the network
            # this value will be merged with possible prexisting contributions
            self.network.fit(state, Q, epochs=1, verbose=0)

