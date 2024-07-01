from collections import deque
import random
import numpy as np
from keras import layers, models, optimizers, losses, metrics

def network(input_size: int, output_size: int):
    network = models.Sequential([
        layers.Dense(24, input_dim=input_size, activation='relu'),
        layers.Dense(24, activation='relu'),
        layers.Dense(output_size, activation='linear')
    ])
    network.compile(optimizer=optimizers.Adam(), loss=losses.MeanSquaredError())
    network.summary()
    return network

class QNetwork:
    def __init__(self, input_size: int, output_size: int):
        self.replay_buffer = deque(maxlen=2000)
        self.network = network(input_size, output_size)

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        minibatch = np.array(random.sample(self.memory, batch_size))
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.network.predict(next_state)[0])
            target_f = self.network.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)        

