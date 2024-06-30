from collections import defaultdict
import numpy as np

class QDictionary:
    def __init__(self, n_actions):
        self.Q = defaultdict(lambda: np.zeros(n_actions))

    def action(self, obs) -> int:
        return np.argmax(self.Q[obs])
    
    def getMax(self, obs) -> float:
        return np.max(self.Q[obs])
    
    def get(self, obs, action: int) -> float:
        return self.Q[obs][action]
    
    def set(self, obs, action: int, value: action):
        self.Q[obs][action] = value