from collections import deque
import random
import numpy as np


class Env:
    def __init__(self):
        self.state = 0
        self.end_state = 5
        self.dead_state = -5

    def run(self, action):
        self.state += action
        if self.state == self.end_state:
            return 1, True
        elif self.state == self.dead_state:
            return -1, True
        else:
            return 0, False

    def observe(self):
        return np.array([self.state])

    def reset(self):
        self.state = 0

    def choose(self, action):
        return 1 if action[0, 0] > action[0, 1] else -1


class ExperiencePool:
    def __init__(self):
        self.pool = deque(maxlen=1000)

    def append(self, item):
        self.pool.append(item)

    def __len__(self):
        return self.pool.__len__()

    def sample(self, batch_size=8):
        samp = random.sample(self.pool, batch_size)
        batch_state = []
        batch_action = []
        batch_nstate = []
        batch_reward = []
        batch_done = []
        for exp in samp:
            batch_state.append(exp.state)
            batch_nstate.append(exp.nstate)
            batch_action.append(exp.action)
            batch_reward.append(exp.reward)
            batch_done.append(exp.done)
        return np.array(batch_state).reshape(batch_size, 1), \
               np.array(batch_action).reshape(batch_size, 2), \
               np.array(batch_reward).reshape(batch_size, 1), \
               np.array(batch_nstate).reshape(batch_size, 1), \
               np.array(batch_done).reshape(batch_size, 1)

class Experience:
    def __init__(self, state, action, reward, nstate, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.nstate = nstate
        self.done = done

