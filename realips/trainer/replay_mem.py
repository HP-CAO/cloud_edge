import numpy as np
import pickle


def shape(exp):
    if type(exp) is list:
        return len(exp)
    else:
        return 1


def type_of(exp):
    if type(exp) is bool:
        return bool
    else:
        return float


class ReplayMemory(object):
    """
    Replay memory class to store trajectories
    """

    def __init__(self, size, beta=1, alpha=1):
        """
        initializing the replay memory
        :param: beta, alpha are for prioritized replay_mem
        """
        self.new_head = False
        self.beta = beta
        self.alpha = alpha
        self.k = 0
        self.head = -1
        self.full = False
        self.size = size
        self.memory = None

    def initialize(self, experience):
        self.memory = [np.zeros(shape=(self.size, shape(exp)), dtype=type_of(exp)) for exp in experience]
        self.memory.append(np.zeros(shape=self.size, dtype=float))

    def add(self, experience):
        if self.memory is None:
            self.initialize(experience)
            priority = 1.0
        else:
            priority = np.max(self.memory[-1])

        if len(experience) + 1 != len(self.memory):
            raise Exception('Experiment not the same size as memory', len(experience), '!=', len(self.memory))

        for e, mem in zip(experience, self.memory):
            mem[self.k] = e
        self.memory[-1][self.k] = priority

        self.head = self.k
        self.new_head = True
        self.k += 1
        if self.k >= self.size:
            self.k = 0  # replace the oldest one with the latest one
            self.full = True

    def sample(self, batch_size):
        r = self.size
        if not self.full:
            r = self.k
        random_idx = np.random.choice(r, size=batch_size, replace=False)
        if self.new_head:
            random_idx[0] = self.head  # always add the latest one
            self.new_head = False

        return [mem[random_idx] for mem in self.memory]

    def get(self, start, length):
        return [mem[start:start + length] for mem in self.memory]

    def get_size(self):
        if self.full:
            return self.size
        return self.k

    def get_max_size(self):
        return self.size

    def reset(self):
        self.k = 0
        self.head = -1
        self.full = False
        self.memory = None
        self.alpha = 1  # put a parameter to reset alpha beta
        self.beta = 1
        self.new_head = False

    def shuffle(self):
        """
        to shuffle the whole memory
        """
        self.memory = self.sample(self.get_size())

    def save2file(self, file_path):
        with open(file_path, 'wb') as fp:
            pickle.dump(self.memory, fp)

    def load_memory_caches(self, path):

        with open(path, 'rb') as fp:
            memory = pickle.load(fp)
            if self.memory is None:
                self.memory = memory
            else:
                self.memory = np.hstack((self.memory, memory))

        print("Load memory caches, pre-filled replay memory!")

    def sample_prioritized(self, batch_size):
        r = self.size
        if not self.full:
            r = self.k
        P = (self.memory[-1][:r] / np.sum(self.memory[-1][:r]))
        random_idx = np.random.choice(r, size=batch_size, replace=False, p=P)

        importance_sampling_weight = (P[random_idx] / max(P)) ** (-self.beta)

        return random_idx, [mem[random_idx] for mem in self.memory], importance_sampling_weight

    def update_priority(self, idx, priorities):
        self.memory[-1][idx] = np.squeeze(priorities) ** self.alpha + 1e-3
