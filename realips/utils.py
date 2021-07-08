import numpy as np
import math
from datetime import datetime


class OrnsteinUhlenbeckActionNoise(object):

    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx
        return self.X


def states2observations(states):
    x, x_dot, theta, theta_dot, failed = states
    observations = [x, x_dot, math.sin(theta), math.cos(theta), theta_dot]
    return observations, failed


def get_current_time():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    return current_time
