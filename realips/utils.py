import numpy as np
import math
from datetime import datetime


def clip_or_wrap_func(a, a_min, a_max, clip_or_wrap):
    if clip_or_wrap == 0:
        return np.clip(a, a_min, a_max)
    return (a - a_min) % (a_max - a_min) + a_min


class ActionNoise:

    def __init__(self, action_dim, bounds, clip_or_wrap):
        self.action_dim = action_dim
        self.bounds = bounds
        self.clip_or_wrap = clip_or_wrap

    def sample(self) -> np.ndarray:
        pass

    def clip_or_wrap_action(self, action):
        if len(action) == 1:
            return clip_or_wrap_func(action, self.bounds[0], self.bounds[1], self.clip_or_wrap)
        return np.array([clip_or_wrap_func(a, self.bounds[0][k], self.bounds[1][k], self.clip_or_wrap[k]) for k, a in
                         enumerate(action)])

    def add_noise(self, action):
        sample = self.sample()
        action = self.clip_or_wrap_action(action + sample)
        return action


class OrnsteinUhlenbeckActionNoise(ActionNoise):

    def __init__(self, action_dim, bounds=(-1, 1), clip_or_wrap=0, mu=0, theta=0.15, sigma=0.1, dt=0.04):
        super().__init__(action_dim, bounds, clip_or_wrap)
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X) * self.dt
        dx = dx + self.sigma * np.random.randn(len(self.X)) * np.sqrt(self.dt)
        self.X = self.X + dx
        return self.X


class NoNoise(ActionNoise):
    def __init__(self, action_dim, bounds, clip_or_wrap):
        super().__init__(action_dim, bounds, clip_or_wrap)

    def sample(self):
        return np.zeros(self.action_dim)


class GaussianNoise:
    # todo implement GaussianNoise here
    # https://spinningup.openai.com/en/latest/algorithms/ddpg.html
    pass


def states2observations(states):
    x, x_dot, theta, theta_dot, failed = states
    observations = [x, x_dot, math.sin(theta), math.cos(theta), theta_dot]
    return observations, failed


def observations2states(observations, failed):
    x, x_dot, s_theta, c_theta, theta_dot = observations[:5]
    states = [x, x_dot, np.arctan2(s_theta, c_theta), theta_dot, failed]
    return states


def get_current_time():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    return current_time

# class OrnsteinUhlenbeckActionNoise(object):
#
#     def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
#         self.action_dim = action_dim
#         self.mu = mu
#         self.theta = theta
#         self.sigma = sigma
#         self.X = np.ones(self.action_dim) * self.mu
#
#     def reset(self):
#         self.X = np.ones(self.action_dim) * self.mu
#
#     def sample(self):
#         dx = self.theta * (self.mu - self.X)
#         dx = dx + self.sigma * np.random.randn(len(self.X))
#         self.X = self.X + dx
#         return self.X
