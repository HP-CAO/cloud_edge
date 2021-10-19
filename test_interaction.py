import pickle
import threading
import struct
import time
import math
import numpy as np
from realips.agent.base import BaseAgent
from realips.remote.redis import RedisParams, RedisConnection
from realips.agent.ddpg import DDPGAgentParams


class ControlParams:
    def __init__(self):
        self.frequency = 30.00  # hz
        self.x_threshold = 0.3
        self.theta_dot_threshold = 20
        self.random_reset_target = False
        self.control_targets = [0., 0.]
        self.is_quick_recover = True
        self.agent_type = None
        self.initialize_from_cloud = True
        self.train_real = True
        self.action_factor = 5
        self.calibrating_period_steps = 10000
        self.random_reset_ini = True


class EdgeControlParams:
    def __init__(self):
        self.redis_params = RedisParams()
        self.ddpg_params = DDPGAgentParams()
        self.control_params = ControlParams()


class EdgeControl:
    def __init__(self, params: EdgeControlParams, eval_weights=None):
        self.params = params
        self.redis_connection = RedisConnection(self.params.redis_params)

        self.control_targets = self.params.control_params.control_targets

        self.control_frequency = self.params.control_params.frequency
        self.sample_period = 1. / self.control_frequency
        self.action_noise_factor = self.params.ddpg_params.action_noise_factor

        self.shape_observations = 5
        if self.params.ddpg_params.add_actions_observations:
            self.shape_observations += self.params.ddpg_params.action_observations_dim
        self.agent_a = BaseAgent(self.params.ddpg_params, shape_observations=self.shape_observations, on_edge=True)
        self.agent_a.initial_model()
        self.step = 0
        self.ep = 0
        self.training = True if eval_weights is None else False
        self.last_action = 0

        self.states_subscriber = self.redis_connection.subscribe(
            channel=self.params.redis_params.ch_plant_trajectory_segment)

        self.agent_a.load_weights(eval_weights)

    def reset_targets(self):
        if self.params.control_params.random_reset_target:
            x_target = np.random.uniform(-self.params.control_params.x_threshold,
                                         self.params.control_params.x_threshold) * 0.5
            self.control_targets = [x_target, 0.]
        else:
            self.control_targets = self.params.control_params.control_targets

    def receive_plant_trajectory(self):
        plant_states_pack = self.states_subscriber.parse_response()[2]
        plant_states = pickle.loads(plant_states_pack)
        return plant_states

    def sending_action_command(self, action_real): #todo check the sub channel
        # action_real = pickle.dumps(action_real)
        action_real = struct.pack('f', action_real)
        self.redis_connection.publish(channel=self.params.redis_params.ch_plant_reset, message=action_real)

    def generate_action(self):
        if self.params.ddpg_params.add_actions_observations:
            action_observations = np.zeros(shape=self.params.ddpg_params.action_observations_dim)
        else:
            action_observations = []
        while True:
            states = self.receive_plant_trajectory()
            stats_observation, failed = states2observations(states)
            observations = np.hstack((stats_observation, action_observations)).tolist()
            action = self.agent_a.get_exploitation_action(observations, self.control_targets)
            action_real = action * self.params.control_params.action_factor
            self.sending_action_command(action_real)
            if self.params.ddpg_params.add_actions_observations:
                action_observations = np.append(action_observations, action)[1:]

    def run(self):
        self.generate_action()


def states2observations(states):
    x, x_dot, theta, theta_dot, failed = states
    observations = [x, x_dot, math.sin(theta), math.cos(theta), theta_dot]
    return observations, failed