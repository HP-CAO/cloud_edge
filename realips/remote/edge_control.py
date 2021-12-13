import pickle
import threading
import time
# import asyncio

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

        self.weights_subscriber = self.redis_connection.subscribe(channel=self.params.redis_params.ch_edge_weights)
        self.training_mode_subscriber = self.redis_connection.subscribe(channel=self.params.redis_params.ch_edge_mode)
        self.plant_reset_subscriber = self.redis_connection.subscribe(channel=self.params.redis_params.ch_plant_reset)
        self.control_targets = self.params.control_params.control_targets
        self.agent_a_active = True  # True: agent_a is controller, False: agent_b is controller

        self.control_frequency = self.params.control_params.frequency
        self.sample_period = 1. / self.control_frequency
        self.action_noise_factor = self.params.ddpg_params.action_noise_factor

        self.shape_observations = 5
        if self.params.ddpg_params.add_actions_observations:
            self.shape_observations += self.params.ddpg_params.action_observations_dim
        self.agent_a = BaseAgent(self.params.ddpg_params, shape_observations=self.shape_observations, on_edge=True)
        self.agent_b = BaseAgent(self.params.ddpg_params, shape_observations=self.shape_observations, on_edge=True)
        self.agent_a.initial_model()
        self.agent_b.initial_model()
        self.agent_b.action_noise = self.agent_a.action_noise  # Correlate noise

        self.t2 = threading.Thread(target=self.update_weights)
        self.t3 = threading.Thread(target=self.receive_mode)
        self.t4 = threading.Thread(target=self.receive_reset_command)
        self.t5 = threading.Thread(target=self.loop_sending_edge_trajectory)

        self.trajectory_sending_condition = threading.Condition()
        self.training = True if eval_weights is None else False
        self.step = 0
        self.last_action = 0
        self.edge_trajectory = [0, 0, 0, 0, 0]

        if eval_weights is not None:
            self.agent_a.load_weights(eval_weights)
            self.agent_b.load_weights(eval_weights)
        elif params.control_params.initialize_from_cloud:
            print("waiting for weights from cloud")
            self.ini_weights_and_noise_factor_from_cloud(self.agent_a, self.agent_b)

    def reset_targets(self):
        if self.params.control_params.random_reset_target:
            x_target = np.random.uniform(-self.params.control_params.x_threshold,
                                         self.params.control_params.x_threshold) * 0.5
            self.control_targets = [x_target, 0.]
        else:
            self.control_targets = self.params.control_params.control_targets

    def receives_weights_and_noise_factor(self):
        weights_pack = self.weights_subscriber.parse_response()[2]
        weights, action_noise_factor = pickle.loads(weights_pack)
        return weights, action_noise_factor

    def send_ready_update(self, ready):
        ready_pack = pickle.dumps(ready)
        self.redis_connection.publish(channel=self.params.redis_params.ch_edge_ready_update, message=ready_pack)

    def send_edge_trajectory(self, edge_trajectory):
        """send trajectory from edge"""
        edge_trajectory_pack = pickle.dumps(edge_trajectory)
        # print("BW lower bound:", len(edge_trajectory_pack) * 8 * self.params.control_params.frequency / 2**20)
        self.redis_connection.publish(channel=self.params.redis_params.ch_edge_trajectory, message=edge_trajectory_pack)

    def ini_weights_and_noise_factor_from_cloud(self, *args):
        self.send_ready_update(True)
        weights, action_noise_factor = self.receives_weights_and_noise_factor()
        for agent in args:
            agent.set_actor_weights(weights)
            agent.set_action_noise_factor(action_noise_factor)

    def generate_action(self):
        pass

    def update_weights(self):

        while True:
            self.send_ready_update(True)

            weights, action_noise_factor = self.receives_weights_and_noise_factor()

            if self.agent_a_active:
                for i, w in enumerate(weights):
                    self.agent_b.actor.weights[i].assign(w)
                    time.sleep(0.001)
                    # asyncio.sleep(0)
                self.agent_b.set_action_noise_factor(action_noise_factor)
            else:
                for i, w in enumerate(weights):
                    self.agent_a.actor.weights[i].assign(w)
                    time.sleep(0.001)
                    # asyncio.sleep(0)
                self.agent_a.set_action_noise_factor(action_noise_factor)

            self.agent_a_active = not self.agent_a_active

    def run(self):
        self.t2.daemon = True
        self.t3.daemon = True
        self.t4.daemon = True
        self.t5.daemon = True

        self.t2.start()
        self.t3.start()
        self.t4.start()
        self.t5.start()

        self.generate_action()

    def receive_mode(self):
        """
        receive_mode to switch between training and testing
        """
        while True:
            mode_pack = self.training_mode_subscriber.parse_response()[2]
            mode = pickle.loads(mode_pack)
            self.training = mode[0]
            print("training:", self.training)

    def loop_sending_edge_trajectory(self):

        self.trajectory_sending_condition.acquire()
        while True:
            self.trajectory_sending_condition.wait()
            self.send_edge_trajectory(self.edge_trajectory)

    def reset_control(self):
        pass

    def initialize_plant(self):
        self.reset_control()

    def receive_reset_command(self):
        """
        receive reset command from the cloud trainer to reset the plant;
        resetting command comes when the current steps reach the max_steps of a single episode
        """
        while True:
            _ = self.plant_reset_subscriber.parse_response()[2]
            self.set_normal_mode(False)

    def set_normal_mode(self, normal_mode):
        pass

