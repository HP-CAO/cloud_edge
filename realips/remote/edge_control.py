import copy
import pickle
import threading
import struct
import time
import signal
import sys
import numpy
import numpy as np
from simple_pid import PID
from quanser.hardware import HILError

from realips.agent.base import BaseAgent
from realips.env.quanser_plant import QuanserParams, QuanserPlant
from realips.utils import get_current_time
from realips.remote.redis import RedisParams, RedisConnection
from realips.agent.ddpg import DDPGAgent, DDPGAgentParams
from realips.utils import states2observations


class ControlParams:
    def __init__(self):
        self.frequency = 30.00  # hz
        self.x_threshold = 0.3
        self.theta_dot_threshold = 15
        self.random_reset_target = False
        self.control_targets = [0., 0.]
        self.is_quick_recover = True
        self.agent_type = None
        self.initialize_from_cloud = True
        self.train_real = True
        self.action_factor = 7
        self.calibrating_period = 5


class EdgeControlParams:
    def __init__(self):
        self.redis_params = RedisParams()
        self.ddpg_params = DDPGAgentParams()
        self.control_params = ControlParams()
        self.quanser_params = QuanserParams()


class EdgeControl:
    def __init__(self, params: EdgeControlParams, eval_weights=None):
        self.params = params
        self.redis_connection = RedisConnection(self.params.redis_params)

        self.weights_subscriber = self.redis_connection.subscribe(channel=self.params.redis_params.ch_edge_weights)
        self.training_mode_subscriber = self.redis_connection.subscribe(channel=self.params.redis_params.ch_edge_mode)
        self.plant_reset_subscriber = self.redis_connection.subscribe(channel=self.params.redis_params.ch_plant_reset)
        self.control_targets = self.params.control_params.control_targets
        self.agent_a_active = True  # True: agent_a is controller, False: agent_b is controller
        self.quanser_plant = QuanserPlant(self.params.quanser_params,
                                          self.params.control_params.frequency,
                                          self.params.control_params.x_threshold,
                                          self.params.control_params.theta_dot_threshold)

        self.control_frequency = self.params.control_params.frequency
        self.sample_period = self.quanser_plant.sample_period
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
        self.step = 0
        self.ep = 0
        self.training = True if eval_weights is None else False
        self.pid_controller = PID(Kp=0.0005, setpoint=0, sample_time=self.sample_period)
        self.last_action = 0

        if eval_weights is not None:
            self.agent_a.load_weights(eval_weights)
            self.agent_b.load_weights(eval_weights)
        elif params.control_params.initialize_from_cloud:
            print("waiting for weights from cloud")
            self.ini_weights_and_noise_factor_from_cloud(self.agent_a, self.agent_b)

        self.calibration()

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
        self.redis_connection.publish(channel=self.params.redis_params.ch_edge_trajectory, message=edge_trajectory_pack)

    def send_plant_trajectory(self, plant_trajectory):
        """send plant states list"""
        plant_trajectory_pack = pickle.dumps(plant_trajectory)
        self.redis_connection.publish(channel=self.params.redis_params.ch_plant_trajectory_segment,
                                      message=plant_trajectory_pack)

    def ini_weights_and_noise_factor_from_cloud(self, *args):
        self.send_ready_update(True)
        weights, action_noise_factor = self.receives_weights_and_noise_factor()
        for agent in args:
            agent.set_actor_weights(weights)
            agent.set_action_noise_factor(action_noise_factor)

    def generate_action(self):

        while True:

            self.ep += 1

            if self.params.ddpg_params.add_actions_observations:

                action_observations = np.zeros(shape=self.params.ddpg_params.action_observations_dim)
            else:
                action_observations = []

            while not self.quanser_plant.normal_mode:
                self.reset_control()

            t0 = time.time()
            time_out_counter = 0
            while self.quanser_plant.normal_mode:

                self.step += 1

                states = self.quanser_plant.get_encoder_readings()

                self.send_plant_trajectory(states)  # this is sent to the plant scope for monitoring

                normal_mode = self.quanser_plant.normal_mode

                stats_observation, failed = states2observations(states)

                observations = np.hstack((stats_observation, action_observations)).tolist()

                agent = self.agent_a if self.agent_a_active else self.agent_b

                if self.training:
                    action = agent.get_exploration_action(observations, self.control_targets)
                else:
                    action = agent.get_exploitation_action(observations, self.control_targets)

                # delta_t = time.time() - t0

                action_real = action * self.params.control_params.action_factor

                # print("normal_mode: ", self.quanser_plant.normal_mode)

                self.quanser_plant.write_analog_output(action_real)

                edge_trajectory = [observations, self.last_action, failed, normal_mode, self.step]

                self.last_action = action

                self.send_edge_trajectory(edge_trajectory)  # this is sent to the cloud trainer
                # print("Inference took {}s".format(delta_t))

                if self.params.ddpg_params.add_actions_observations:
                    action_observations = np.append(action_observations, action)[1:]

                one_loop_time = time.time() - t0

                if one_loop_time < self.sample_period:
                    time.sleep(self.sample_period - one_loop_time)
                    time_out_counter = 0
                else:
                    time_out_counter += 1

                if time_out_counter >= 10:
                    t0 = time.time()
                    print("TIMEOUT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                else:
                    t0 = t0 + self.sample_period

    def update_weights(self):

        while True:

            self.send_ready_update(True)

            weights, action_noise_factor = self.receives_weights_and_noise_factor()

            if self.agent_a_active:
                self.agent_b.set_actor_weights(weights)
                self.agent_b.set_action_noise_factor(action_noise_factor)
            else:
                self.agent_a.set_actor_weights(weights)
                self.agent_a.set_action_noise_factor(action_noise_factor)

            self.agent_a_active = not self.agent_a_active

    def run(self):
        self.t2.start()
        self.t3.start()
        self.t4.start()
        self.generate_action()

    def receive_mode(self):
        """
        receive_mode to switch between training and testing
        """
        while True:
            message = self.training_mode_subscriber.parse_response()[2]
            self.training = struct.unpack("?", message)

    def reset_control(self):

        t0 = time.time()

        while time.time() - t0 < 10:
            x = self.quanser_plant.encoder_buffer[0].copy()
            control_action = self.pid_controller(x)
            control_action = np.clip(control_action, -2.5, 2.5)  # set an action range
            self.quanser_plant.write_analog_output(control_action)

            self.quanser_plant.get_encoder_readings()

            print("resetting.....")

        self.quanser_plant.write_analog_output(0)
        self.quanser_plant.normal_mode = True
        self.last_action = 0
        print("<========== resetting finished ==========>")

        if self.ep % self.params.control_params.calibrating_period == 0:
            self.calibration()

    def calibration(self):

        still_step = 0

        while True:

            t0 = time.time()

            print("calibrating...")

            _, x_dot, _, theta_dot, _ = self.quanser_plant.get_encoder_readings()

            still_step = still_step + 1 if x_dot == 0. and theta_dot == 0. else 0

            if still_step > 50:
                break

            dt = time.time() - t0

            time.sleep(self.sample_period - dt) if dt < self.sample_period else print("time_out")

        _, self.quanser_plant.theta_ini = self.quanser_plant.encoder_buffer.copy()
        self.quanser_plant.get_encoder_readings()
        print("<========= calibration done =========>")

    def receive_reset_command(self):
        """
        receive reset command from the cloud trainer to reset the plant;
        resetting command comes when the current steps reach the max_steps of a single episode
        """

        while True:
            _ = self.plant_reset_subscriber.parse_response()[2]
            self.quanser_plant.normal_mode = False
