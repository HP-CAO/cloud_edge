import copy
import pickle
import random
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
from realips.remote.edge_control import EdgeControlParams, EdgeControl
from realips.utils import get_current_time
from realips.remote.redis import RedisParams, RedisConnection
from realips.agent.ddpg import DDPGAgent, DDPGAgentParams
from realips.utils import states2observations


class QuanserEdgeControlParams(EdgeControlParams):
    def __init__(self):
        super().__init__()
        self.quanser_params = QuanserParams()


class QuanserEdgeControl(EdgeControl):
    def __init__(self, params: QuanserEdgeControlParams, run_eval):
        super().__init__(params, run_eval)
        self.params = params
        self.quanser_plant = QuanserPlant(self.params.quanser_params,
                                          self.params.control_params.frequency,
                                          self.params.control_params.x_threshold,
                                          self.params.control_params.theta_dot_threshold)
        self.pid_controller = PID(Kp=0.0005, setpoint=0, sample_time=self.sample_period)

        self.initialize_plant()
        # self.calibration()

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
                    time_out_counter = 0
                else:
                    t0 = t0 + self.sample_period

    def reset_control(self):

        t0 = time.time()

        if self.params.control_params.random_reset_ini:
            reset_point \
                = np.random.uniform(-0.8 * self.params.control_params.x_threshold / self.quanser_plant.x_resolution,
                                    0.8 * self.params.control_params.x_threshold / self.quanser_plant.x_resolution)

            self.pid_controller.setpoint = reset_point

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
        x_old = theta_old = 0
        while True:

            t0 = time.time()

            print("calibrating...")

            x, _, theta, _, _ = self.quanser_plant.get_encoder_readings()

            still_step = still_step + 1 if x == x_old and theta == theta_old else 0

            x_old, theta_old = x, theta

            if still_step > 50:
                break

            dt = time.time() - t0

            time.sleep(self.sample_period - dt) if dt < self.sample_period else print("time_out")

        _, self.quanser_plant.theta_ini = self.quanser_plant.encoder_buffer.copy()
        self.quanser_plant.get_encoder_readings()
        print("<========= calibration done =========>")

    def set_normal_mode(self, normal_mode):
        self.quanser_plant.normal_mode = False
