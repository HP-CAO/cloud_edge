import numpy as np
from quanser.hardware import HIL, Clock, HILError
import math


class QuanserParams:
    def __init__(self):
        self.x_left = - 18825
        self.x_right = 16528
        self.x_length = 0.814
        self.x_center = 0


class QuanserPlant:
    def __init__(self, params: QuanserParams, sample_frequency, x_watchdog, theta_watchdog):
        # for simplicity, using immediate write and  immediate read interface
        self.params = params
        self.card = HIL("q2_usb", "0")
        self.analog_channels = np.array([0], dtype=np.uint32)
        self.encoder_channels = np.array([0, 1], dtype=np.int32)
        self.num_analog_channels = len(self.analog_channels)
        self.num_encoder_channels = len(self.encoder_channels)
        self.sample_period = 1 / sample_frequency
        self.analog_buffer = np.zeros(self.num_analog_channels, dtype=np.float64)
        self.encoder_buffer = np.zeros(self.num_encoder_channels, dtype=np.int32)

        self.normal_mode = True  # False if the pendulum is in the resetting phrase
        self.x_center = self.params.x_center
        self.x_resolution = self.get_x_resolution()
        self.theta_resolution = self.get_theta_resolution()
        self.x_threshold = x_watchdog
        self.theta_threshold = theta_watchdog
        self.x_center = 0
        self.theta_ini = 0
        # ditch default initialized reading if not start with [0, 0]
        self.card.read_encoder(self.encoder_channels, self.num_encoder_channels, self.encoder_buffer)
        print("Quanser Plant Initialized!")

    def get_encoder_readings(self):
        x_old, theta_old = self.encoder_buffer
        x_old_rescaled = self.rescale_x(x_old, self.x_center)

        self.card.read_encoder(self.encoder_channels, self.num_encoder_channels, self.encoder_buffer)
        # print("step_status", self.encoder_buffer)
        x_new, theta_new = self.encoder_buffer
        x_new_rescaled = self.rescale_x(x_new, self.x_center)
        theta_new_rescaled = self.rescale_theta(theta_new, self.theta_ini)

        x_dot = (x_new_rescaled - x_old_rescaled) / self.sample_period
        theta_dot = -1 * (theta_new - theta_old) * self.theta_resolution / self.sample_period

        print(theta_new, theta_old)

        failed = self.is_failed(x_new_rescaled, theta_dot)

        if failed:
            self.normal_mode = False
        print("failure", failed)
        print("States:", x_new_rescaled, x_dot, theta_new_rescaled, theta_dot, failed)
        return [x_new_rescaled, x_dot, theta_new_rescaled, theta_dot, failed]

    def write_analog_output(self, action):
        self.analog_buffer = np.array([action], np.float64)
        self.card.write_analog(self.analog_channels, self.num_analog_channels, self.analog_buffer)

    def rescale_x(self, x_readings, x_center):
        """
        rescale x_position sensor reading to world relative position with respect to track center
        :param x_readings: sensor reading from the cart encoder
        :return: cart position
        """
        x = (x_readings - x_center) * self.x_resolution
        return x

    def rescale_theta(self, theta_readings, theta_ini):
        """
        rescale angle readings to [-pi to pi]
        :param x_readings: sensor reading from the angle encoder
        :return: pendulum's angle
        """
        theta = (theta_readings - theta_ini) * self.theta_resolution
        theta += -1 * math.pi
        theta_rescale = -1 * math.atan2(math.sin(theta), math.cos(theta))
        return theta_rescale

    def get_theta_resolution(self):
        theta_0 = 0
        theta_1 = 4096
        theta_resolution = math.pi * 2 / (theta_1 - theta_0)
        return theta_resolution

    def get_x_resolution(self):
        x_l = self.params.x_left
        x_r = self.params.x_right
        x_resolution = self.params.x_length / (x_r - x_l)
        return x_resolution

    def is_failed(self, x, theta_dot):
        failed = bool(abs(x) >= self.x_threshold or theta_dot > self.theta_threshold)
        return failed
