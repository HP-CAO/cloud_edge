import numpy as np
from quanser.hardware import HIL, Clock, HILError
import math

class QuanserParams:
    def __init__(self):
        self.frequency = 50.00 # hz
        self.theta_dot_limit = 2
        self.x_threshold = 0.55
        self.theta_dot_threshold = 15
        self.x_left = -23749
        self.x_right = 11946
        self.x_length = 0.814


class QuanserPlant:
    def __init__(self, params: QuanserParams):
        self.params = params
        self.card = HIL("q2_usb", "0")
        self.analog_channels = np.array([0], dtype=np.uint32)
        self.encoder_channels = np.array([0, 1], dtype=np.uint32)
        self.num_analog_channels = len(self.analog_channels)
        self.num_encoder_channels = len(self.encoder_channels)
        self.num_samples_max = np.iinfo(np.int32).max
        self.samples_in_buffer = int(self.params.frequency)
        self.num_samples_to_read = 1
        self.num_samples_to_write = 1

        self.sample_period = 1 / self.params.frequency

        self.analog_buffer = np.zeros(self.num_analog_channels, dtype=np.float32)
        self.encoder_buffer = np.zeros(self.num_encoder_channels, dtype=np.float32)
        self.analog_write_buffer = np.zeros(self.num_analog_channels, dtype=np.float32)

        self.encoder_read_task = self.card.task_create_encoder_reader(self.samples_in_buffer,
                                                                       self.encoder_channels,
                                                                       self.num_encoder_channels)

        # self.analog_write_task = self.card.task_create_analog_writer(self.samples_in_buffer,
        #                                                              self.analog_channels,
        #                                                              self.num_analog_channels)

        self.analog_read_task = self.card.task_create_analog_reader(self.samples_in_buffer,
                                                                    self.analog_channels,
                                                                    self.num_analog_channels)

        self.normal_mode = True  # False if the pendulum is in the resetting phrase

        self.x_center = self.get_center_x()
        self.x_resolution = self.get_x_resolution()
        self.theta_resolution = self.get_theta_resolution()

        try:
            self.start_task()
            print("Quanser Plant Initialized!")

        except HILError:
            print("HILError--")
            self.card.task_stop_all()
            self.card.task_stop_all()

    def start_task(self):
        # self.card.task_start(
        #     self.analog_write_task, Clock.HARDWARE_CLOCK_0, self.params.frequency, self.num_samples_max)
        self.card.task_start(
            self.encoder_read_task, Clock.HARDWARE_CLOCK_0, self.params.frequency, self.num_samples_max)
        self.card.task_start(
            self.analog_read_task, Clock.HARDWARE_CLOCK_0, self.params.frequency, self.num_samples_max)

    def get_encoder_readings(self):
        x_old, theta_old = self.encoder_buffer
        x_old_rescaled = self.rescale_x(x_old)
        theta_old_rescaled = self.rescale_theta(theta_old)

        self.card.task_read_encoder(self.encoder_read_task, self.num_samples_to_read, self.encoder_buffer)
        x_new, theta_new = self.encoder_buffer
        x_new_rescaled = self.rescale_x(x_new)
        theta_new_rescaled = self.rescale_theta(theta_new)

        x_dot = (x_new_rescaled - x_old_rescaled) / self.sample_period
        theta_dot = (theta_new_rescaled - theta_old_rescaled) / self.sample_period

        failed = self.is_failed(x_new_rescaled, theta_dot)

        if failed:
            self.normal_mode = False

        return [x_new_rescaled, x_dot, theta_new_rescaled, theta_dot, failed]

    def write_analog_output(self, action):
        self.analog_write_buffer = np.array(action)
        self.card.write_analog(self.analog_channels, self.num_analog_channels, self.analog_write_buffer)

    def rescale_x(self, x_readings):
        """
        rescale x_position sensor reading to world relative position with respect to track center
        :param x_readings: sensor reading from the cart encoder
        :return: cart position
        """
        x = (x_readings - self.x_center) * self.x_resolution
        return x

    def rescale_theta(self, theta_readings):
        """
        rescale angle readings to [-pi to pi]
        :param x_readings: sensor reading from the angle encoder
        :return: pendulum's angle
        """
        theta_ini = 2
        theta = (theta_readings - theta_ini) * self.theta_resolution
        theta += -1 * math.pi
        theta_rescale = math.atan2(math.sin(theta), math.cos(theta))
        return theta_rescale

    def get_theta_resolution(self):
        theta_0 = 2
        theta_1 = 4094
        theta_resolution = math.pi * 2 / (theta_1 - theta_0)
        return theta_resolution

    def get_x_resolution(self):
        x_l = self.params.x_left
        x_r = self.params.x_right
        x_resolution = self.params.x_length / (x_r - x_l)
        return x_resolution

    def get_center_x(self):
        x_c = (self.params.x_right - self.params.x_left) * 0.5
        return x_c

    def is_failed(self, x, theta_dot):
        failed = bool(x <= -self.params.x_threshold
                      or x >= self.params.x_threshold
                      or theta_dot > self.params.theta_dot_threshold)
        return failed

