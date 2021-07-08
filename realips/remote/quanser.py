import numpy as np
from quanser.hardware import HIL, Clock, HILError

class QuanserParams:
    def __init__(self):
        self.frequency = 50.00 # hz
        self.x_limit = 0.5
        self.theta_dot_limit = 2


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

        self.analog_write_task = self.card.task_create_analog_writer(self.samples_in_buffer,
                                                                     self.analog_channels,
                                                                     self.num_analog_channels)

        self.analog_read_task = self.card.task_create_analog_reader(self.samples_in_buffer,
                                                                    self.analog_channels,
                                                                    self.num_analog_channels)

        self.normal_mode = True  # False if the pendulum is in the resetting phrase

        print("Quanser Plant Initialized!")

    def start_task(self):
        self.card.task_start(
            self.analog_write_task, Clock.HARDWARE_CLOCK_0, self.params.frequency, self.num_samples_max)
        self.card.task_start(
            self.encoder_read_task, Clock.HARDWARE_CLOCK_0, self.params.frequency, self.num_samples_max)
        self.card.task_start(
            self.analog_read_task, Clock.HARDWARE_CLOCK_0, self.params.frequency, self.num_samples_max)

    def get_encoder_readings(self):
        x_old, theta_old = self.encoder_buffer
        x_old_rescaled = self.rescale_x(x_old)
        theta_old_rescaled = self.rescale_theta(theta_old)

        self.card.task_read_analog(self.encoder_read_task, self.num_samples_to_read, self.analog_buffer)
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

    @staticmethod
    def rescale_x(x):
        pass

    @staticmethod
    def rescale_theta(theta):
        pass

    @staticmethod
    def is_failed(x, theta_dot):
        pass

