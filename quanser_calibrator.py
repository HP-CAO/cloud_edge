import numpy as np
import math
import os
import argparse
from simple_pid import PID
from quanser.hardware import HIL, Clock, HILError


parser = argparse.ArgumentParser()
parser.add_argument('--kp', default=None, help='try different k gains for pid controller')
parser.add_argument('control', action='store_true', help='to test reset controller')
args = parser.parse_args()

if args.kp is not None:
    kp = float(args.kp)

else:
    kp = 30.0  # this value works quite well


def get_x_resolution():
    x_l = - 18825
    x_r = 16528
    x_res = 0.814 / (x_r - x_l)
    return x_res


def rescale_theta(theta_readings, theta_resolution):
    """
    rescale angle readings to [-pi to pi]
    :param x_readings: sensor reading from the angle encoder
    :return: pendulum's angle
    """
    theta_ini = 0
    theta = (theta_readings - theta_ini) * theta_resolution
    theta += -1 * math.pi
    theta_rescale = -1 * math.atan2(math.sin(theta), math.cos(theta))
    return theta_rescale


card = HIL("q2_usb", "0")
analog_channels = np.array([0], dtype=np.uint32)
encoder_channels = np.array([0, 1], dtype=np.int32)
num_analog_channels = len(analog_channels)
num_encoder_channels = len(encoder_channels)
frequency = 10.00  # increase periods
samples = np.iinfo(np.int32).max
samples_in_buffer = int(frequency)
samples_to_read = 1
samples_to_write = 1
analog_buffer = np.zeros(num_analog_channels, dtype=np.float64)
analog_write_buffer = np.zeros(num_analog_channels, dtype=np.float64)
encoder_buffer = np.zeros(num_encoder_channels, dtype=np.int32)

analog_task = card.task_create_analog_reader(samples_in_buffer, analog_channels, num_analog_channels)
encoder_task = card.task_create_encoder_reader(samples_in_buffer, encoder_channels, num_encoder_channels)
pid_controller = PID(kp, setpoint=0)
x_resolution = get_x_resolution()
theta_resolution = 0.00153

try:
    card.task_start(analog_task, Clock.HARDWARE_CLOCK_0, frequency, samples)
    card.task_start(encoder_task, Clock.HARDWARE_CLOCK_0, frequency, samples)
    i = 0

    while True:
        card.task_read_analog(analog_task, samples_to_read, analog_buffer)
        card.task_read_encoder(encoder_task, samples_to_read, encoder_buffer)
        x = encoder_buffer[0]
        theta = encoder_buffer[1]
        theta = rescale_theta(theta, theta_resolution)
        x = x * x_resolution
        print('Position and angle', x, theta)
        action = pid_controller(x)
        analog_write_buffer = np.array([action], dtype=np.float64)
        card.write_analog(analog_channels, num_analog_channels, analog_write_buffer)
        print("Encoder: ", encoder_buffer)
        print("analog_control", analog_write_buffer)


except HILError:
    print("HILError--")
    card.task_stop_all()
    card.task_delete_all()

card.close()

