import time
import signal
import sys
import numpy as np
import math
import os
import argparse
from simple_pid import PID
from datetime import datetime
from realips.env.quanser_plant import QuanserPlant, QuanserParams
from quanser.hardware import HIL, Clock, HILError


parser = argparse.ArgumentParser()
parser.add_argument('--kp', default=None, help='try different k gains for pid controller')
parser.add_argument('control', action='store_true', help='to test reset controller')
args = parser.parse_args()

if args.kp is not None:
    kp = float(args.kp)

else:
    kp = 0.0005  # this value works quite well


def signal_handler(signal, frame):
    global run
    print("Safe exiting")
    run = False


def get_current_time():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    return current_time


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
period_time = 1 / frequency
samples = np.iinfo(np.int32).max

analog_buffer = np.zeros(num_analog_channels, dtype=np.float64)
analog_write_buffer = np.zeros(num_analog_channels, dtype=np.float64)
encoder_buffer = np.zeros(num_encoder_channels, dtype=np.int32)


pid_controller = PID(kp, setpoint=0)
x_resolution = get_x_resolution()
theta_resolution = 0.00153398

run = True
signal.signal(signal.SIGINT, signal_handler)
quanser_plant = QuanserPlant(QuanserParams(), sample_frequency=30, x_watchdog=0.3, theta_watchdog=15)


def reset_control():
    t0 = time.time()

    while True:

        x = quanser_plant.encoder_buffer[0].copy()
        control_action = pid_controller(x)
        control_action = np.clip(control_action, -2.5, 2.5)  # set an action range
        quanser_plant.write_analog_output(control_action)

        if time.time() - t0 > 10:  # simply setting time threshold for resetting control
            break

        quanser_plant.get_encoder_readings()

        print("resetting.....")


def re_calibration():

    print("calibrating...")

    _, x_dot, _, theta_dot, _ = quanser_plant.get_encoder_readings()

    if x_dot == 0 and theta_dot == 0:
        time.sleep(5)
        quanser_plant.x_center, quanser_plant.theta_ini = quanser_plant.encoder_buffer.copy()
        print(quanser_plant.x_center, quanser_plant.theta_ini)

    quanser_plant.write_analog_output(0)
    quanser_plant.normal_mode = True
    print("<==========resetting finished==========>")