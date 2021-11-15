import time
import signal
import sys
import numpy as np
from simple_pid import PID
from quanser.hardware import HILError

from realips.env.quanser_plant import QuanserParams, QuanserPlant
from realips.remote.edge_control import EdgeControlParams, EdgeControl
from realips.utils import states2observations


def signal_handler(signal, frame):
    global run
    print("Safe exiting")
    run = False


class QuanserEdgeControlParams(EdgeControlParams):
    def __init__(self):
        super().__init__()
        self.quanser_params = QuanserParams()


run = True
signal.signal(signal.SIGINT, signal_handler)


class QuanserEdgeControl(EdgeControl):

    def __init__(self, params: QuanserEdgeControlParams, run_eval):
        super().__init__(params, run_eval)
        self.steps_since_calibration = 0
        self.params = params
        self.quanser_plant = QuanserPlant(self.params.quanser_params,
                                          self.params.control_params.frequency,
                                          self.params.control_params.x_threshold,
                                          self.params.control_params.theta_dot_threshold)

        self.pid_controller = PID(Kp=0.0005, setpoint=0, sample_time=self.sample_period)

        self.calibration()
        self.initialize_plant()

    def generate_action(self):

        while True:

            if self.params.ddpg_params.add_actions_observations:

                action_observations = np.zeros(shape=self.params.ddpg_params.action_observations_dim)
            else:
                action_observations = []

            while not self.quanser_plant.normal_mode:
                self.reset_control()

            t0 = time.perf_counter()
            time_out_counter = 0

            while self.quanser_plant.normal_mode:

                self.step += 1
                self.steps_since_calibration += 1

                states = self.quanser_plant.get_encoder_readings()

                normal_mode = self.quanser_plant.normal_mode

                stats_observation, failed = states2observations(states)

                observations = np.hstack((stats_observation, action_observations)).tolist()

                agent = self.agent_a if self.agent_a_active else self.agent_b

                if self.training:
                    action = agent.get_exploration_action(observations, self.control_targets)
                else:
                    action = agent.get_exploitation_action(observations, self.control_targets)

                action_real = action * self.params.control_params.action_factor

                self.quanser_plant.write_analog_output(action_real)

                self.edge_trajectory = [observations, self.last_action, failed, normal_mode, self.step]

                self.last_action = action

                if self.params.ddpg_params.add_actions_observations:
                    action_observations = np.append(action_observations, action)[1:]

                if self.trajectory_sending_condition.acquire(False):
                    self.trajectory_sending_condition.notify_all()
                    self.trajectory_sending_condition.release()

                one_loop_time = time.perf_counter() - t0

                if one_loop_time < self.sample_period:
                    time.sleep(self.sample_period - one_loop_time)
                    # asyncio.sleep(self.sample_period - one_loop_time) # todo double check
                    time_out_counter = 0
                else:
                    time_out_counter += 1

                if time_out_counter >= 10:
                    t0 = time.perf_counter()
                    print("TIMEOUT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    time_out_counter = 0
                else:
                    t0 = t0 + self.sample_period

                if run is False:
                    self.quanser_plant.write_analog_output(0)
                    sys.exit("Safe exiting...")

    def reset_control(self):

        if self.steps_since_calibration >= self.params.control_params.calibrating_period_steps:
            self.calibration()

        t0 = time.perf_counter()

        if self.params.control_params.random_reset_ini:
            reset_point \
                = np.random.uniform(-0.8 * self.params.control_params.x_threshold / self.quanser_plant.x_resolution,
                                    0.8 * self.params.control_params.x_threshold / self.quanser_plant.x_resolution)

            self.pid_controller.setpoint = reset_point

        print("resetting.....")

        while time.perf_counter() - t0 < 5:
            x = self.quanser_plant.encoder_buffer[0].copy()
            control_action = self.pid_controller(x)
            control_action = np.clip(control_action, -2.5, 2.5)  # set an action range
            self.quanser_plant.write_analog_output(control_action)
            self.quanser_plant.get_encoder_readings()

        self.quanser_plant.write_analog_output(0)
        self.quanser_plant.normal_mode = True
        self.last_action = 0
        print("<========== resetting finished ==========>")

    def calibration(self):
        
        self.quanser_plant.write_analog_output(0)
        still_step = 0
        x_old = theta_old = 0
        while True:

            t0 = time.perf_counter()

            print("calibrating...")

            x, _, theta, _, _ = self.quanser_plant.get_encoder_readings()

            still_step = still_step + 1 if x == x_old and theta == theta_old else 0

            x_old, theta_old = x, theta

            if still_step > 50:
                break

            dt = time.perf_counter() - t0

            time.sleep(self.sample_period - dt) if dt < self.sample_period else print("time_out")

        _, self.quanser_plant.theta_ini = self.quanser_plant.encoder_buffer.copy()
        self.quanser_plant.get_encoder_readings()
        self.steps_since_calibration = 0
        print("<========= calibration done =========>")

    def set_normal_mode(self, normal_mode):
        self.quanser_plant.normal_mode = False
