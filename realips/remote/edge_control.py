import pickle
import threading
import struct
import numpy as np
from simple_pid import PID
from quanser.hardware import HILError
from realips.env.quanser_plant import QuanserParams, QuanserPlant
from realips.utils import get_current_time
from realips.remote.redis import RedisParams, RedisConnection
from realips.agent.ddpg import DDPGAgent, DDPGAgentParams
from realips.utils import states2observations


class ControlParams:
    def __init__(self):
        self.random_reset_target = False
        self.x_boundary = 0.4
        self.control_targets = [0., 0.]
        self.is_quick_recover = True
        self.agent_type = None
        self.initialize_from_cloud = True
        self.train_real = True
        self.action_factor = 3


class EdgeControlParams:
    def __init__(self):
        self.redis_params = RedisParams()
        self.ddpg_params = DDPGAgentParams()
        self.control_params = ControlParams()
        self.quanser_params = QuanserParams()


class EdgeControl:
    def __init__(self, params: EdgeControlParams):
        self.params = params
        self.redis_connection = RedisConnection(self.params.redis_params)

        self.weights_subscriber = self.redis_connection.subscribe(channel=self.params.redis_params.ch_edge_weights)
        self.training_mode_subscriber = self.redis_connection.subscribe(channel=self.params.redis_params.ch_edge_mode)
        self.runnint_reset_subscriber = self.redis_connection.subscribe(channel=self.params.redis_params.ch_plant_reset)
        self.control_targets = self.params.control_params.control_targets
        self.active_agent = True  # True: agent_a is controller, False: agent_b is controller
        self.quanser_plant = QuanserPlant(self.params.quanser_params)

    def reset_targets(self):
        if self.params.control_params.random_reset_target:
            x_target = np.random.uniform(-self.params.control_params.x_boundary,
                                         self.params.control_params.x_boundary) * 0.5
            self.control_targets = [x_target, 0.]
        else:
            self.control_targets = self.params.control_params.control_targets

    def receives_weights(self):
        weights_pack = self.weights_subscriber.parse_response()[2]
        weights = pickle.loads(weights_pack)
        return weights

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

    def initialize_weights_from_cloud(self, *args):
        weights = self.receives_weights()
        for agent in args:
            agent.set_actor_weights(weights)


class DDPGEdgeControl(EdgeControl):
    def __init__(self, params: EdgeControlParams, eval=None):
        super().__init__(params)
        self.params = params
        self.shape_observations = 5
        if self.params.ddpg_params.add_actions_observations:
            self.shape_observations += self.params.ddpg_params.action_observations_dim
        self.agent_a = DDPGAgent(self.params.ddpg_params, shape_observations=self.shape_observations, on_edge=True)
        self.agent_b = DDPGAgent(self.params.ddpg_params, shape_observations=self.shape_observations, on_edge=True)
        # self.t1 = threading.Thread(target=self.generate_action)
        self.t2 = threading.Thread(target=self.update_weights)
        self.t3 = threading.Thread(target=self.receive_mode)
        self.t4 = threading.Thread(target=self.receive_reset_command())
        self.step = 0
        self.training = True if eval is None else False
        self.pid_controller = PID(Kp=1.0, setpoint=0, sample_time=0.02)

        if eval is not None:
            self.agent_a.load_weights(eval)
            self.agent_b.load_weights(eval)
        elif params.control_params.initialize_from_cloud:
            print("waiting for weights from cloud")
            self.initialize_weights_from_cloud(self.agent_a, self.agent_b)

    def generate_action(self):

        # try:

        self.quanser_plant.start_task()

        print("Quanser Plant Initialized!")

        while True:

            if self.params.ddpg_params.add_actions_observations:

                action_observations = np.zeros(shape=self.params.ddpg_params.action_observations_dim)
            else:
                action_observations = []

            while not self.quanser_plant.normal_mode:
                self.reset_control()

            while self.quanser_plant.normal_mode:

                self.step += 1
                # t0 = time.time()
                states = self.quanser_plant.get_encoder_readings()

                self.send_plant_trajectory(states)  # this is sent to the plant scope for monitoring

                normal_mode = self.quanser_plant.normal_mode

                last_action = self.quanser_plant.analog_write_buffer.item()

                stats_observation, failed = states2observations(states)

                observations = np.hstack((stats_observation, action_observations)).tolist()

                agent = self.agent_a if self.active_agent else self.agent_b

                if self.training:
                    action = agent.get_exploration_action(observations, self.control_targets)
                else:
                    action = agent.get_exploitation_action(observations, self.control_targets)

                # delta_t = time.time()-t0

                action = action * self.params.control_params.action_factor

                print("normal_mode: ", self.quanser_plant.normal_mode)

                self.quanser_plant.write_analog_output(action)

                edge_trajectory = [observations, last_action, failed, normal_mode]

                self.send_edge_trajectory(edge_trajectory)  # this is sent to the cloud trainer
                # print("Inference took {}s".format(delta_t))
                self.action_noise_decay()

                if self.params.ddpg_params.add_actions_observations:
                    action_observations = np.append(action_observations, action)[1:]

                if failed:
                    self.quanser_plant.normal_mode = False
                    break

        # except HILError:
        #     print("HILError--")
        #     self.quanser_plant.card.task_stop_all()
        #     self.quanser_plant.card.task_stop_all()

    def update_weights(self):

        while True:

            self.send_ready_update(True)

            weights = self.receives_weights()

            if self.active_agent:
                self.agent_b.set_actor_weights(weights)
            else:
                self.agent_a.set_actor_weights(weights)

            self.active_agent = not self.active_agent

            # print("[{}] ===> Agents toggled".format(get_current_time()))

    def action_noise_decay(self):
        self.agent_a.noise_factor_decay(self.step)
        self.agent_b.noise_factor_decay(self.step)

    def run(self):
        self.t2.start()
        self.t3.start()
        self.t4.start()
        self.generate_action()

    def receive_mode(self):
        """
        receive_mode to switch between training and testing
        """
        message = self.training_mode_subscriber.parse_response()[2]
        self.training = struct.unpack("?", message)

    def reset_control(self):
        position_counter = 0
        while True:
            print("resetting...")
            states = self.quanser_plant.get_encoder_readings()
            x_ = states[0]
            if x_ <= 0.01:
                position_counter += 1
            else:
                position_counter = 0
            if position_counter >= 5:
                break
            control_action = self.pid_controller(x_)
            self.quanser_plant.write_analog_output(control_action)
        self.quanser_plant.normal_mode = True

    def receive_reset_command(self):
        """
        receive reset command from the cloud trainer to reset the plant;
        resetting command comes when the current steps reach the max_steps of a single episode
        """
        message = self.training_mode_subscriber.parse_response()[2]
        self.reset = struct.unpack("?", message)