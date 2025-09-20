import time

import numpy as np

from realips.env.gym_physics import GymPhysicsParams, GymPhysics
from realips.remote.edge_control_safe import EdgeControlParams, EdgeControl
from realips.agent.ddpg import DDPGAgentParams, DDPGAgent
from realips.utils import states2observations
from realips.agent.AutoSafe import ModelbasedAgent


class SimEdgeControlParams(EdgeControlParams):
    def __init__(self):
        super().__init__()
        self.physics_params = GymPhysicsParams()


class SimEdgeControl(EdgeControl):
    def __init__(self, params: SimEdgeControlParams):
        super().__init__(params)
        self.params = params
        self.physics = GymPhysics(self.params.physics_params)
        self.physics.random_reset()
        self.normal_mode = True
        self.ep = 0
        self.safe_policy = ModelbasedAgent()


    def generate_action(self):

        while True:

            self.ep += 1

            if self.params.agent_params.add_actions_observations:
                action_observations = np.zeros(shape=self.params.agent_params.action_observations_dim)
            else:
                action_observations = []

            while not self.normal_mode:
                self.reset_control()

            t0 = time.time()
            time_out_counter = 0
            while self.normal_mode:

                self.step += 1

                states = self.physics.states

                normal_mode = self.normal_mode

                stats_observation, failed = states2observations(states)

                tracking_error = np.array(states[:4]) - self.model_based_equilibrium

                observations = np.hstack([stats_observation, action_observations, tracking_error]).tolist()

                agent = self.agent_a if self.agent_a_active else self.agent_b

                if self.training:
                    action = agent.get_action(observations, mode='train').squeeze()
                else:
                    action = agent.get_action(observations, mode='test').squeeze()

                self.physics.step(action)

                edge_trajectory = [observations, self.last_action, failed, normal_mode, self.step]

                self.last_action = action

                self.send_edge_trajectory(edge_trajectory)  # this is sent to the cloud trainer

                if self.params.agent_params.add_actions_observations:
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

    def reset_control(self):

        self.physics.random_reset()
        # self.physics.reset()

        time.sleep(5)
        self.normal_mode = True
        self.last_action = 0

    def set_normal_mode(self, normal_mode):
        self.normal_mode = False
