import pickle
import copy
import struct
import math
import threading
import time

import numpy as np
from realips.agent.td3 import TD3Agent, TD3AgentParams
from realips.remote.transition import TrajectorySegment
from realips.trainer.trainer_td3 import TD3TrainerParams, TD3Trainer
from realips.utils import get_current_time, states2observations
from realips.system.ips import IpsSystemParams, IpsSystem
from realips.remote.redis import RedisParams, RedisConnection
from realips.agent.ddpg import DDPGAgent, DDPGAgentParams
from realips.trainer.trainer_ddpg import DDPGTrainer, DDPGTrainerParams


class CloudParams:
    def __init__(self):
        self.on_target_reset_steps = 100  # num steps on target after which the episode is terminated
        self.sleep_after_reset = 2  # seconds of sleep because it makes sense
        self.agent_type = 0  # 0: DDPG, 1: TD3
        self.pre_fill_steps = 0


class CloudSystemParams(IpsSystemParams):
    def __init__(self):
        super().__init__()
        self.redis_params = RedisParams()
        self.cloud_params = CloudParams()
        self.agent_params = DDPGAgentParams() if self.cloud_params.agent_type == 0 else TD3AgentParams()
        self.trainer_params = DDPGTrainerParams() if self.cloud_params.agent_type == 0 else TD3TrainerParams()


class CloudSystem(IpsSystem):
    def __init__(self, params: CloudSystemParams):
        super().__init__(params)
        self.params = params
        self.redis_connection = RedisConnection(self.params.redis_params)
        self.trajectory_subscriber = self.redis_connection.subscribe(
            channel=self.params.redis_params.ch_plant_trajectory_segment)
        self.edge_trajectory_subscriber = self.redis_connection.subscribe(
            channel=self.params.redis_params.ch_edge_trajectory)
        self.ep = 0

        if self.params.agent_params.add_actions_observations:
            self.shape_observations += self.params.agent_params.action_observations_dim

        if self.params.cloud_params.agent_type == 0:
            self.agent = DDPGAgent(self.params.agent_params, self.shape_observations, self.shape_targets,
                                   shape_action=1)
            self.agent.initial_model()
            self.trainer = DDPGTrainer(self.params.trainer_params, self.agent)
        elif self.params.cloud_params.agent_type == 1:
            self.agent = TD3Agent(self.params.agent_params, self.shape_observations, self.shape_targets, action_shape=1)
            self.agent.initial_model()
            self.trainer = TD3Trainer(self.params.trainer_params, self.agent)

        self.edge_status_subscriber = self.redis_connection.subscribe(
            channel=self.params.redis_params.ch_edge_ready_update)
        self.edge_ready = None

        # self.t1 = threading.Thread(target=self.store_trajectory)
        self.t2 = threading.Thread(target=self.optimize)
        self.t3 = threading.Thread(target=self.waiting_edge_ready)

        self.optimize_condition = threading.Condition()

        if self.params.stats_params.weights_path is not None:
            self.agent.load_weights(self.params.stats_params.weights_path)

        if self.params.cloud_params.pre_fill_steps > 0:
            self.prefill_sim(self.params.cloud_params.pre_fill_steps)

        self.send_weights_and_noise_factor(self.agent.get_actor_weights(), self.agent.action_noise_factor)
        self.target = [0., 0.]

    def run(self):
        """It's triple threads"""
        # self.t1.start()
        self.t2.start()
        self.t3.start()
        self.store_trajectory()

    def store_trajectory(self):

        best_dsas = 0.0  # Best distance score and survived
        moving_average_dsas = 0.0

        while self.model_stats.total_steps < self.model_stats.params.total_steps:

            self.model_stats.reset_status()
            self.ep += 1

            if self.ep % self.params.stats_params.eval_period == 0:
                self.redis_connection.publish(self.params.redis_params.ch_edge_mode, struct.pack("?", False))
                training = False
                print("EVALUATING!!!!!!")
            else:
                self.redis_connection.publish(self.params.redis_params.ch_edge_mode, struct.pack("?", True))
                training = True

            dsas = self.run_episode(training)
            self.agent.save_weights(self.params.stats_params.model_name)

            if not training:
                moving_average_dsas = 0.95 * moving_average_dsas + 0.05 * dsas

                if moving_average_dsas > best_dsas:
                    self.agent.save_weights(self.params.stats_params.model_name + '_best')
                    best_dsas = moving_average_dsas

    def run_episode(self, training):

        traj_segment = self.receive_edge_trajectory()

        step_count = self.params.stats_params.max_episode_steps
        step = 0
        on_target_steps = 0
        for step in range(step_count):
            last_seg = traj_segment
            traj_segment = self.receive_edge_trajectory()
            stat_observations = last_seg.observations[0:5]

            r = self.reward_fcn.reward(stat_observations,
                                       self.target,
                                       traj_segment.last_action,
                                       traj_segment.failed, pole_length=self.params.physics_params.length).squeeze()

            if training:
                if traj_segment.sequence_number == (last_seg.sequence_number + 1):
                    # only save the experience if the two trajectory are consecutive
                    self.trainer.store_experience(last_seg.observations, self.target, traj_segment.last_action, r,
                                                  traj_segment.observations, traj_segment.failed)

                    if self.optimize_condition.acquire(False):
                        self.optimize_condition.notify_all()
                        self.optimize_condition.release()
                else:
                    print("Package loss... terrible loss :/")
            else:

                self.model_stats.cart_positions.append(last_seg.observations[0])
                self.model_stats.pendulum_angele.append(
                    math.atan2(last_seg.observations[2], last_seg.observations[3]))
                self.model_stats.actions.append(traj_segment.last_action)

            self.model_stats.observations = copy.deepcopy(traj_segment.observations[0:5])
            self.model_stats.targets = copy.deepcopy(self.target)
            self.model_stats.measure(self.model_stats.observations, self.model_stats.targets,
                                     traj_segment.failed,
                                     pole_length=self.params.physics_params.length,
                                     distance_score_factor=self.params.reward_params.distance_score_factor)

            self.model_stats.reward.append(r)

            if training and self.model_stats.consecutive_on_target_steps > self.params.cloud_params.on_target_reset_steps:
                break

            if not traj_segment.normal_operation:
                break

        self.agent.noise_factor_decay(self.model_stats.total_steps)

        self.initiate_reset()
        time.sleep(self.params.cloud_params.sleep_after_reset)
        # Clean the received trajectory
        stale_segments = self.edge_trajectory_subscriber.parse_response(block=False)
        while stale_segments is not None:
            stale_segments = self.edge_trajectory_subscriber.parse_response(block=False)

        if training:
            self.model_stats.add_steps(step)
            self.model_stats.training_monitor(self.ep)
        else:
            # self.model_stats.add_steps(1)  # Add one step so that it doesn't overlap with possible previous evals
            self.model_stats.evaluation_monitor_scalar(self.ep)
            self.model_stats.evaluation_monitor_image(self.ep)

        dsas = float(self.model_stats.survived) * self.model_stats.get_average_distance_score()
        # self.agent.save_weights(self.params.stats_params.model_name + '_' + str(ep))
        return dsas

    def optimize(self):

        while True:

            self.optimize_condition.acquire()
            self.optimize_condition.wait()

            self.trainer.optimize()

            if self.edge_ready:
                weights = self.agent.get_actor_weights()
                self.edge_ready = False
                self.send_weights_and_noise_factor(weights, self.agent.action_noise_factor)
                # self.agent.save_weights(self.params.stats_params.model_name)
                print("[{}] ===>  Training: current training finished, sending weights, mem_size: {}"
                      .format(get_current_time(), self.trainer.replay_mem.get_size()))

    def waiting_edge_ready(self):

        while True:
            edge_status = self.receive_edge_status()
            self.edge_ready = edge_status
            print("Edge is ready for update", self.edge_ready)

    def receive_edge_status(self):
        edge_status_pack = self.edge_status_subscriber.parse_response()
        edge_status = pickle.loads(edge_status_pack[2])
        return edge_status

    def receive_trajectory_segment(self):
        trajectory_pack = self.trajectory_subscriber.parse_response()[2]
        seg = TrajectorySegment.from_packet(trajectory_pack)
        return seg

    def send_weights_and_noise_factor(self, weights, noise_factor):
        weights_and_noise_pack = pickle.dumps([weights, noise_factor])
        self.redis_connection.publish(channel=self.params.redis_params.ch_edge_weights, message=weights_and_noise_pack)

    def receive_edge_trajectory(self):
        edge_trajectory_pack = self.edge_trajectory_subscriber.parse_response()[2]
        edge_seg = TrajectorySegment.pickle_load_pack(edge_trajectory_pack)
        return edge_seg

    def initiate_reset(self):
        reset_pack = pickle.dumps(True)
        self.redis_connection.publish(channel=self.params.redis_params.ch_plant_reset, message=reset_pack)

    def prefill_sim(self, pre_fill_steps):
        steps = 0
        while steps < pre_fill_steps:

            self.model_stats.init_episode()

            if self.params.agent_params.add_actions_observations:
                action_observations = np.zeros(shape=self.params.agent_params.action_observations_dim)
            else:
                action_observations = []

            step = 0
            for step in range(self.params.stats_params.max_episode_steps):

                observations = np.hstack((self.model_stats.observations, action_observations)).tolist()

                action = self.agent.get_exploration_action(observations, self.model_stats.targets)

                if self.params.agent_params.add_actions_observations:
                    action_observations = np.append(action_observations, action)[1:]

                states_next = self.physics.step(action)

                stats_observations_next, failed = states2observations(states_next)

                observations_next = np.hstack((stats_observations_next, action_observations)).tolist()

                r = self.reward_fcn.reward(self.model_stats.observations, self.model_stats.targets, action, failed,
                                           pole_length=self.params.physics_params.length)

                self.trainer.store_experience(observations, self.model_stats.targets, action, r,
                                              observations_next, failed)

                self.model_stats.observations = copy.deepcopy(stats_observations_next)

                if failed:
                    break
            steps += step
