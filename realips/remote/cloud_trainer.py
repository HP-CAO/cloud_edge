import pickle
import copy
import struct
import math
import threading
from realips.remote.transition import TrajectorySegment
from realips.utils import get_current_time
from realips.system.ips import IpsSystemParams, IpsSystem
from realips.remote.redis import RedisParams, RedisConnection
from realips.agent.ddpg import DDPGAgent, DDPGAgentParams
from realips.trainer.trainer_ddpg import DDPGTrainer, DDPGTrainerParams


class CloudTrainerParams(IpsSystemParams):
    def __init__(self):
        super().__init__()
        self.redis_params = RedisParams()


class CloudTrainer(IpsSystem):
    def __init__(self, params: CloudTrainerParams):
        super().__init__(params)
        self.params = params
        self.redis_connection = RedisConnection(self.params.redis_params)
        self.trajectory_subscriber = self.redis_connection.subscribe(
            channel=self.params.redis_params.ch_plant_trajectory_segment)
        self.edge_trajectory_subscriber = self.redis_connection.subscribe(
            channel=self.params.redis_params.ch_edge_trajectory)
        self.ep = 0

    def receive_trajectory_segment(self):
        trajectory_pack = self.trajectory_subscriber.parse_response()[2]
        seg = TrajectorySegment.from_packet(trajectory_pack)
        return seg

    def send_weights(self, weights):
        weights_pack = pickle.dumps(weights)
        self.redis_connection.publish(channel=self.params.redis_params.ch_edge_weights, message=weights_pack)

    def receive_edge_trajectory(self):
        edge_trajectory_pack = self.edge_trajectory_subscriber.parse_response()[2]
        edge_seg = TrajectorySegment.pickle_load_pack(edge_trajectory_pack)
        return edge_seg


class CloudTrainerDDPGParams(CloudTrainerParams):
    def __init__(self):
        super().__init__()
        self.agent_params = DDPGAgentParams()
        self.trainer_params = DDPGTrainerParams()


class CloudTrainerDDPG(CloudTrainer):
    def __init__(self, params: CloudTrainerDDPGParams):
        super().__init__(params)
        self.params = params
        if self.params.agent_params.add_actions_observations:
            self.shape_observations += self.params.agent_params.action_observations_dim
        self.agent = DDPGAgent(self.params.agent_params, self.shape_observations, self.shape_targets, shape_action=1)
        self.agent.initial_model()
        self.trainer = DDPGTrainer(self.params.trainer_params, self.agent)
        self.edge_status_subscriber = self.redis_connection.subscribe(
            channel=self.params.redis_params.ch_edge_ready_update)
        self.edge_ready = None

        # self.t1 = threading.Thread(target=self.store_trajectory)
        self.t2 = threading.Thread(target=self.optimize_ddpg)
        self.t3 = threading.Thread(target=self.waiting_edge_ready)

        self.optimize_condition = threading.Condition()

        if self.params.stats_params.weights_path is not None:
            self.agent.load_weights(self.params.stats_params.weights_path)

        self.send_weights(self.agent.get_actor_weights())
        self.target = [0., 0.]

    def store_trajectory(self):

        best_dsas = 0.0  # Best distance score and survived
        moving_average_dsas = 0.0

        while True:
            self.model_stats.reset_status()
            self.ep += 1
            step = 0

            if self.ep % self.params.stats_params.eval_period == 0:
                self.redis_connection.publish(self.params.redis_params.ch_edge_mode, struct.pack("?", False))
                training = False
                print("EVALUATING!!!!!!")
            else:
                self.redis_connection.publish(self.params.redis_params.ch_edge_mode, struct.pack("?", True))
                training = True

            # Ditch the safety active data
            traj_segment = self.receive_edge_trajectory()
            # while not traj_segment.normal_operation:
            #     traj_segment = self.receive_edge_trajectory()

            for step in range(self.params.stats_params.max_episode_steps):
                last_seg = traj_segment
                traj_segment = self.receive_edge_trajectory()
                stat_observations = last_seg.observations[0:5]

                r = self.reward_fcn.reward(stat_observations,
                                           self.target,
                                           traj_segment.last_action,
                                           traj_segment.failed, pole_length=self.params.physics_params.length).squeeze()
                if training:
                    self.trainer.store_experience(last_seg.observations, self.target, traj_segment.last_action, r,
                                                  traj_segment.observations, traj_segment.failed)

                    if self.optimize_condition.acquire(False):
                        self.optimize_condition.notify_all()
                        self.optimize_condition.release()
                else:

                    self.model_stats.cart_positions.append(last_seg.observations[0])
                    self.model_stats.pendulum_angele.append(math.atan2(last_seg.observations[2], last_seg.observations[3]))
                    self.model_stats.actions.append(traj_segment.last_action)

                self.model_stats.observations = copy.deepcopy(traj_segment.observations[0:5])
                self.model_stats.targets = copy.deepcopy(self.target)
                self.model_stats.measure(self.model_stats.observations, self.model_stats.targets, traj_segment.failed,
                                         pole_length=self.params.physics_params.length,
                                         distance_score_factor=self.params.reward_params.distance_score_factor)

                self.model_stats.reward.append(r)

                if not traj_segment.normal_operation:
                    break

            if training:
                self.model_stats.add_steps(step)
                self.model_stats.training_monitor(self.ep)
            else:
                self.model_stats.add_steps(1)  # Add one step so that it doesn't overlap with possible previous evals
                self.model_stats.evaluation_monitor_scalar(self.ep)
                self.model_stats.evaluation_monitor_image(self.ep)

                dsas = float(self.model_stats.survived) * self.model_stats.get_average_distance_score()
                # self.agent.save_weights(self.params.stats_params.model_name + '_' + str(ep))

                moving_average_dsas = 0.95 * moving_average_dsas + 0.05 * dsas

                if moving_average_dsas > best_dsas:
                    self.agent.save_weights(self.params.stats_params.model_name + '_best')
                    best_dsas = moving_average_dsas

            self.initiate_reset()
            # t0 = time.time()
            # while time.time() - t0 < self.params.stats_params.reset_delay:
            #     self.receive_edge_trajectory()

    def optimize_ddpg(self):

        while True:

            self.optimize_condition.acquire()
            self.optimize_condition.wait()

            self.trainer.optimize()

            if self.edge_ready:
                weights = self.agent.get_actor_weights()
                self.edge_ready = False
                self.send_weights(weights)
                self.agent.save_weights(self.params.stats_params.model_name)
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

    def run(self):
        """It's triple threads"""
        # self.t1.start()
        self.t2.start()
        self.t3.start()
        self.store_trajectory()

    def initiate_reset(self):
        reset_pack = pickle.dumps(True)
        self.redis_connection.publish(channel=self.params.redis_params.ch_plant_reset, message=reset_pack)
