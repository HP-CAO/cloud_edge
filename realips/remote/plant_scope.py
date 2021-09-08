import time
from realips.env.gym_physics import GymPhysics, GymPhysicsParams
from realips.remote.redis import RedisParams, RedisConnection
from realips.remote.transition import TrajectorySegment
import pickle
import tensorflow as tf

from realips.utils import observations2states

import matplotlib.pyplot as plt


class PlantScopeParams:
    def __init__(self):
        self.redis_params = RedisParams()
        self.physics_params = GymPhysicsParams()


class PlantScope:
    def __init__(self, params: PlantScopeParams):
        self.params = params
        self.physics = GymPhysics(self.params.physics_params)
        self.redis_connection = RedisConnection(self.params.redis_params)
        self.states_subscriber = self.redis_connection.subscribe(
            channel=self.params.redis_params.ch_plant_trajectory_segment)
        self.edge_trajectory_subscriber = self.redis_connection.subscribe(
            channel=self.params.redis_params.ch_edge_trajectory)

        self.actions = [0.0] * 100
        self.state = [0.0] * 5
        self.last_update = time.time()

    def receive_plant_trajectory(self):
        plant_states_pack = self.states_subscriber.parse_response()[2]
        plant_states = pickle.loads(plant_states_pack)
        return plant_states

    def receive_edge_trajectory(self):
        edge_trajectory_pack = self.edge_trajectory_subscriber.parse_response()[2]
        edge_seg = TrajectorySegment.pickle_load_pack(edge_trajectory_pack)
        return edge_seg

    def receive_edge_trajectory_non_blocking(self):
        message = self.edge_trajectory_subscriber.parse_response(block=False)
        if message is None:
            return None
        edge_trajectory_pack = message[2]
        edge_seg = TrajectorySegment.pickle_load_pack(edge_trajectory_pack)
        return edge_seg

    def visualize_states(self):

        print("Connected, waiting for messages")
        step = 0
        plt.figure("Action")
        plt.axis([-0, 100, -1.5, 1.5])
        plt.ion()
        plt.show()
        while True:
            traj = self.receive_edge_trajectory_non_blocking()
            if traj is None:
                self.physics.render(states=self.state)
                self.visualize_actions()
                traj = self.receive_edge_trajectory()

            self.actions.append(traj.last_action)
            self.actions = self.actions[1:]
            self.state = traj.state


            # edge_seg = self.receive_edge_trajectory()
            # with self.tf_monitor.as_default():
            #     tf.summary.scalar('action_live', edge_seg.last_action, step)
            # step += 1

    def visualize_actions(self):

        t = time.time()
        if t - self.last_update > 1 / 30.:
            plt.cla()
            plt.plot(range(100), self.actions)
            plt.draw()
            plt.pause(0.001)
            self.last_update = t
