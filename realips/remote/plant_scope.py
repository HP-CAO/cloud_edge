from realips.env.gym_physics import GymPhysics, GymPhysicsParams
from realips.remote.redis import RedisParams, RedisConnection
import struct


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

    def receive_trajectory_segment(self):
        """"
        return a list of states: [x, x_dot, theta_rescale, theta_dot], action, failed, normal_operation
        where theta_rescale should be in [-pi, pi]
        """
        experience_pack = self.states_subscriber.parse_response()[2]
        states = struct.unpack("I5f2?", experience_pack)[1:5]
        return states

    def visualize_states(self):

        print("Connected, waiting for messages")

        while True:

            states = self.receive_trajectory_segment()
            self.physics.render(states=states)
