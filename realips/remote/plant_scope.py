from realips.env.gym_physics import GymPhysics, GymPhysicsParams
from realips.remote.redis import RedisParams, RedisConnection
from realips.remote.transition import TrajectorySegment
import struct
import pickle


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

    def receive_plant_trajectory(self):
        plant_states_pack = self.states_subscriber.parse_response()[2]
        plant_states = pickle.loads(plant_states_pack)
        return plant_states

    def receive_edge_trajectory(self):
        edge_trajectory_pack = self.edge_trajectory_subscriber.parse_response()[2]
        edge_seg = TrajectorySegment.pickle_load_pack(edge_trajectory_pack)
        return edge_seg

    def visualize_states(self):

        print("Connected, waiting for messages")

        while True:
            states = self.receive_plant_trajectory()
            self.physics.render(states=states)
