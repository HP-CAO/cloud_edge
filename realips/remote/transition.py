import math
import pickle
import struct

from realips.utils import observations2states


class TrajectorySegment:
    def __init__(self):
        self.state = [0.0] * 4
        self.last_action = 0.0
        self.failed = False
        self.normal_operation = True
        self.sequence_number = 0
        self.observations = []

    @staticmethod
    def from_packet(packet):
        s = TrajectorySegment()
        seg = struct.unpack("I5f2?", packet)
        s.state = seg[1:5]
        s.last_action = seg[5]
        s.failed = seg[6]
        s.normal_operation = seg[7]
        s.sequence_number = seg[0]
        s.observations = s.get_observation()
        return s

    @staticmethod
    def pickle_load_pack(packet):
        s = TrajectorySegment()
        seg = pickle.loads(packet)  # observations, last_action, failed, operations
        s.observations = seg[0]
        s.last_action = seg[1]
        s.failed = seg[2]
        s.normal_operation = seg[3]
        s.sequence_number = seg[-1]
        s.state = observations2states(s.observations, s.failed)
        return s

    def get_observation(self):
        x, x_dot, theta, theta_dot = self.state
        observations = [x, x_dot, math.sin(theta), math.cos(theta), theta_dot]
        return observations
