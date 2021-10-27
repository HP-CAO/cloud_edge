import pickle
import matplotlib.pyplot as plt
from realips.env.gym_physics import GymPhysics, GymPhysicsParams
from realips.remote.redis import RedisParams, RedisConnection
from realips.remote.transition import TrajectorySegment


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
        self.states = [[0.0] * 5] * 100

        fig, self.ax = plt.subplots(3, 1)
        self.pos_ax = self.ax[1]
        self.vel_ax = self.ax[1].twinx()
        self.the_ax = self.ax[2]
        self.ome_ax = self.ax[2].twinx()
        plt.axis([-0, 100, -1.5, 1.5])
        plt.ion()
        plt.show()

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
        while True:
            traj = self.receive_edge_trajectory_non_blocking()
            if traj is None:
                self.physics.render(states=self.states[-1])
                self.plot_actions()
                self.plot_states()
                plt.draw()
                plt.pause(0.001)
                traj = self.receive_edge_trajectory()

            self.actions.append(traj.last_action)
            self.actions = self.actions[1:]
            self.states.append(traj.state)
            self.states = self.states[1:]

            # edge_seg = self.receive_edge_trajectory()
            # with self.tf_monitor.as_default():
            #     tf.summary.scalar('action_live', edge_seg.last_action, step)
            # step += 1

    def plot_actions(self):
        plt.sca(self.ax[0])
        plt.cla()
        plt.plot(range(100), self.actions)
        plt.grid(True)

    def plot_states(self):
        plt.sca(self.pos_ax)
        plt.cla()
        states = list(zip(*self.states))
        plt.plot(range(100), states[0])
        y_lim = plt.ylim()
        y_lim_max = max([abs(y) for y in y_lim])
        plt.ylim([-y_lim_max, y_lim_max])
        plt.sca(self.vel_ax)
        plt.cla()
        plt.plot(range(100), states[1], color="orange")
        y_lim = plt.ylim()
        y_lim_max = max([abs(y) for y in y_lim])
        plt.ylim([-y_lim_max, y_lim_max])
        plt.grid(True)

        plt.sca(self.the_ax)
        plt.cla()
        states = list(zip(*self.states))
        plt.plot(range(100), states[2])
        y_lim = plt.ylim()
        y_lim_max = max([abs(y) for y in y_lim])
        plt.ylim([-y_lim_max, y_lim_max])
        plt.sca(self.ome_ax)
        plt.cla()
        plt.plot(range(100), states[3], color="orange")
        y_lim = plt.ylim()
        y_lim_max = max([abs(y) for y in y_lim])
        plt.ylim([-y_lim_max, y_lim_max])
        plt.grid(True)
