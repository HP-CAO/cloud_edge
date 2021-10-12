import math
import gym
from gym.utils import seeding
from realips.utils import states2observations
import numpy as np


class GymPhysicsParams:
    def __init__(self):
        self.x_threshold = 0.3
        self.theta_dot_threshold = 15
        self.kinematics_integrator = 'euler'

        self.ini_states = [0., 0., -math.pi, 0., False]
        self.gravity = 9.8
        self.mass_cart = 0.94
        self.mass_pole = 0.23
        self.force_mag = 5.0
        self.voltage_mag = 5.0

        self.length = 0.64
        self.theta_random_std = 0.8
        self.friction_cart = 10
        self.friction_pole = 0.0011
        self.with_friction = True
        self.force_input = False
        self.simulation_frequency = 30
        self.actuation_delay = 1


class GymPhysics(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30  # this is not working
    }

    def __init__(self, params: GymPhysicsParams):

        self.params = params
        self.total_mass = (self.params.mass_cart + self.params.mass_pole)
        self.pole_mass_length = self.params.mass_pole / 2

        self.seed()
        self.viewer = None
        self.states = None
        self.steps_beyond_terminal = None
        self.last_actions = [0.0] * params.actuation_delay
        self.tau = 1 / self.params.simulation_frequency

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def step(self, action: float):
        """
        param: action: a scalar value (not numpy type) [-1,1]
        return: a list of states
        """

        if self.params.actuation_delay > 0:
            self.last_actions.append(action)
            self.last_actions = self.last_actions[1:]
            action = self.last_actions[0]

        x, x_dot, theta, theta_dot, failed = self.states

        if self.params.force_input:
            force = action * self.params.force_mag
        else:
            voltage = action * self.params.voltage_mag
            force = self.voltage2force(voltage, x_dot)

        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # kinematics of the inverted pendulum
        if self.params.with_friction:
            """ with friction"""
            temp \
                = (force + self.pole_mass_length * theta_dot ** 2 *
                   sintheta - self.params.friction_cart * x_dot) / self.total_mass

            thetaacc = \
                (self.params.gravity * sintheta - costheta * temp -
                 self.params.friction_pole * theta_dot / self.pole_mass_length) / \
                (self.params.length / 2 * (4.0 / 3.0 - self.params.mass_pole * costheta ** 2 / self.total_mass))

            xacc = temp - self.pole_mass_length * thetaacc * costheta / self.total_mass

        else:
            """without friction"""
            temp = (force + self.pole_mass_length * theta_dot ** 2 * sintheta) / self.total_mass
            thetaacc = (self.params.gravity * sintheta - costheta * temp) / \
                       (self.params.length / 2 * (4.0 / 3.0 - self.params.mass_pole * costheta ** 2 / self.total_mass))
            xacc = temp - self.pole_mass_length * thetaacc * costheta / self.total_mass

        if self.params.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
            failed = self.is_failed(x, theta_dot)

        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot
            failed = self.is_failed(x, theta_dot)

        theta_rescale = math.atan2(math.sin(theta), math.cos(theta))
        new_states = [x, x_dot, theta_rescale, theta_dot, failed]

        self.states = new_states  # to update animation

        return self.states

    def reset(self):
        self.states = self.params.ini_states
        self.last_actions = [0.0] * self.params.actuation_delay

    def random_reset(self):
        ran_x = np.random.uniform(-0.8 * self.params.x_threshold, 0.8 * self.params.x_threshold)
        if self.is_failed(ran_x, 0):
            ran_x += 0
        ran_v = 0
        ran_theta = np.random.normal(math.pi, self.params.theta_random_std)
        ran_theta_v = 0
        failed = False
        self.states = [ran_x, ran_v, ran_theta, ran_theta_v, failed]
        self.last_actions = [0.0] * self.params.actuation_delay

    def render(self, mode='human', states=None, is_normal_operation=True):

        screen_width = 600
        screen_height = 400
        world_width = self.params.x_threshold * 2 + 1
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * self.params.length
        cartwidth = 50.0
        cartheight = 30.0
        target_width = 45
        target_height = 45

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            # target
            self.targettrans = rendering.Transform()
            target = rendering.Image('./target.svg', width=target_width, height=target_height)
            target.add_attr(self.targettrans)
            self.viewer.add_geom(target)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            if not is_normal_operation:
                cart.set_color(1.0, 0, 0)
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)
            self._pole_geom = pole

        if states is None:
            if self.states is None:
                return None
            else:
                x = self.states
        else:
            x = states

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        targetx = 0 * scale + screen_width / 2.0
        targety = polelen + carty

        self.carttrans.set_translation(cartx, carty)
        self.targettrans.set_translation(targetx, targety)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def is_failed(self, x, theta_dot):
        failed = bool(x <= -self.params.x_threshold
                      or x >= self.params.x_threshold
                      or theta_dot > self.params.theta_dot_threshold)
        return failed

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def get_shape_observations(self):
        observations, _ = states2observations(self.params.ini_states)
        return len(observations)

    def voltage2force(self, voltage, cart_v):
        """
        Convert voltage control to force control
        :param voltage: voltage action from the agent
        :return: force actuation to the plant
        """

        # f = 0.90 * 3.71 * 0.69 * 7.68 * (voltage * 6.35 - 7.68 * 3.71 * cart_v) / (6.35 * 6.35 * 2.6)
        f = 17.69 * (voltage * 6.35 - 28.4928 * cart_v) / 104.8385
        return f
