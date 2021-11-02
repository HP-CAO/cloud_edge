import math
import time
from realips.env.gym_physics import GymPhysics, GymPhysicsParams

car_pole_params = GymPhysicsParams()
car_pole_params.ini_states = [0., 0., 1, 0., False]
cart_pole = GymPhysics(car_pole_params)
cart_pole.reset()

while True:
    # print(cart_pole.states)
    t0 = time.time()
    cart_pole.render(mode='human')
    cart_pole.step(action=0.0)
    print(time.time() - t0)
