import argparse

import numpy as np
import copy

from utils import read_config, write_config

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config/ddpg.json', help='Path to config file')
    parser.add_argument('--var', default=0.1, help='Variation in the parameter [1 - var, 1 + var]')
    parser.add_argument('--n', default=4, help='Number of files to be generated')
    parser.add_argument('--name', default="./config/ddpg_var{}.json", help='Base name with {} to give to gen configs')

    args = parser.parse_args()

    params = read_config(args.config)

    physics_params = copy.deepcopy(params.physics_params)
    v = float(args.var)
    N = int(args.n)
    var = 1 + np.random.uniform(-v, v, size=(4, N))

    for n in range(N):
        params.physics_params.mass_cart = var[0, n] * physics_params.mass_cart
        params.physics_params.mass_pole = var[1, n] * physics_params.mass_pole
        params.physics_params.length = var[2, n] * physics_params.length
        params.physics_params.x_threshold = var[3, n] * physics_params.x_threshold
        params.stats_params.log_file_name = args.name[args.name.rfind('/') + 1: args.name.rfind('.')].format(n+1)
        params.stats_params.model_name = args.name[args.name.rfind('/') + 1: args.name.rfind('.')].format(n+1)
        write_config(params, args.name.format(n + 1))
