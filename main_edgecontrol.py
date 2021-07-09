import argparse
import os
from realips.remote.edge_control import EdgeControlParams, DDPGEdgeControl

from utils import *

def main_ddpg(p, eval=None):
    edge_ddpg = DDPGEdgeControl(p, eval)
    edge_ddpg.run()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--generate_config', action='store_true', help='Enable to write default config only')
    parser.add_argument('--config', default=None, help='Path to config file')
    parser.add_argument('--weights', default=None, help='Path to pretrained weights')
    parser.add_argument('--params', nargs='*', default=None)
    parser.add_argument('--eval', default=None, help='Path to pretrained weights to evaluate')

    args = parser.parse_args()

    if args.generate_config:
        generate_config(EdgeControlParams(), "config/default_edgecontrol.json")
        exit("Edgecontrol config file generated")

    if args.config is None:
        exit("Config file needed")

    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    params = read_config(args.config)

    if args.params is not None:
        params = override_params(params, args.params)

    main_ddpg(params, args.eval)
