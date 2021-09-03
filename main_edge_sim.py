import argparse
from realips.remote.edge_sim import SimEdgeControlParams, SimEdgeControl

from utils import *


def main(p):
    edge = SimEdgeControl(p)
    edge.run()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--generate_config', action='store_true', help='Enable to write default config only')
    parser.add_argument('--config', default=None, help='Path to config file')
    parser.add_argument('--params', nargs='*', default=None)

    args = parser.parse_args()

    if args.generate_config:
        generate_config(SimEdgeControlParams(), "config/default_edgecontrol_sim.json")
        exit("Edgecontrol config file generated")

    if args.config is None:
        exit("Config file needed")

    params = read_config(args.config)

    if args.params is not None:
        params = override_params(params, args.params)

    main(params)
