import argparse
import os
import tensorflow as tf
from realips.remote.cloud_trainer import CloudSystem, CloudSystemParams
from utils import *


def main(p):
    cloud = CloudSystem(p)
    cloud.run()


def main_eval(p, episodes):
    cloud = CloudSystem(p)
    cloud.params.stats_params.converge_episodes = episodes
    for cloud.ep in range(episodes):
        cloud.model_stats.reset_status()
        cloud.run_episode(training=False)
        cloud.model_stats.add_steps(1)  # For proper logging


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', action='store_true', help='Activate usage of GPU')
    parser.add_argument('--generate_config', action='store_true', help='Enable to write default config only')
    parser.add_argument('--config', default=None, help='Path to config file')
    parser.add_argument('--id', default=None, help='If set overrides the logfile name and the save name')
    parser.add_argument('--force', action='store_true', help='Override log file without asking')
    parser.add_argument('--weights', default=None, help='Path to pretrained weights')
    parser.add_argument('--params', nargs='*', default=None)
    parser.add_argument('--eval_episodes', default=None, help='Set to number of evaluation episodes if eval only')

    args = parser.parse_args()

    if args.generate_config:
        generate_config(CloudSystemParams(), "config/default_cloud.json")
        exit("cloud_ddpg_config file generated")

    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        physical_devices = tf.config.list_physical_devices('GPU')
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except:
            exit("GPU allocated failed")

    if args.config is None:
        exit("config file needed")

    params = read_config(args.config)

    if args.params is not None:
        params = override_params(params, args.params)

    if args.id is not None:
        params.stats_params.model_name = args.id
        params.stats_params.log_file_name = args.id

    if args.force:
        params.stats_params.force_override = True

    if args.weights is not None:
        params.stats_params.weights_path = args.weights

    if args.eval_episodes is not None:
        main_eval(params, int(args.eval_episodes))
    else:
        main(params)
