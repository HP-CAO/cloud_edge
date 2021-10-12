import argparse
import os
import tensorflow as tf

from realips.system.ips_ddpg import IpsDDPG, IpsDDPGParams
from utils import *


def main_train(p):
    ips = IpsDDPG(p)
    ips.train()


def main_test(p):
    ips = IpsDDPG(p)
    ips.test()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true', help='Activate usage of GPU')
    parser.add_argument('--generate_config', action='store_true', help='Enable to write default config only')
    parser.add_argument('--config', default='./config/local_ddpgips.json', help='Path to config file')
    parser.add_argument('--id', default=None, help='If set overrides the logfile name and the save name')
    parser.add_argument('--force', action='store_true', help='Override log file without asking')
    parser.add_argument('--weights', default=None, help='Path to pretrained weights')
    parser.add_argument('--params', nargs='*', default=None)
    parser.add_argument('--mode', default='train', help='Choose the mode, train or test')

    args = parser.parse_args()

    if args.generate_config:
        generate_config(IpsDDPGParams(), "config/default_ddpg_ips.json")
        exit("ddpgips_config file generated")

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

    params.stats_params.running_mode = args.mode

    if args.mode == 'train':
        main_train(params)
    elif args.mode == 'test':
        if args.weights is None:
            exit("Please load the pretrained weights")
        else:
            main_test(params)
    else:
        assert NameError('No such mode. train or test?')
