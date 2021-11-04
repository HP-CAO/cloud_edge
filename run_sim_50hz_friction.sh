#!/usr/bin/env sh

python main_ddpg.py --config ./config/real_ddpg++/sim_50hz_non_friction.json --id sim_ddpg++_50hz_non_friction --gpu --force &
python main_ddpg.py --config ./config/real_ddpg++/sim_50hz_okay_friction.json --id sim_ddpg++_50hz_okay_friction --gpu --force &
python main_ddpg.py --config ./config/real_ddpg++/sim_50hz_over_friction.json --id sim_ddpg++_50hz_over_friction --gpu --force

