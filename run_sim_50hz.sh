#!/usr/bin/env sh

python main_ddpg.py --config ./config/sim_50hz/sim_ddpg_50hz_non_friction.json --id sim_ddpg_50hz_non_friction --gpu --force &
python main_ddpg.py --config ./config/sim_50hz/sim_ddpg_50hz_non_friction_action_noise.json --id sim_ddpg_50hz_non_friction_action_noise --gpu --force &
python main_ddpg.py --config ./config/sim_50hz/sim_ddpg_50hz_non_friction_action_noise_no_prefill.json --id sim_ddpg_50hz_non_friction_action_noise_no_prefill --gpu --force &
python main_ddpg.py --config ./config/sim_50hz/sim_ddpg_50hz_non_friction_delayed_update_action_noise.json --id sim_ddpg_50hz_non_friction_delayed_update_action_noise --gpu --force &
python main_td3.py --config ./config/sim_50hz/sim_td3_50hz_non_friction.json --id sim_td3_50hz_non_friction --gpu --force