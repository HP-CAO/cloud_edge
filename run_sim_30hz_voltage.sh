#!/usr/bin/env sh

python main_ddpg.py --config ./config/sim_30hz/sim_30hz_voltage/sim_ddpg_30hz_okay_friction.json --id sim_ddpg_30hz_okay_friction_voltage --gpu --force &
python main_ddpg.py --config ./config/sim_30hz/sim_30hz_voltage/sim_ddpg_+_30hz_okay_friction.json --id sim_ddpg_+_30hz_okay_friction_voltage --gpu --force &
python main_ddpg.py --config ./config/sim_30hz/sim_30hz_voltage/sim_ddpg_++_30hz_okay_friction.json --id sim_ddpg_++_30hz_okay_friction_voltage --gpu --force &
python main_td3.py --config ./config/sim_30hz/sim_30hz_voltage/sim_td3_30hz_okay_friction.json --id sim_td3_30hz_okay_friction_voltage --gpu --force