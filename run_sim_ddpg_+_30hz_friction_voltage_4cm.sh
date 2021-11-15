#!/usr/bin/env sh

python main_ddpg.py --config ./config/sim_30hz/sim_ddpg_+_30hz_friction_voltage_4cm/sim_30hz_non_friction.json --id sim_ddpg+_30hz_non_friction_voltage_4cm --gpu --force &
python main_ddpg.py --config ./config/sim_30hz/sim_ddpg_+_30hz_friction_voltage_4cm/sim_30hz_okay_friction.json --id sim_ddpg+_30hz_okay_friction_voltage_4cm --gpu --force &
python main_ddpg.py --config ./config/sim_30hz/sim_ddpg_+_30hz_friction_voltage_4cm/sim_30hz_low_friction.json --id sim_ddpg+_30hz_low_friction_voltage_4cm --gpu --force &
python main_ddpg.py --config ./config/sim_30hz/sim_ddpg_+_30hz_friction_voltage_4cm/sim_30hz_high_friction.json --id sim_ddpg+_30hz_high_friction_voltage_4cm --gpu --force &
python main_ddpg.py --config ./config/sim_30hz/sim_ddpg_+_30hz_friction_voltage_4cm/sim_30hz_over_friction.json --id sim_ddpg+_30hz_over_friction_voltage_4cm --gpu --force

