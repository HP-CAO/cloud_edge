#!/usr/bin/env sh

python main_cloudtrain.py --config ./config/remote/remote_cloud_ddpg+.json --weights ./models/models_11_11/sim_ddpg+_30hz_okay_friction_voltage_4cm_best_DDPG/ --id real_bandwidth_0.06_with_cer_1st --gpu --params cloud_params/artificial_bandwidth 0.06
python main_cloudtrain.py --config ./config/remote/remote_cloud_ddpg+.json --weights ./models/models_11_11/sim_ddpg+_30hz_okay_friction_voltage_4cm_best_DDPG/ --id real_bandwidth_0.5_with_cer_1st --gpu --params cloud_params/artificial_bandwidth 0.5
python main_cloudtrain.py --config ./config/remote/remote_cloud_ddpg+.json --weights ./models/models_11_11/sim_ddpg+_30hz_okay_friction_voltage_4cm_best_DDPG/ --id real_bandwidth_5_with_cer_1_st --gpu --params cloud_params/artificial_bandwidth 5
