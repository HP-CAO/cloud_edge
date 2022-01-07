#!/usr/bin/env sh

python main_cloudtrain.py --config ./config/remote/remote_cloud_ddpg+.json --weights ./models/models_11_11/sim_ddpg+_30hz_okay_friction_voltage_4cm_best_DDPG/ --id real_bandwidth_100_no_cer_4th --gpu --params trainer_params/combined_experience_replay false
python main_cloudtrain.py --config ./config/remote/remote_cloud_ddpg+.json --weights ./models/models_11_11/sim_ddpg+_30hz_okay_friction_voltage_4cm_best_DDPG/ --id real_bandwidth_100_no_cer_5th --gpu --params trainer_params/combined_experience_replay false
python main_cloudtrain.py --config ./config/remote/remote_cloud_ddpg+.json --weights ./models/models_11_11/sim_ddpg+_30hz_okay_friction_voltage_4cm_best_DDPG/ --id real_bandwidth_100_no_cer_6th --gpu --params trainer_params/combined_experience_replay false