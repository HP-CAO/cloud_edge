#!/usr/bin/env sh

#python main_cloudtrain.py --config ./config/remote/remote_cloud_ddpg+.json --weights ./models/models_11_11/sim_ddpg+_30hz_okay_friction_voltage_4cm_best_DDPG/ --id real_bandwidth_15_no_cer_1st --gpu --params cloud_params/artificial_bandwidth 15 trainer_params/combined_experience_replay false
#python main_cloudtrain.py --config ./config/remote/remote_cloud_ddpg+.json --weights ./models/models_11_11/sim_ddpg+_30hz_okay_friction_voltage_4cm_best_DDPG/ --id real_bandwidth_15_no_cer_2nd --gpu --params cloud_params/artificial_bandwidth 15 trainer_params/combined_experience_replay false
python main_cloudtrain.py --config ./config/remote/remote_cloud_ddpg+.json --weights ./models/models_11_11/sim_ddpg+_30hz_okay_friction_voltage_4cm_best_DDPG/ --id real_bandwidth_15_no_cer_3rd --gpu --params cloud_params/artificial_bandwidth 15 trainer_params/combined_experience_replay false
#python main_cloudtrain.py --config ./config/remote/remote_cloud_ddpg+.json --weights ./models/models_11_11/sim_ddpg+_30hz_okay_friction_voltage_4cm_best_DDPG/ --id real_bandwidth_15_no_cer_4th --gpu --params cloud_params/artificial_bandwidth 15 trainer_params/combined_experience_replay false
#python main_cloudtrain.py --config ./config/remote/remote_cloud_ddpg+.json --weights ./models/models_11_11/sim_ddpg+_30hz_okay_friction_voltage_4cm_best_DDPG/ --id real_bandwidth_15_no_cer_5th --gpu --params cloud_params/artificial_bandwidth 15 trainer_params/combined_experience_replay false



