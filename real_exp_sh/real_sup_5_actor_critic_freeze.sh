#!/usr/bin/env sh

python main_cloudtrain.py --config ./config/remote/remote_cloud_ddpg+.json --weights ./models/models_11_11/sim_ddpg+_30hz_okay_friction_voltage_4cm_best_DDPG/ --id real_actor_5k_critic_3.5k_5th --gpu --params trainer_params/pre_fill_exp 3500 trainer_params/actor_freeze_step_count 5000
python main_cloudtrain.py --config ./config/remote/remote_cloud_ddpg+.json --weights ./models/models_11_11/sim_ddpg+_30hz_okay_friction_voltage_4cm_best_DDPG/ --id real_actor_3.5k_critic_3.5k_5th --gpu --params trainer_params/pre_fill_exp 3500 trainer_params/actor_freeze_step_count 3500
python main_cloudtrain.py --config ./config/remote/remote_cloud_ddpg+.json --weights ./models/models_11_11/sim_ddpg+_30hz_okay_friction_voltage_4cm_best_DDPG/ --id real_actor_5k_critic_128_5th --gpu --params trainer_params/pre_fill_exp 128 trainer_params/actor_freeze_step_count 5000
python main_cloudtrain.py --config ./config/remote/remote_cloud_ddpg+.json --weights ./models/models_11_11/sim_ddpg+_30hz_okay_friction_voltage_4cm_best_DDPG/ --id real_actor_128_critic_128_5th --gpu --params trainer_params/pre_fill_exp 128 trainer_params/actor_freeze_step_count 128
