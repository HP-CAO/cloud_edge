#!/usr/bin/env sh

python main_cloudtrain.py --config ./config/remote/remote_cloud_ddpg+.json --weights ./models/sup_sim/sim_ddpg_+_low_friction_sup_1_best_DDPG/ --id real_low_sup_1_5th --gpu
python main_cloudtrain.py --config ./config/remote/remote_cloud_ddpg+.json --weights ./models/sup_sim/sim_ddpg_+_okay_friction_sup_1_best_DDPG/ --id real_okay_sup_1_5th --gpu
python main_cloudtrain.py --config ./config/remote/remote_cloud_ddpg+.json --weights ./models/sup_sim/sim_ddpg_+_high_friction_sup_1_best_DDPG/ --id real_high_sup_1_5th --gpu
python main_cloudtrain.py --config ./config/remote/remote_cloud_ddpg+.json --weights ./models/sup_sim/sim_ddpg_+_over_friction_sup_1_best_DDPG/ --id real_over_sup_1_5th --gpu
python main_cloudtrain.py --config ./config/remote/remote_cloud_ddpg+.json --weights ./models/sup_sim/sim_ddpg_+_non_friction_sup_1_best_DDPG/ --id real_non_sup_1_5th --gpu