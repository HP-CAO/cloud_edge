import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sns.set_theme(style="whitegrid")


def pad_dict_list(dict_list, padel):
    lmax = 0
    for lname in dict_list.keys():
        lmax = max(lmax, len(dict_list[lname]))
    for lname in dict_list.keys():
        ll = len(dict_list[lname])
        if ll < lmax:
            dict_list[lname] += [padel] * (lmax - ll)
    return dict_list


def dict_from_list(dict):
    new_dict = {}
    for k in dict.keys():
        a = np.array(dict[k])
        mean = np.mean(a)
        min = np.min(a)
        max = np.max(a)
        new_dict[k] = [mean, min, max]

    return new_dict


def plot_dif_len(data):
    nd = dict_from_list(data)
    df = pd.DataFrame(nd)
    sns.pointplot(data=df, capsize=.2)
    plt.show()


def plot(data, legend=None, lables=None, name=None):
    # nd = dict_from_list(data)
    df = pd.DataFrame(data)
    ax = sns.pointplot(data=df, capsize=.2, join=False)
    if lables is not None:
        ax.set_xlabel(lables[0])
        ax.set_ylabel(lables[1])
    if legend is not None:
        ax.legend(legend, loc=2)
    plt.show()
    save_path = os.path.join("{}_plot.png".format(name))
    plt.savefig(save_path)
    # plt.show(block=False)


def plot_box(data, legend=None, lables=None):
    # nd = dict_from_list(data)
    df = pd.DataFrame(data)
    ax = sns.boxplot(data=df)
    ax = sns.swarmplot(data=df, color=".25")
    if lables is not None:
        ax.set_xlabel(lables[0])
        ax.set_ylabel(lables[1])
    if legend is not None:
        ax.legend(legend, loc=2)
    plt.show()
    # plt.show(block=False)


if __name__ == "__main__":
    data_fric = {
        "no_fr": ["logs_15_11/real_ddpg_+_30hz_non_friction_vol_4cm_both_con_5_vol_4",
                  "logs_15_11/real_ddpg_+_30hz_non_friction_vol_4cm_both_con_5_vol_4_3rd",
                  "logs_20_11/real_ddpg_+_30hz_non_friction_vol_4cm_both_con_5_200k"],
        "lo_fr": ["logs_20_11/real_ddpg_+_30hz_low_friction_vol_4cm_both_con_5_200k",
                  "logs_16_11/real_ddpg_+_30hz_low_friction_vol_4cm_both_con_5_vol_4_2nd",
                  "logs_13_12/low_friction_eval_after_freeze"],
        "ok_fr": ["logs_14_11/real_ddpg_+_30hz_okay_friction_vol_4cm_both_con_5",
                  "logs_14_11/real_ddpg_+_30hz_okay_friction_vol_4cm_both_con_5_3rd",
                  "logs_14_11/real_ddpg_+_30hz_okay_friction_vol_4cm_both_con_5_4th_video"],
        "hi_fr": ["logs_20_11/real_ddpg_+_30hz_high_friction_vol_4cm_both_con_5_200k",
                  "logs_20_11/real_ddpg_+_30hz_high_friction_vol_4cm_both_con_5_200k_2nd",
                  "logs_16_11/real_ddpg_+_30hz_high_friction_vol_4cm_both_con_5_vol_4_3rd"],
        "over_fr": ["logs_20_11/real_ddpg_+_30hz_over_friction_vol_4cm_both_con_5_200k",
                    "logs_16_11/real_ddpg_+_30hz_over_friction_vol_4cm_both_con_5_vol_4_2nd",
                    "logs_20_11/real_ddpg_+_30hz_over_friction_vol_4cm_both_con_5_200k_2nd"]
    }
    lab_fric = ['frction level', 'steps in [k]']

    data_freeze = {
        "a_5_c_3.5": ["logs_14_11/real_ddpg_+_30hz_okay_friction_vol_4cm_both_con_5",
                      "logs_14_11/real_ddpg_+_30hz_okay_friction_vol_4cm_both_con_5_3rd",
                      "logs_14_11/real_ddpg_+_30hz_okay_friction_vol_4cm_both_con_5_4th_video"],
        "a_3.5_c_3.5": ["logs_17_11/real_ddpg_+_30hz_okay_friction_vol_4cm_both_con_5_vol_4_3.5k_all",
                        "logs_17_11/real_ddpg_+_30hz_okay_friction_vol_4cm_both_con_5_vol_4_3.5k_all_2nd",
                        "logs_17_11/real_ddpg_+_30hz_okay_friction_vol_4cm_both_con_5_vol_4_3.5k_all_3rd"],
        "a_5_c_0": ["logs_20_11/real_ddpg_+_30hz_okay_friction_vol_4cm_both_con_5_200k_actor_5_critic_0",
                    "logs_14_11/real_ddpg_+_30hz_okay_friction_vol_4cm_both_con_5_freeze_actor_only_2nd",
                    "logs_20_11/real_ddpg_+_30hz_okay_friction_vol_4cm_both_con_5_200k_actor_5_critic_0_2nd"],
        "a_0_c_0": ["logs_20_11/real_ddpg_+_30hz_okay_friction_vol_4cm_both_con_5_200k_non_freezeing",
                    "logs_15_11/real_ddpg_+_30hz_okay_friction_vol_4cm_both_con_5_non_freeze_2nd",
                    "logs_15_11/real_ddpg_+_30hz_okay_friction_vol_4cm_both_con_5_non_freeze_3rd"]
    }

    data_bandwidth = {
        "bw_0.06": ["logs_24_11/real_ddpg_+_30hz_okay_friction_vol_4cm_both_con_5_bw_0.06",
                    "logs_24_11/real_ddpg_+_30hz_okay_friction_vol_4cm_both_con_5_bw_0.06_2nd",
                    "logs_24_11/real_ddpg_+_30hz_okay_friction_vol_4cm_both_con_5_bw_0.06_3rd"],
        "bw_0.1": ["logs_24_11/real_ddpg_+_30hz_okay_friction_vol_4cm_both_con_5_bw_0.1",
                   "logs_24_11/real_ddpg_+_30hz_okay_friction_vol_4cm_both_con_5_bw_0.1_2nd",
                   "logs_24_11/real_ddpg_+_30hz_okay_friction_vol_4cm_both_con_5_bw_0.1_3rd"],
        "bw_1": ["logs_24_11/real_ddpg_+_30hz_okay_friction_vol_4cm_both_con_5_bw_1",
                 "logs_24_11/real_ddpg_+_30hz_okay_friction_vol_4cm_both_con_5_bw_1_2nd",
                 "logs_24_11/real_ddpg_+_30hz_okay_friction_vol_4cm_both_con_5_bw_1_3rd"],
        "bw_5": ["logs_24_11/real_ddpg_+_30hz_okay_friction_vol_4cm_both_con_5_bw_5",
                 "logs_24_11/real_ddpg_+_30hz_okay_friction_vol_4cm_both_con_5_bw_5_2nd",
                 "logs_24_11/real_ddpg_+_30hz_okay_friction_vol_4cm_both_con_5_bw_5_3rd"],
        "bw_50": ["logs_24_11/real_ddpg_+_30hz_okay_friction_vol_4cm_both_con_5_bw_50_2nd",
                  "logs_24_11/real_ddpg_+_30hz_okay_friction_vol_4cm_both_con_5_bw_50_3rd",
                  "logs_24_11/real_ddpg_+_30hz_okay_friction_vol_4cm_both_con_5_bw_50_4th"],
        "bw_100": ["logs_14_11/real_ddpg_+_30hz_okay_friction_vol_4cm_both_con_5",
                   "logs_14_11/real_ddpg_+_30hz_okay_friction_vol_4cm_both_con_5_3rd",
                   "logs_14_11/real_ddpg_+_30hz_okay_friction_vol_4cm_both_con_5_4th_video"],
    }

    data_eval = {
        "pre": ["logs_13_12/eval_low_friction_pre"],
        "low_1": ["logs_13_12/eval_low_friction_real_20_11_200k"],
        "low_2": ["logs_13_12/eval_low_friction_real_16_11_2nd"],
        "low_3": ["logs_13_12/eval-low_friction_after_freeze"]
    }

    freeze_legend = ['hp1 : actor_freeze critic_freeze 3.5k',
                     'hp2 : actor_freeze5k critic_freeze 3.5k',
                     'hp3 : actor_freeze critic_no_freeze',
                     'hp4 : both_no_freeze']

    lab_frz = ['Friction_settings', 'converging time [k]']
    lab_AC = ['AC_freezing_settings', 'converging time [k]']
    lab_BW = ['Bw_settings', 'converging time [k]']
    lab_eval = ['evaluation', 'converging time [k]']

    # plot(data_fric, lables=lab_frz, name="friction")
    # plot(data_freeze, lables=lab_AC, name="freezing")
    # plot(data_bandwidth, lables=lab_BW, name="bandwidth")

    plot(data_eval, lables=lab_eval, name="evaluation")
