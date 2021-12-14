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


def plot(data, legend=None, lables=None):

    # nd = dict_from_list(data)
    df = pd.DataFrame(data)
    ax = sns.pointplot(data=df, capsize=.2, join=False)
    if lables is not None:
        ax.set_xlabel(lables[0])
        ax.set_ylabel(lables[1])
    if legend is not None:
        ax.legend(legend, loc=2)
    plt.show()
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
    data_all = {
        "no_fr": [25.14, 39.09, 97.23],
        "lo_fr": [38.52, 36.87, 85.06, 15.7, 7.194],
        "ok_fr": [17.77, 16.7, 52.47, 44.74, 18.06, 21.26, 45.5, 41.67, 17.87, 19.78, 15.4, 15.94, 53.28, 52.28, 23.25],
        "hi_fr": [32.06, 79.06, 65.06, 82.49, 29.44, 23.4],
        "over_fr": [61.48, 55.88, 71.41, 108.7, 101.6]
    }

    data_fric = {
        "no_fr": [25.14, 39.09, 97.23],
        "lo_fr": [36.87, 15.7, 7.194],
        "ok_fr": [17.77, 16.7, 18.06],
        "hi_fr": [82.49, 29.44, 23.4],
        "over_fr": [61.48, 108.7, 101.6]
    }
    data_fric = {
        "no_fr": ["logs_15_11/real_ddpg_+_30hz_non_friction_vol_4cm_both_con_5_vol_4_2nd", 39.09, 97.23],
        "lo_fr": [36.87, 15.7, 7.194],
        "ok_fr": [17.77, 16.7, 18.06],
        "hi_fr": [82.49, 29.44, 23.4],
        "over_fr": [61.48, 108.7, 101.6]
    }
    lab_fric = ['frction level', 'steps in [k]']

    data_freeze = {
        "hp1": [17.77, 18.06, 16.7],
        "hp2": [19.78, 15.4, 15.94],
        "hp3": [52.28, 21.26, 23.25],
        "hp4": [53.28, 41.67, 17.87]
    }

    freeze_legend = ['hp1 : actor_freeze critic_freeze 3.5k',
                    'hp2 : actor_freeze5k critic_freeze 3.5k',
                    'hp3 : actor_freeze critic_no_freeze',
                    'hp4 : both_no_freeze']

    lab_frz = ['hyperparams', 'steps in [k]']

    plot(data_fric, lables=lab_fric)
    plot(data_freeze, freeze_legend, lables=lab_frz)
    
    plot_box(data_fric, lables=lab_fric)
    plot_box(data_freeze, freeze_legend, lables=lab_frz)
