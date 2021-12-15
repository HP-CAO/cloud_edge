import json
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


def extract_data(dat, run_ids, data_id):
    if isinstance(run_ids, str):
        return dat[run_ids][data_id]
    return [dat[run_id][data_id] for run_id in run_ids]


def plot_cat(dat, categories):
    plt.figure()
    results = {}
    for k, runs in categories.items():
        results[k] = np.array(extract_data(dat, runs, "convergence_time")) / 1000.
    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in results.items()]))
    ax = sns.pointplot(data=df, capsize=.2, join=False)
    return ax


def plot_timeseries(dat, run_ids):
    plt.figure()
    results = extract_data(dat, run_ids, "swing_up_time_series")
    for result in results:
        l = list(zip(*[[int(k), v] for k, v in result.items() if int(k) > 5000]))
        sns.lineplot(x=l[0], y=l[1], marker='o')

    plt.xlabel('Steps')
    plt.ylabel('Swing-up-time [steps]')
    lims = plt.xlim()
    plt.hlines(y=250, xmin=0, xmax=lims[1], colors="green", linestyles='--')
    plt.hlines(y=1000, xmin=0, xmax=lims[1], colors="red", linestyles='--')
    plt.text(lims[1] // 17, 930, "Fail", {"color": "red"})
    plt.text(lims[1] // 17, 110, "Success", {"color": "green"})
    plt.axhspan(0, 250, color='green', alpha=0.2)
    plt.xlim(0, lims[1])
    plt.ylim(0, plt.ylim()[1])


if __name__ == "__main__":
    with open("paper_data.json", "r") as f:
        runs = json.load(f)
    with open("data.json", "r") as f:
        data = json.load(f)

    runs_fric = runs['data_fric']
    runs_freeze = runs['data_freeze']
    runs_bw = runs['data_bandwidth']
    runs_eval = runs['data_eval']

    plt.figure()
    results = {}
    crashes = {}
    conv_times = {}
    for k, r in runs_eval.items():
        results[k] = extract_data(data, r, "swing_up_time_series")[0]
        crashes[k] = list(results[k].values()).count(1000.0)
    for i, k in enumerate(runs_fric["lo_fr"]):
        conv_times[f"low_{i + 1}"] = extract_data(data, k, "convergence_time") // 1000
    df = pd.DataFrame(results)
    df[df == 1000.0] = np.NaN
    # ax = sns.pointplot(data=df, capsize=.2, join=False)
    ax = sns.boxplot(data=df)
    plt.xticks(range(4), (
        f"Pretrained\n\nCrashed {crashes['pre']}/{df.shape[0]}",
        f"Converged\nafter {conv_times['low_1']}k\nCrashed {crashes['low_1']}/{df.shape[0]}",
        f"Converged\nafter {conv_times['low_2']}k\nCrashed {crashes['low_2']}/{df.shape[0]}",
        f"Converged\nafter {conv_times['low_3']}k\nCrashed {crashes['low_3']}/{df.shape[0]}"
    ))
    plt.ylabel("Swing-up-time [steps]")
    plt.tight_layout()
    plt.savefig("eval_plot.png", dpi=300)

    ax_fric = plot_cat(data, runs_fric)
    plt.xticks(range(5), ('0', '5', '10', '12', '16'))
    plt.xlabel('Friction factor')
    plt.ylabel('Convergence time [k steps]')
    plt.ylim([0, plt.ylim()[1]])
    plt.tight_layout()
    plt.savefig("friction_plot.png", dpi=300)

    ax_freeze = plot_cat(data, runs_freeze)
    plt.xticks(range(4), (
        'Actor 5k\n Critic 3.5k',
        'Actor 3.5k\n Critic 3.5k',
        'Actor 5k\n Critic 128',
        'Actor 128\n Critic 128'
    ))
    plt.xlabel('Freezing combinations')
    plt.ylabel('Convergence time [k steps]')
    plt.ylim([0, plt.ylim()[1]])
    plt.tight_layout()
    plt.savefig("freezing_plot.png", dpi=300)

    ax_bw = plot_cat(data, runs_bw)
    plt.xticks(range(6), ('0.06', '0.1', '1', '5', '50', '>100'))
    plt.xlabel('Bandwidth [Mbit/s]')
    plt.ylabel('Convergence time [k steps]')
    plt.ylim([0, plt.ylim()[1]])
    plt.tight_layout()
    plt.savefig("bandwidth_plot.png", dpi=300)

    plot_timeseries(data, runs['data_fric']['ok_fr'])

    plt.legend(["Run 1", "Run 2", "Run 3"])
    plt.savefig("real_okayplot.png", dpi=300)

    plot_timeseries(data, runs['data_fric']['lo_fr'])
    plt.legend(["Run 1", "Run 2", "Run 3"])
    plt.savefig("real_lowplot.png", dpi=300)
