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


def plot_cat(dat, categories, ci):
    plt.figure()
    results = {}
    for k, runs in categories.items():
        results[k] = np.array(extract_data(dat, runs, "convergence_time")) / 1000.
    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in results.items()]))
    # max_index = df.idxmax()
    # for i in range(len(max_index)):
    #     df.at[max_index[i], max_index.keys()[i]] = np.NaN
    ax = sns.pointplot(data=df, capsize=.2, join=False, ci=ci)
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


def plot_timeseries_median(dat, run_ids):
    # Get convergence times

    conv_times = {}
    for k, runs in run_ids.items():
        conv_times[k] = np.array(extract_data(dat, runs, "convergence_time")) / 1000.

    i = 0
    for k, runs in run_ids.items():
        median = np.median(conv_times[k])
        idx = np.where(conv_times[k] == median)[0][0]
        results = extract_data(dat, runs, "swing_up_time_series")
        result = results[idx]
        l = list(zip(*[[int(k) / 1000, 1000 - v] for k, v in result.items() if int(k) > 5000]))
        a = sns.lineplot(x=l[0], y=l[1], marker='o', zorder=101)
        plt.scatter(x=l[0][-1], y=l[1][-1], s=200, marker='*', color=a.get_lines()[i].get_color(), zorder=100)
        i += 1

    plt.xlabel('Training time [k steps]')
    plt.ylabel('Consecutive on-target steps $n_{T_e}$')
    # plt.ylabel('Successful on-target steps ${T_s}$')
    lims = plt.xlim()
    plt.hlines(y=750, xmin=0, xmax=lims[1], colors="green", linestyles='--')
    # plt.hlines(y=1000, xmin=0, xmax=lims[1], colors="red", linestyles='--')
    # plt.text(lims[1] // 41, 930, "Fail", {"color": "red"})
    plt.text(lims[1] // 41, 930, "Success", {"color": "green"})
    plt.axhspan(750, 1000, color='green', alpha=0.2)
    plt.xlim(0, lims[1])
    plt.ylim(0, 1000)


if __name__ == "__main__":

    confidence_score = 100

    with open("paper_data.json", "r") as f:
        runs = json.load(f)
    with open("data.json", "r") as f:
        data = json.load(f)

    runs_fric = runs['data_fric_new']
    runs_freeze = runs['data_freeze']
    runs_bw = runs['data_bandwidth']
    runs_bw_no_combined = runs['data_bandwidth_without_cer']
    runs_eval = runs['data_eval']
    runs_dis = runs['dis']

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
    ax = sns.boxplot(data=df, width=.4)
    plt.xticks(range(4), (
        f"Pretrained\n\nFailed {crashes['pre']}/{df.shape[0]}",
        f"Converged\nafter {conv_times['low_1']}k\nFailed {crashes['low_1']}/{df.shape[0]}",
        f"Converged\nafter {conv_times['low_2']}k\nFailed {crashes['low_2']}/{df.shape[0]}",
        f"Converged\nafter {conv_times['low_3']}k\nFailed {crashes['low_3']}/{df.shape[0]}"
    ))
    plt.ylabel("Swing-up-time [steps]")
    plt.ylim(0, plt.ylim()[1])
    plt.tight_layout()
    plt.savefig("eval_plot.png", dpi=300)
    # plt.show()

    ax_fric = plot_cat(data, runs_fric, confidence_score)
    plt.xticks(range(5), ('0', '5', '10', '12', '16'))
    plt.xlabel('Friction factor $k_f$')
    plt.ylabel('Convergence time [k steps]')
    plt.ylim([0, plt.ylim()[1]])
    plt.tight_layout()
    plt.savefig("friction_plot.png", dpi=300)

    ax_freeze = plot_cat(data, runs_freeze, confidence_score)
    # plt.xticks(range(6), (
    #     '$N_c=3500$\n$N_a=5000$',
    #     '$N_c=3500$\n$N_a=3500$',
    #     '$N_c=3500$\n$N_a=3500$\nTD3-delay',
    #     '$N_c=5000$\n$N_a=5000$',
    #     '$N_c=128$\n$N_a=128$',
    #     '$N_c=128$\n$N_a=5000$'
    # ))

    plt.xticks(range(6), (
        '$N_c=128$\n$N_a=128$',
        '$N_c=128$\n$N_a=5000$',
        '$N_c=3500$\n$N_a=3500$',
        '$N_c=3500$\n$N_a=3500$\nTD3-delay',
        '$N_c=3500$\n$N_a=5000$',
        '$N_c=5000$\n$N_a=5000$',
    ))

    plt.xlabel('Optimization delay configurations')
    plt.ylabel('Convergence time [k steps]')
    plt.ylim([0, plt.ylim()[1]])
    plt.tight_layout()
    plt.savefig("freezing_plot.png", dpi=300)

    # ax_bw = plot_cat(data, runs_bw)
    # Adding no combined
    plt.figure()
    df = pd.DataFrame(columns=("convergence_time", "bandwidth", "Method"))
    for k, rs in runs_bw.items():
        d = np.array(extract_data(data, rs, "convergence_time")) / 1000.
        # max_index = d.argmax()
        # d[max_index] = np.NaN
        for c in d:
            df = df.append({"convergence_time": c, "bandwidth": k, "Method": "w CER"}, ignore_index=True)

    for k, rs in runs_bw_no_combined.items():
        d = np.array(extract_data(data, rs, "convergence_time")) / 1000.
        # max_index = d.argmax()
        # d[max_index] = np.NaN
        for c in d:
            df = df.append({"convergence_time": c, "bandwidth": k, "Method": "w/o CER"}, ignore_index=True)

    ax = sns.pointplot(data=df, x="bandwidth", y="convergence_time", hue="Method", capsize=.2, join=False, dodge=0.25,
                       ci=confidence_score)

    # ax = sns.scatterplot(data=df, x="bandwidth", y="convergence_time", hue="Method")
    # ax = sns.boxplot(data=df, x="bandwidth", y="convergence_time", hue="Method")

    ylims = plt.ylim()
    plt.vlines(2.5, 0, ylims[1], colors="black", linestyles="--")
    plt.text(0.5, 3, "Low BW")
    plt.text(4, 3, "High BW")
    plt.xticks(range(7), ('0.06', '0.1', '0.5', '5', '10', '15', '>50'))
    plt.xlabel('Bandwidth [Mbit/s]')
    plt.ylabel('Convergence time [k steps]')
    plt.ylim([0, ylims[1]])
    plt.tight_layout()
    plt.savefig("bandwidth_plot.png", dpi=300)

    fig = plt.figure(figsize=(15, 4))
    plot_timeseries_median(data, runs['data_fric_new'])
    leg = plt.legend(["0", "5", "10", "12", "16"], title="Friction factor", loc='lower right')
    plt.tight_layout()
    plt.savefig("real_good.png", dpi=300)
    plt.show()

    ax_dis = plot_cat(data, runs_dis, confidence_score)
    plt.xticks(range(8), ('0.06', '0.1', '0.5', '5', '10', '15', '>50', '>50 dis'))
    plt.xlabel('Bandwidth (w/o CER) $k_f$')
    plt.ylabel('Convergence time [k steps]')
    plt.vlines(2.5, 0, ylims[1], colors="black", linestyles="--")
    plt.text(0.5, 3, "Low BW")
    plt.text(4, 3, "High BW")
    plt.ylim([0, plt.ylim()[1]])
    plt.tight_layout()
    plt.savefig("bandwidth_dis.png", dpi=300)
