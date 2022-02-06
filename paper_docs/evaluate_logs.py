import json

import tqdm
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import tensorflow as tf
import numpy as np

import os


def get_timeseries(event, ts_id):
    _, steps, vals = zip(*event.Tensors(ts_id))
    value = [tf.make_ndarray(val).tolist() for val in vals]
    return steps, value


dirs = [x[0] for x in os.walk('../logs')]

stats = {}

for d in tqdm.tqdm(dirs):
    if not d.endswith('training') and not d.endswith('eval'):
        continue
    event_acc = EventAccumulator(d, size_guidance={'tensors': 100000})
    event_acc.Reload()

    log_name = d[d.find('logs/')+5:d.rfind('/')]

    if log_name not in stats:
        stats[log_name] = {}

    log_stats = stats[log_name]

    if d.endswith('training'):
        continue
    if d.endswith('eval'):
        try:
            s, v = get_timeseries(event_acc, 'swing_up_time')
            if len(s) < 5:
                continue
        except KeyError:
            continue

        if np.all(np.array(v[-5:]) <= 250.0):
            log_stats["convergence_time"] = s[-1]
        else:
            log_stats["convergence_time"] = -1

        log_stats["swing_up_time_series"] = {}
        for step, value in zip(s, v):
            log_stats["swing_up_time_series"][step] = value

with open('data.json', 'w') as f:
    json.dump(stats, f, indent=4)
