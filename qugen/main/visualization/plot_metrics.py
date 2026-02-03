import os

import matplotlib.pyplot as plt
import pandas as pd


def plot_metrics_progression(path_to_models, metrics,
                                     max_epoch=None, min_epoch=None):

    if isinstance(metrics, str):
        metrics = [metrics]

    path_to_eval_csv = os.path.join(path_to_models, "evaluation_summary.csv")
    df = pd.read_csv(path_to_eval_csv)

    for metric in metrics:
        no_nans_mask = ~df[metric].isnull()
        plt.plot(df[no_nans_mask]["iteration"], df[no_nans_mask][metric])
        # mark minimum and maximum with red x:
        argmin = df[no_nans_mask][metric].argmin()
        argmax = df[no_nans_mask][metric].argmax()
        plt.scatter(df[no_nans_mask]["iteration"].iloc[argmin], df[no_nans_mask][metric].iloc[argmin])
        plt.scatter(df[no_nans_mask]["iteration"].iloc[argmax], df[no_nans_mask][metric].iloc[argmax])

    return
