# This file is a modification of the open‑source 'qugen' project: https://github.com/QutacQuantum/qugen
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Anonymous contributors
# Licensed under the Apache License, Version 2.0: https://www.apache.org/licenses/LICENSE-2.0

import glob
import os
import re
import shutil
from abc import ABC, abstractmethod
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from qugen.main.data.metrics_factory import metrics_factory


class BaseModelHandler(ABC):
    """
    It implements the interface for each of the models handlers (continuous QGAN/QCBM and discrete QGAN/QCBM),
    which includes building the models, training them, saving and reloading them, and generating samples from them.
    """

    def __init__(self):
        """"""
        self.device_configuration = None

    @abstractmethod
    def build(self, *args, **kwargs) -> "BaseModelHandler":
        """
        Define the architecture of the model. Weights initialization is also typically performed here.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, file_path: Path, overwrite: bool = True) -> "BaseModelHandler":
        """
        Saves the model weights to a file.

        Parameters:
            file_path (pathlib.Path): destination file for model weights
            overwrite (bool): Flag indicating if any existing file at the target location should be overwritten
        """
        raise NotImplementedError

    @abstractmethod
    def reload(self, file_path: Path) -> "BaseModelHandler":
        """
        Loads the model from a set of weights.

        Parameters:
            file_path (pathlib.Path): source file for the model weights
        """
        raise NotImplementedError

    @abstractmethod
    def train(self, *args) -> "BaseModelHandler":
        """
        Perform training of the model.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, *args) -> np.array:
        """
        Draw samples from the model.
        """
        raise NotImplementedError

    def evaluate(
        self, train_dataset_original_space: np.ndarray, metrics=None, number_bins=16, plot=False,
            n_samples=None, n_iters_skip=0
    ) -> pd.DataFrame:



        if metrics is None:
            metrics = ["kl_div"]
        elif isinstance(metrics, str) or isinstance(metrics, dict):
            metrics = [metrics]
        metrics_col_names = []
        for m in metrics:
            if isinstance(m, dict):
                m = m['key']
            if n_samples is not None and n_samples != len(train_dataset_original_space):  # not full data set
                m_col_name = f"{m}_n{n_samples}"
            else:
                m_col_name = m
            metrics_col_names.append(m_col_name)
        # for varying number of samples used to calculate the metric, a different column is created in the df/csv

        path_to_eval_csv = f"{self.path_to_models}/evaluation_summary.csv"
        # Check if CSV exists
        if os.path.exists(path_to_eval_csv):

            # Create copy of original evaluation file (stores important results from the training process):
            path_to_eval_csv_copy = f"{self.path_to_models}/evaluation_summary_orig_copy.csv"
            if not os.path.exists(path_to_eval_csv_copy):
                shutil.copyfile(path_to_eval_csv, path_to_eval_csv_copy)

            df = pd.read_csv(path_to_eval_csv)
        else:
            # create empty CSV
            df = pd.DataFrame({'iteration': []})

        # Add columns for new metrics if they do not exist
        df = df.reindex(
            df.columns.union(metrics_col_names, sort=False), axis=1, fill_value=None)

        # Check if any metrics left to be evaluated:
        if not df[df['iteration'] % (n_iters_skip + 1) == 0][metrics_col_names].isnull().any(axis=None):
            print("No new evaluations need to be performed.")
            return df

        parameters_all_training_iterations = glob.glob(
            f"{self.path_to_models}/parameters_training_iteration=*"
        )
        parameters_all_training_iterations.sort(key=lambda s: (len(s), s))  # sort them by iteration

        progress = tqdm(range(len(parameters_all_training_iterations)))
        progress.set_description("Evaluating")
        for it in progress:
            parameters_path = parameters_all_training_iterations[it]
            iteration = re.search(
                "parameters_training_iteration=(.*).(pickle|npy)",
                os.path.basename(parameters_path),
            ).group(1)
            iteration = int(iteration)

            if iteration % (n_iters_skip + 1) != 0:
               continue

            if iteration not in df['iteration']:
                df.loc[len(df)] = {k: iteration if k == 'iteration' else None for k in df.columns}

            self.reload(self.model_name, iteration)

            train_dataset_original_space = train_dataset_original_space[:n_samples]
            synthetic_dataset_original_space = self.sample(n_samples=len(train_dataset_original_space))

            for metric, metric_col_name in zip(metrics, metrics_col_names):
                if not pd.isna(df[df['iteration'] == iteration][metric_col_name].item()):
                    continue

                if isinstance(metric, dict):
                    assert 'key' in metric.keys()
                    metric_fn = metrics_factory(**metric)
                    metric = metric['key']
                else:
                    metric_fn = metrics_factory(metric)

                m = metric_fn(train_dataset_original_space, synthetic_dataset_original_space)
                iteration_idx = df[df['iteration'] == iteration].index.item()
                df.at[iteration_idx, metric_col_name] = m

            # Show latest metrics in progress bar:
            last_iter_metrics = df[df['iteration'] == iteration].drop('iteration', axis='columns').to_dict(orient='records')[0]
            progress.set_postfix(
                refresh=True,
                **last_iter_metrics
            )

        if plot:
            fig = plt.figure()
            raise NotImplementedError("Plotting evaluation results not implemented")

        df.to_csv(path_to_eval_csv, index=False)
        return df
