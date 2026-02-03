import multiprocessing
import os
import traceback
from functools import partial

import pandas as pd
from fire import Fire

from qugen.main.generator.continuous_qgan_model_handler import ContinuousQGANModelHandler
from run_experiments import _read_experiment_csv
from train_image_qgan import _load_data


def _evaluate(result_path, metrics=None, n_iters_skip=0, n_samples=None):
    if metrics is None and not isinstance(result_path, str):
        result_path, metrics = result_path  # unpack everything from the first argument (e.g., via multiprocessing Pool)

    try:
        # Set up model handler:
        model = ContinuousQGANModelHandler()
        model_name = os.path.split(result_path)[-1]
        model.reload(model_name=model_name, epoch=0)

        # Load data:
        training_data, loaded_data_set_name = _load_data(data_set_name=model.data_set)
        assert model.data_set == loaded_data_set_name, \
            (f"Data set name mismatch: "
             f"Requested data set ({model.data_set}) vs actually loaded data set ({loaded_data_set_name})")

        # Run evaluation:
        model.evaluate(training_data, metrics=metrics, n_iters_skip=n_iters_skip, n_samples=n_samples)
    except Exception as e:
        print(f"Experiment failed: {repr(e)}")
        print(traceback.format_exc())
        pass


def main(experiments_csv_path, eval_config_csv_path, n_processes=1, n_iters_skip=0, n_samples=None):
    # Read (and pre-process) CSV file:
    df_experiments = _read_experiment_csv(experiments_csv_path)

    if eval_config_csv_path == 'auto':  # Infer the evaluation config file from experiments config automatically
        eval_config_csv_path = experiments_csv_path.replace('.csv', '_eval.csv')

    df_eval_configs = pd.read_csv(eval_config_csv_path)

    print(f"Evaluation configurations ({eval_config_csv_path}):")
    with pd.option_context("expand_frame_repr", False):
        print(df_eval_configs)

    metrics = []
    for metric_row in df_eval_configs.itertuples(index=False):
        metric_row = metric_row._asdict()
        # filter empty metric kwargs
        metric_row = {k: v for k, v in metric_row.items()
                       if v is not None and not pd.isnull(v) and (not isinstance(v, str) or v != '')}
        if len(metric_row) == 1:  # only the key is left -> use this key (str) directly instead of singleton dict
            metric_row = metric_row['key']
        metrics.append(metric_row)

    # Filter for completed experiments:
    result_paths = [p for p in df_experiments["result_path"]
                        if isinstance(p, str) and not p.startswith("...")]
    n_experiments = len(result_paths)
    print(f"Evaluate {n_experiments} experiments ({experiments_csv_path})...")

    # Run experiments
    if n_processes == 1:  # run experiments sequentially
        for result_path in result_paths:
            _evaluate(result_path, metrics=metrics, n_iters_skip=n_iters_skip, n_samples=n_samples)
    else:  # run experiments in parallel
        if n_processes == -1:  # auto-detect
            n_processes = min(n_experiments, multiprocessing.cpu_count())
        print(f"... multi-processed via {n_processes} processes")
        with multiprocessing.Pool(n_processes) as pool:
            metrics_rep = [metrics] * len(result_paths) # pack the metrics into the first argument
            map_fn = partial(_evaluate, n_iters_skip=n_iters_skip, n_samples=n_samples)
            pool.map(map_fn, zip(result_paths, metrics_rep))

if __name__ == "__main__":
    Fire(main)
