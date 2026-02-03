import os
import shutil
from glob import glob

import pandas as pd
from fire import Fire


def clean_experiments(experiments_dir="."):
    print(f"Cleaning experiments in directory {experiments_dir}...")

    # Search for csv files (glob) and load them with pandas to get experiment dirs to keep
    experiments_to_keep = []
    experiment_cfg_files = glob(f"{experiments_dir}/*.csv")
    print(f"Keep experiments from the following experiment CSV files:\n{', '.join(experiment_cfg_files)}")
    for experiment_cfg_file in experiment_cfg_files:
        if experiment_cfg_file.endswith("eval.csv"):  # skip evaluation configurations
            continue
        df = pd.read_csv(experiment_cfg_file)
        for experiment_path in df["result_path"]:
            if not isinstance(experiment_path, str):  # completely empty column
                continue
            if experiment_path.startswith("..."):
                if experiment_path.startswith("...ERR:"):
                    continue
                else:
                    raise RuntimeError(f"Experiments are currently running (see {experiment_cfg_file}). Wait til done.")
            experiment_path = os.path.split(experiment_path)[-1]
            experiment_path = os.path.join(experiments_dir, experiment_path)
            experiments_to_keep.append(experiment_path)

    # Delete all directories that do not match
    for experiment_dir in os.listdir(experiments_dir):
        if os.path.isdir(experiment_dir):
            experiment_path = os.path.join(experiments_dir, experiment_dir)
            if experiment_path not in experiments_to_keep:
                print(f"Delete experiment {experiment_dir}...")
                shutil.rmtree(experiment_path)

    # Delete file locks
    experiment_cfg_locks = glob(f"{experiments_dir}/*.csv.lock")
    for experiment_cfg_lock in experiment_cfg_locks:
        print(f"Remove experiment lock {experiment_cfg_lock}...")
        os.remove(experiment_cfg_lock)


if __name__ == '__main__':
    Fire(clean_experiments)