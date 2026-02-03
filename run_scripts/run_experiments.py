import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
EXCLUDE_GPUS = [0, 1, 2, 3]
import numpy as np
import pandas as pd
import traceback
from fire import Fire
import multiprocessing
import filelock
import jax
import time
import cuda_selector
import ast

import train_image_qgan


def _run_experiment(csv_path):

    pid_str = 'p' + str(os.getpid())
    print(f"New experiment started ({pid_str}). Update experiment csv file {csv_path}")
    df = _update_csv(path=csv_path, match_val="...", column="result_path", value=f"...({pid_str})", multi_match='force')
    experiment = df.drop(columns=['result_path', 'version']).to_dict(orient='records')[0]  # convert experiment to dict

    try:
        experiment = _pre_process_experiment_config(experiment)

        try:  # if GPU via JAX, determine the least busy GPU
            with filelock.FileLock("cuda_selector.lock") as lock:
                device_id = int(cuda_selector.auto_cuda('memory', exclude=EXCLUDE_GPUS).split(':')[-1])
                os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
                # os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".75"
                os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
                jax_device = jax.devices("gpu")[0]   # for gpu, only single gpu visible at this point
                print(f"Least busy GPU device: {device_id}")
                time.sleep(5)
        except (RuntimeError, ValueError):  # No CUDA support installed -> CPU fallback
            lock.release()
            jax_device = jax.devices("cpu")[0]

        jax.config.update("jax_default_device", jax_device)
        result_path = train_image_qgan.main(**experiment)
    except Exception as e:
        result_path = "...ERR:" + repr(e)[:255]  # truncate after a max of 100 characters
        print(f"Experiment failed: {result_path}")
        print(traceback.format_exc())

    print(f"Experiment completed {result_path}. Update experiment csv file {csv_path}")
    _update_csv(path=csv_path, match_val=f"...({pid_str})", column="result_path", value=result_path, multi_match='raise')

    return result_path


def _read_experiment_csv(path):
    df = pd.read_csv(path, skip_blank_lines=True)
    df = df.replace([np.nan], [None])
    return df


def _update_csv(path, match_val, column, value, multi_match=None):
    # Save the new result path to the csv file at the corresponding row
    lock = filelock.FileLock(path + ".lock")
    with lock:
        with open(path, 'r+', newline="") as f:  # Set newline is important for to_csv working correctly
            # Read (and pre-process) CSV file:
            df = _read_experiment_csv(f)

            # Match values to determine rows for replacement:
            if callable(match_val):
                match_indexer = match_val(df[column])
            else:
                match_indexer = df[column] == match_val
            n_matches = len(df[match_indexer])

            # Checks before replacement:
            assert n_matches > 0, f"No matches for {match_val} in column {column} in {path} to set value {value}"
            if multi_match == 'raise':
                assert n_matches <= 1, (f"{n_matches} matches for {match_val} in column {column} in {path}. "
                                        f"However, expected only a single match to set value {value}")
            elif multi_match == 'force':
                match_indexer[match_indexer.index != match_indexer.argmax()] = False
                n_matches = len(df[match_indexer])
                assert n_matches == 1

            # Replace value and write to file again:
            df.loc[match_indexer, column] = value
            f.seek(0)
            df.to_csv(f, index=False)

    return df[match_indexer]


def _pre_process_experiment_config(experiment):
    # convert to lists / tuples:
    for k, v in experiment.items():
        try:
            experiment[k] = ast.literal_eval(v)
        except ValueError:
            pass
        except SyntaxError:
            pass

    data_set_kwargs = {k: v for k, v in experiment.items() if k.startswith("data_set_") and k != 'data_set_name'}
    print(f"{data_set_kwargs=}")
    if len(data_set_kwargs) > 0:
        experiment['data_set_kwargs'] = data_set_kwargs
        # Remove from experiment dict:
        for k in data_set_kwargs.keys():
            experiment.pop(k, None)
    return experiment

def main(experiments_csv_path, n_processes=1, check_new_experiments=True, eval_config_csv_path=None):

    while True:  # do-while
        # Read CSV file and update accordingly to indicate experiments being in progress
        print("Experiments to run:")
        try:
            df = _update_csv(path=experiments_csv_path, match_val=lambda x: x.isnull(),
                             column="result_path", value="...")
            assert len(df) > 0
        except AssertionError as e:
            print("None left")
            break
        with pd.option_context("expand_frame_repr", False):
            print(df)
        n_experiments = len(df)

        # Run experiments
        if n_processes == 1:  # run experiments sequentially
            for _ in range(n_experiments):
                _run_experiment(experiments_csv_path)
        else:  # run experiments in parallel
            if n_processes == -1:  # auto-detect
                n_processes = min(n_experiments, multiprocessing.cpu_count())
            with multiprocessing.get_context('spawn').Pool(n_processes) as pool:
                pool.map(_run_experiment, [experiments_csv_path]*n_experiments)

        if not check_new_experiments: # do-while
            break

    if eval_config_csv_path is not None:
        from run_evaluations import main as run_evaluations

        # Evaluate experiments
        run_evaluations(experiments_csv_path=experiments_csv_path,
                        eval_config_csv_path=eval_config_csv_path,
                        n_processes=n_processes)

if __name__ == "__main__":
    Fire(main)
