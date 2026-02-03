import os
import shutil
from glob import glob

import pandas as pd
from fire import Fire
from tqdm import tqdm


def export_experiments(experiments_csv_path=None, skip_n_iterations=0, overwrite=False):

    if experiments_csv_path is None:  # loop over all
        experiments_csv_path_list = glob("./runs_*.csv")

    if isinstance(experiments_csv_path, str):
        experiments_csv_path_list = [experiments_csv_path]

    final_storage = 0.

    for experiments_csv_path in tqdm(experiments_csv_path_list, desc="Export CSV files"):

        print(f"Exporting experiments of CSV {experiments_csv_path}...")

        # Create new directory for the new experiments with the same name as the csv file and at same place
        experiments_dir = os.path.dirname(experiments_csv_path)
        experiments_dir = os.path.join(experiments_dir, os.path.basename(experiments_csv_path).replace(".csv", ""))
        if skip_n_iterations > 0:
            experiments_dir += f"_{skip_n_iterations + 1}_compr"  # compression ratio in percent

        if overwrite and os.path.exists(experiments_dir):  # check if exists and remove
            # remove experiments dir recusively
            shutil.rmtree(experiments_dir)
        os.makedirs(experiments_dir, exist_ok=False)

        # Copy CSV file into directory
        shutil.copyfile(experiments_csv_path, os.path.join(experiments_dir, os.path.basename(experiments_csv_path)))

        # Copy all experiments
        df = pd.read_csv(experiments_csv_path)
        for experiment_path in tqdm(df["result_path"], desc="Process experiments"):
            # completely empty column or unfinished
            if not isinstance(experiment_path, str) or experiment_path.startswith("..."):
                continue
            experiment_path = os.path.split(experiment_path)[-1]  # assumes cwd to be in experiments base directory
            experiment_path_dest = os.path.join(experiments_dir, experiment_path)

            # Create the folder for the directory at the destination
            os.makedirs(experiment_path_dest, exist_ok=False)

            # Copy all subdirectories (recursively) in experiment path completely:
            for file in os.listdir(experiment_path):
                file = os.path.join(experiment_path, file)
                if os.path.isdir(file):
                    shutil.copytree(file, os.path.join(experiment_path_dest, os.path.basename(file)))
                else:
                    # Copy all files but filter out the ones that are not needed
                    # (iteration not equal to a multiple of skip_n_iterations+1)
                    if "_iteration=" in file and int(file.split("=")[-1].split('.')[0]) % (skip_n_iterations + 1) != 0:
                        continue  # skip
                    # Copy file
                    shutil.copyfile(file, os.path.join(experiment_path_dest, os.path.basename(file)))
        print(f"Exported experiments into folder {experiments_dir}")

        # Compress directory into zip archive
        print(f"Compress experiments...")
        shutil.make_archive(experiments_dir, 'zip', experiments_dir)
        shutil.rmtree(experiments_dir)
        file_size = os.path.getsize(experiments_dir + '.zip') / 1000 / 1000 / 1000
        print(f"Compressed experiments into archive {experiments_dir}.zip ({file_size} GB)")
        final_storage += file_size

    print(f"Total storage {final_storage} GB")

if __name__ == '__main__':
    Fire(export_experiments)