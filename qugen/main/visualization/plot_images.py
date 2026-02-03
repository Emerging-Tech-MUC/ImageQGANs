import collections
import os
import typing
from datetime import datetime
from glob import glob

import cv2
import fire
import jax
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

from qugen.main.generator.continuous_qgan_model_handler import ContinuousQGANModelHandler


def plot_images_training_progression(path_to_models=None, experiment_base_path=None,
                                     n_images=10, n_samples_per_epoch=5, max_epoch=None, min_epoch=None,
                                     n_channels=1, is_hsv_color=False):

    if path_to_models is None:  # Auto-detect latest experiment directory
        path_to_models = max(glob(os.path.join(experiment_base_path, '*/')), key=os.path.getmtime)
        print(f"Auto-detected latest experiment!")
    print(f"Experiment path to models: {path_to_models} "
          f"(modified last {datetime.fromtimestamp(os.path.getmtime(path_to_models)) : %Y-%m-%d %H:%M:%S})")

    fig, axs = plt.subplots(nrows=n_samples_per_epoch, ncols=n_images, squeeze=False, sharex=True,
                            figsize=(n_images, n_samples_per_epoch + 2))

    assert os.path.exists(path_to_models), f"Path to training artifacts {path_to_models} does not exist! {os.getcwd()}"

    sample_file_paths = glob("samples_iteration*.npy", root_dir=path_to_models)
    epochs = sorted([int(os.path.basename(f).split('=')[1].split('.')[0]) for f in sample_file_paths])
    if min_epoch is None:
        min_epoch = min(epochs)
    if max_epoch is None:
        max_epoch = max(epochs)
    epochs = [e for e in epochs if min_epoch <= e <= max_epoch]
    # Picks the epochs to plot evenly spaced between min and max epoch (assuming epochs provided are evenly spaced!)
    epoch_indices = list(map(round, np.linspace(min_epoch, len(epochs) - 1, n_images)))
    plot_iters = [epochs[i] for i in epoch_indices]
    print(plot_iters)

    for i_col, it in enumerate(plot_iters):
        sample_file_path = os.path.join(path_to_models, f"samples_iteration={it}.npy")
        samples = np.load(sample_file_path)
        assert samples.shape[0] >= n_samples_per_epoch
        n_pixels = samples.shape[1] // n_channels
        img_size = int(np.sqrt(n_pixels))

        for i_sample in range(n_samples_per_epoch):
            sample = samples[i_sample]
            ax = typing.cast(plt.Axes, axs[i_sample, i_col])
            ax.set_xticks([])
            ax.set_xlabel(f"Avg {np.mean(sample):.2f}")
            ax.axes.get_yaxis().set_visible(False)
            sample = sample.reshape(img_size, img_size, n_channels)
            if is_hsv_color:
                sample *= 255
                # convert from 0 to 255 to degree (0 to 180)
                sample[:, :, 0] = sample[:, :, 0] / 255 * 180
                sample = sample.astype(np.uint8)
                sample = cv2.cvtColor(sample, cv2.COLOR_HSV2RGB)
                sample = sample / 255
            ax.imshow(sample, cmap=plt.get_cmap("gray") if n_channels == 1 else None,
                      vmin=0, vmax=1, interpolation='nearest')

            if i_sample == 0:  # First row
                ax.set_title(f"Epoch {it}")

    plt.tight_layout()

def _ax_remove_ticks(ax: plt.Axes):
    """Remove ticks from the axes."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

def plot_images_by_input_modes(path_to_models, epoch=None, n_per_mode=1, seed=None, batch_size=2, n_channels=1,
                               compact_subplots=None, invert_colors=False, apply_pca_per_mode=None, save_path=None, mode_order=None):

    cmap_name = "gray_r" if invert_colors else "gray"

    # If no epoch is specified, use the last one:
    sample_file_paths = glob("parameters_training_iteration*.pickle", root_dir=path_to_models)
    if epoch is None:
        epoch = max([int(os.path.basename(f).split('=')[1].split('.')[0]) for f in sample_file_paths])

    # Reload model for sampling images:
    model_handler = ContinuousQGANModelHandler()
    model_name = os.path.basename(path_to_models)
    print(f"Loading images for {model_name=} iteration {epoch}...")

    orig_cwd = os.getcwd()
    for chdir in [".", "apps/logistics/", ".."]:
        try:
            os.chdir(chdir)
            model_handler.reload(model_name=model_name, epoch=epoch)
            break
        except FileNotFoundError as e:
            print(f"Failed to load model {model_name} at epoch {epoch} in {os.getcwd()}. "
                  f"Trying alternative directory.")
            # print full error message
            print(e)

    n_modes = model_handler.metadata['noise_tuning']
    pre_computed_path = os.path.join(path_to_models, "images_by_mode", f"{epoch=}_{n_per_mode=}_{seed=}.npz")

    if isinstance(n_per_mode, int):
        noise_mode_dist_names = None
    else:
        if isinstance(n_per_mode, float):
            noise_mode_dist_names = [str(n_per_mode)]
        else:
            noise_mode_dist_names = list(map(str, n_per_mode))
        n_per_mode = len(noise_mode_dist_names)


    assert isinstance(n_modes, int), "Only new noise tuning mode supported"
    assert model_handler.noise_distr == 'normal_multi_mode', "Only normal multi mode noise sampling supported"

    if seed is not None:
        model_handler.random_key = jax.random.PRNGKey(seed)
    all_samples = [[] for _ in range(n_modes)]  # holds samples separated by mode
    modes_completed = [False] * n_modes

    print(f"Pre-computed image samples path: {pre_computed_path}")
    if not os.path.exists(pre_computed_path):
        print("Sampling images...")
        with tqdm(total=n_modes * n_per_mode, desc="Sample model") as pbar:
            while True:
                if noise_mode_dist_names is not None:
                    model_handler.noise_sample_fn.keywords['noise_dist_name'] = noise_mode_dist_names[0]

                samples, info = model_handler.sample(batch_size, return_noise=True)
                noise = info['noise']
                i_modes = noise.mean(axis=1).clip(0, n_modes - 1).round().astype(int)
                for sample, i_mode in zip(samples, i_modes):
                    if not modes_completed[i_mode]:
                        all_samples[i_mode].append(sample)
                        pbar.update(1)
                        if len(all_samples[i_mode]) == n_per_mode:
                            modes_completed[i_mode] = True
                if all(modes_completed):
                    if noise_mode_dist_names is None or len(noise_mode_dist_names) == 0:
                        break
                    else:
                        noise_mode_dist_names.pop(0)
                        modes_completed = [False] * n_modes
                        all_samples = [samples_mode_i[:-len(noise_mode_dist_names)] for samples_mode_i in all_samples]

        # store
        os.makedirs(os.path.dirname(pre_computed_path), exist_ok=True)
        np.savez_compressed(pre_computed_path, **dict(zip(map(str, list(range(n_modes))), all_samples)))
    else:
        print("Load pre-computed image samples...")
        all_samples_dict = np.load(pre_computed_path)
        for i_str, samples in all_samples_dict.items():
            all_samples[int(i_str)] = samples

    # Change back to original cwd
    os.chdir(orig_cwd)

    if mode_order is not None:
        assert sorted(mode_order) == list(range(n_modes)), f"Invalid mode order {mode_order} for {n_modes} modes"
        all_samples = [all_samples[i] for i in mode_order]

    if apply_pca_per_mode is not None:
        n_per_mode = 3*apply_pca_per_mode if apply_pca_per_mode > 0 else 1
        all_samples_pca = np.empty((n_modes, n_per_mode, all_samples[0][0].size))
        for i_mode in range(n_modes):
            all_samples_pca[i_mode] = pca_images(np.asarray(all_samples[i_mode]), std_factor=3,
                                                 n_components=apply_pca_per_mode).reshape(n_per_mode, -1)
        all_samples = all_samples_pca

    if compact_subplots is None:
        fig, axs = plt.subplots(nrows=n_modes, ncols=n_per_mode, figsize=(n_per_mode, n_modes + 3), squeeze=False)
    else:
        assert isinstance(compact_subplots, collections.abc.Iterable) and len(compact_subplots) == 4
        nrows, ncols, nrows_inset, ncols_inset = compact_subplots
        assert nrows_inset * ncols_inset == n_per_mode
        assert nrows * ncols == n_modes
        figsize = (ncols * ncols_inset, nrows * nrows_inset)
        fig, super_axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False)
        axs = []
        inset_margin_total = 0.05
        inset_margin_h = inset_margin_total / ncols_inset
        inset_margin_v = inset_margin_total / nrows_inset
        inset_width = 1 / ncols_inset - inset_margin_h
        inset_height = 1 / nrows_inset - inset_margin_v
        for i_row in range(nrows):
            for i_col in range(ncols):
                super_ax = super_axs[i_row, i_col]
                inset_axes = []
                for i_row_inset in range(nrows_inset):
                    for i_col_inset in range(ncols_inset):
                        inset_ax = super_ax.inset_axes([i_col_inset * (inset_width + inset_margin_h),
                                                        1 - (i_row_inset + 1) * (inset_height + inset_margin_v),
                                                        inset_width,
                                                        inset_height],)
                        inset_axes.append(inset_ax)
                axs.append(inset_axes)
                _ax_remove_ticks(super_ax)
        axs = np.array(axs)

    for i_axs, samples in enumerate(all_samples):
        for i_col, sample in enumerate(samples):
            ax = axs[i_axs, i_col]
            n_pixels = sample.size // n_channels
            img_size = int(np.sqrt(n_pixels))
            sample = sample.reshape(img_size, img_size, n_channels).squeeze()
            _ax_remove_ticks(ax)
            ax.imshow(sample, cmap=plt.get_cmap(cmap_name) if n_channels == 1 else None,
                      vmin=0, vmax=1, interpolation='nearest')
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved figure to {save_path}")

    # Return the figure and axes for further customization if needed, and also return the samples
    return fig, axs, all_samples


def pca_images(images, n_components, std_factor=1., pick_closest=True, return_explained_var_ratio=False):
    # Fit PCA
    pca = PCA(n_components=n_components)  # just first 2 components for now
    X_pca = pca.fit_transform(images)

    # Get mean and components
    mean_image = pca.mean_

    varied_imgs = []
    explained_var_ratios = []

    if n_components == 0:  # Only mean
        varied_imgs.append((mean_image, mean_image, mean_image))
    else:
        for i in range(n_components):
            pc = pca.components_[i]
            explained_variance_perc = pca.explained_variance_ratio_[i] * 100
            explained_var_ratios.append(explained_variance_perc)
            if return_explained_var_ratio:
                print(f"Component {i}: Explained variance ratio {explained_variance_perc}%")
            # Standard deviation along this PC
            std_dev = np.std(X_pca[:, i])
            # Mean ± 1 std dev along this PC
            img_plus = mean_image + std_factor * std_dev * pc
            img_minus = mean_image - std_factor * std_dev * pc
            varied_imgs.append((img_minus, mean_image, img_plus))

    if pick_closest:  # pick the closest image in the dataset to each varied image
        resampled_imgs = np.empty_like(varied_imgs)
        for i in range(len(varied_imgs)):
            for j in range(3):
                target_img = varied_imgs[i][j]
                dists = np.linalg.norm(images - target_img, axis=1)
                closest_img = images[np.argmin(dists)]
                resampled_imgs[i][j] = closest_img
        varied_imgs = resampled_imgs

    varied_imgs = np.array(varied_imgs)

    if n_components == 0:
        varied_imgs = varied_imgs[:, 1, :]

    if return_explained_var_ratio:
        return varied_imgs, explained_var_ratios
    return varied_imgs


if __name__ == '__main__':
    fire.Fire(plot_images_training_progression)
    plt.savefig("fig.pdf")
    plt.show()
