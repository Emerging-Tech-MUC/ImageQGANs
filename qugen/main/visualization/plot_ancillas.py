import os
import typing
from datetime import datetime
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml


def plot_ancilla_probs_training_progression(path_to_models, n_ancilla_qubits, is_probs=True,
                                            n_snapshots=10, n_samples_per_epoch=1, max_epoch=None, min_epoch=None,
                                            sort_fn=None):

    print(f"Experiment path to models: {path_to_models} "
          f"(modified last {datetime.fromtimestamp(os.path.getmtime(path_to_models)) : %Y-%m-%d %H:%M:%S})")
    assert os.path.exists(path_to_models), f"Path to training artifacts {path_to_models} does not exist! {os.getcwd()}"

    fig, axs = plt.subplots(nrows=n_samples_per_epoch, ncols=n_snapshots,
                            squeeze=False, sharex=True, sharey=True, figsize=(14, 3*n_samples_per_epoch))

    measurement_out_paths = glob("measurement_outputs_iteration*.npy", root_dir=path_to_models)

    epochs = sorted([int(os.path.basename(f).split('=')[1].split('.')[0]) for f in measurement_out_paths])
    if min_epoch is None:
        min_epoch = min(epochs)
    if max_epoch is None:
        max_epoch = max(epochs)
    epochs = [e for e in epochs if min_epoch <= e <= max_epoch]
    # Picks the epochs to plot evenly spaced between min and max epoch (assuming epochs provided are evenly spaced!)
    epoch_indices = list(map(round, np.linspace(min_epoch, len(epochs) - 1, n_snapshots)))
    plot_iters = [epochs[i] for i in epoch_indices]
    print(plot_iters)

    for i_col, it in enumerate(plot_iters):
        measurement_out_path = os.path.join(path_to_models, f"measurement_outputs_iteration={it}.npy")
        measurement_outputs = np.load(measurement_out_path)
        assert measurement_outputs.shape[0] >= n_samples_per_epoch
        n_pixel_data = measurement_outputs.shape[1] // 2 ** n_ancilla_qubits

        joint_dist = measurement_outputs.reshape(-1, n_pixel_data, 2 ** n_ancilla_qubits)  # separate
        joint_dist = joint_dist.transpose((0, 2, 1))  # innermost axis holds measures per image
        ancilla_marginals = joint_dist.sum(axis=2, keepdims=False)  # get marginal probs of ancillas

        for i_row in range(n_samples_per_epoch):
            dist = ancilla_marginals[i_row].flatten()
            # histogram for marginals:
            ax = typing.cast(plt.Axes, axs[i_row, i_col])
            if sort_fn is not None:
                dist = dist.flatten().tolist()
                dist = zip(*sorted(enumerate(dist), key=sort_fn))[1]
                dist = np.array(dist)
            ax.bar(x=np.arange(dist.size), height=dist.flatten())
            # put binary x-labels:
            #ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%b'))
            # plot horizontal bar of uniform distribution in red:
            ax.hlines(1/2**n_ancilla_qubits, xmin=-1., xmax=2**n_ancilla_qubits, color='r')

            ax.hlines(np.mean(dist), xmin=-1., xmax=2**n_ancilla_qubits, color='b')
            ax.hlines(np.median(dist), xmin=-1., xmax=2**n_ancilla_qubits, color='b', linestyle='dotted')
            ax.hlines(np.percentile(dist, q=75), xmin=-1., xmax=2**n_ancilla_qubits, color='b', linestyle='dashed')
            ax.hlines(np.percentile(dist, q=25), xmin=-1., xmax=2**n_ancilla_qubits, color='b', linestyle='dashed')
            # set x limits:
            ax.set_xlim(left=-1., right=2**n_ancilla_qubits)
            ax.set_title(f"Epoch {it}")
            m = ancilla_metric(metric='entropy',
                               full_state=measurement_outputs[i_row], n_ancilla_qubits=n_ancilla_qubits,
                               is_probs=is_probs)
            # Put entropy and percentage of max entropy (log2(2**n_ancilla_qubits)))
            ax.set_xlabel(f"Entropy {m:.1f} ({m / n_ancilla_qubits * 100:.0f}%)")

    plt.tight_layout()


def ancilla_metric(full_state, n_ancilla_qubits, metric, is_probs=False):
    # Possible metrics 'entropy'. Future could include concurrence, entanglement witness, etc.
    if is_probs:
        full_state = np.sqrt(full_state)
    n_qubits_total = round(np.log2(full_state.shape[-1]))
    n_data_qubits = n_qubits_total - n_ancilla_qubits
    red_state = qml.math.reduce_statevector(full_state, indices=range(n_data_qubits, n_qubits_total), check_state=True)
    if metric == 'entropy':
        return qml.math.vn_entropy(red_state, indices=range(n_ancilla_qubits), base=2., check_state=True)
    elif metric == 'mutual_information':  # for pure states, MI is twice the von Neumann entropy
        return 2*ancilla_metric(full_state=full_state, n_ancilla_qubits=n_ancilla_qubits,
                                metric='entropy', is_probs=False)
    else:
        raise NotImplementedError(f"Metric {metric} not implemented for ancilla metrics.")
