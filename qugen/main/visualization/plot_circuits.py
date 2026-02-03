import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
from fire import Fire

from qugen.main.generator.quantum_circuits import generator_factory


def plot_circuit(generator_name, circuit_depth, n_qubits, n_ancilla_qubits=0, decimals=1,
                 decompose=False, parameters_path=None, parameter_epoch=None, save_fig_path=None,
                 **kwargs):
    get_qnode = generator_factory(generator_name)
    qnode, noise_shape, _, params_shape = get_qnode(circuit_depth, n_qubits,
                                                    measurement_fn=None, n_ancilla_qubits=n_ancilla_qubits,
                                                    **kwargs)
    n_params = np.prod(params_shape).item()
    n_noise = np.prod(noise_shape).item()

    noise_labels = np.array([f"$z_{{{i}}}$" for i in range(1, n_noise + 1)]).reshape(noise_shape)
    if parameters_path is not None and parameter_epoch is not None:
        full_path = os.path.join(parameters_path, f"parameters_training_iteration={parameter_epoch}.pickle")
        with open(full_path, "rb") as file:
            params, _ = pickle.load(file)
        assert params.size == n_params
        params = (np.mod(params + np.pi, 2 * np.pi) - np.pi) / np.pi  # map to -pi, pi and remove pi
        formatter = np.vectorize(lambda p: f"${p:.1f}\\pi$")
        param_labels = formatter(params)
    else:
        param_labels = np.array([f"$\\theta_{{{i}}}$" for i in range(1, n_params + 1)])
    param_labels = param_labels.reshape(params_shape)

    print(qnode)
    if decompose:
        if isinstance(decompose, int):
            max_expansion = decompose
        else:
            max_expansion = None
        circ = qml.transforms.decompose(qnode._fun, gate_set={qml.CNOT, qml.RX, qml.RY, qml.RZ, qml.H,
                                                              qml.CRX, qml.CRY, qml.CRZ},
                                        max_expansion=max_expansion)
    else:
        circ = qnode._fun

    if decimals is None or decimals == 0:
        noise_labels = np.zeros_like(noise_labels, dtype=float)
        param_labels = np.zeros_like(param_labels, dtype=float)
    fig, ax = qml.draw_mpl(circ, decimals=decimals)(noise_labels, param_labels)
    title = f"{generator_name} ({circuit_depth=}, {n_qubits=}, {n_ancilla_qubits=})"
    fig.suptitle(title, fontsize=16)
    if save_fig_path is not None:
        plt.savefig(save_fig_path)
    plt.show()


if __name__ == "__main__":
    Fire(plot_circuit)
