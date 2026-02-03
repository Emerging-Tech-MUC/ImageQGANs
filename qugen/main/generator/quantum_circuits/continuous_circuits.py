# This file is a modification of the open‑source 'qugen' project: https://github.com/QutacQuantum/qugen
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Anonymous contributors
# Licensed under the Apache License, Version 2.0: https://www.apache.org/licenses/LICENSE-2.0


import jax
import jax.numpy as jnp
import pennylane as qml

from qugen.main.generator.measurements.measurement_processes import measure_single_qubit_pauli_z


def get_qnode(circuit_depth, n_qubits, n_ancilla_qubits=0, measurement_fn=None, noise_tuning=None, skip_init=False,
              reupload=0):
    if reupload is None:
        reupload = 0
    n_total_qubits = n_qubits + n_ancilla_qubits
    diff_method = "best"
    dev = qml.device("default.qubit", wires=n_total_qubits)
    n_innermost_params = 3

    if noise_tuning is not None and (isinstance(noise_tuning, int) or noise_tuning.isdigit()):
        n_modes = int(noise_tuning)
        n_innermost_params += 2*n_modes  # per mode scaling and shifting
        idx_noise_shift = -n_modes
        idx_noise_scale = -2*n_modes
        idx_noise_signed_shift = None
    else:
        match noise_tuning:
            case 'scale':
                idx_noise_scale = -1
                idx_noise_shift = None
                idx_noise_signed_shift = None
                n_innermost_params += 1
            case 'shift':
                idx_noise_shift = -1
                idx_noise_scale = None
                idx_noise_signed_shift = None
                n_innermost_params += 1
            case 'signed_shift':
                idx_noise_signed_shift = -1
                idx_noise_scale = None
                idx_noise_shift = None
                n_innermost_params += 1
            case 'both':
                idx_noise_scale = -2
                idx_noise_shift = -1
                idx_noise_signed_shift = None
                n_innermost_params += 2
            case 'all':
                idx_noise_scale = -3
                idx_noise_shift = -2
                idx_noise_signed_shift = -1
                n_innermost_params += 3
            case None:
                idx_noise_scale = None
                idx_noise_shift = None
                idx_noise_signed_shift = None
            case _:
                raise KeyError(f"Invalid noise tuning method: {noise_tuning}")

    if measurement_fn is None:
        measurement_fn = measure_single_qubit_pauli_z

    def qnode_fn(inputs, weights):

        if not skip_init:
            # init ancilla qubits in superposition
            for j in range(n_qubits, n_total_qubits):
                qml.Hadamard(wires=j)

        if isinstance(inputs, tuple):  # initial (first layer) noise provided
            inputs, init_inputs = inputs
            qml.AngleEmbedding(init_inputs, wires=range(n_total_qubits))

        for _ in range(reupload + 1):

            for i in range(circuit_depth):

                tuned_inputs = inputs
                if noise_tuning is not None and (isinstance(noise_tuning, int) or noise_tuning.isdigit()):
                    n_modes = int(noise_tuning)
                    # assumes the shifts of the unmodulated noise on integers 0, 1, 2, ...
                    # determine modes first:
                    mode_indices = jnp.around(jnp.clip(tuned_inputs, 0., n_modes - 1)).astype(int)
                    shifts = weights[i, 0, :, 3:].T[idx_noise_shift + mode_indices][0]
                    scales = weights[i, 0, :, 3:].T[idx_noise_scale + mode_indices][0]
                    # shift to origin first:
                    tuned_inputs = tuned_inputs - mode_indices
                    # then apply scaling:
                    tuned_inputs = tuned_inputs * scales
                    # then shift by the shifts:
                    tuned_inputs = tuned_inputs + shifts
                else:
                    if idx_noise_signed_shift is not None:
                        tuned_inputs = tuned_inputs + weights[i,0,:,idx_noise_signed_shift]*jnp.sign(tuned_inputs)
                    if idx_noise_scale is not None:
                        tuned_inputs = tuned_inputs * weights[i,0,:,idx_noise_scale]
                    if idx_noise_shift is not None:
                        tuned_inputs = tuned_inputs + weights[i,0,:,idx_noise_shift]
                qml.AngleEmbedding(tuned_inputs, wires=range(n_total_qubits))
                qml.StronglyEntanglingLayers(weights[i,:,:,:3], wires=range(n_total_qubits))

        return measurement_fn(wires=range(n_total_qubits))

    # Shapes of noise inputs and parameters
    noise_shape = (n_total_qubits,)
    init_noise_shape = (n_total_qubits,)
    params_shape = (circuit_depth, 1, n_total_qubits, n_innermost_params)

    qnode = qml.QNode(
        qnode_fn,
        device=dev,
        diff_method=diff_method,
        interface="jax"
    )

    # Test
    dummy_noise_inputs = jnp.ones(noise_shape)
    dummy_init_noise_inputs = jnp.ones(init_noise_shape)
    dummy_params = jnp.zeros(params_shape)
    specs = qml.specs(qnode, level="device")(inputs=(dummy_noise_inputs, dummy_init_noise_inputs), weights=dummy_params)
    print("Generator circuit specs:\\n", specs)

    # Compile QNode
    qnode = jax.jit(qnode)

    return qnode, noise_shape, init_noise_shape, params_shape


if __name__ == "__main__":
    circuit_depth = 7
    n_qubits = 3
    qnode, *_ = get_qnode(circuit_depth, n_qubits)
    noise = jnp.array([0.5, -0.5])
    key = jax.random.PRNGKey(4)
    weights = jax.random.normal(key, shape=(circuit_depth, 1, n_qubits, 3)) * 2*jnp.pi - jnp.pi
    grad = jax.grad(lambda x: sum(qnode(noise, x)))(weights)
    print(jnp.min(jnp.abs(grad)))