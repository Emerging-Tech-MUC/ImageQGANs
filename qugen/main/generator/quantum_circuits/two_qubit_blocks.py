import jax
import jax.numpy as jnp
import pennylane as qml

from qugen.main.generator.measurements.measurement_processes import measure_single_qubit_pauli_z


def get_qnode(circuit_depth, n_qubits, block_type, noise_enc='block', block_arrangement='top_down',
              include_next_neighbor=False,
              n_ancilla_qubits=0, measurement_fn=None, noise_tuning=None, skip_init=False, reupload=0):
    if reupload is None:
        reupload = 0
    n_total_qubits = n_qubits + n_ancilla_qubits
    blocks_per_layer = n_total_qubits - 1
    diff_method = "best"
    dev = qml.device("default.qubit", wires=n_total_qubits)

    n_weight_params = 0
    if noise_tuning is not None and (isinstance(noise_tuning, int) or noise_tuning.isdigit()):
        n_modes = int(noise_tuning)
        n_weight_params = 2*n_modes  # per mode scaling and shifting
        idx_noise_shift = -n_modes
        idx_noise_scale = -2*n_modes
        idx_noise_signed_shift = None
    else:
        match noise_tuning:
            case 'scale':
                idx_noise_scale = -1
                idx_noise_shift = None
                idx_noise_signed_shift = None
                n_weight_params += 1
            case 'shift':
                idx_noise_shift = -1
                idx_noise_scale = None
                idx_noise_signed_shift = None
                n_weight_params += 1
            case 'signed_shift':
                idx_noise_signed_shift = -1
                idx_noise_scale = None
                idx_noise_shift = None
                n_weight_params += 1
            case 'both':
                idx_noise_scale = -2
                idx_noise_shift = -1
                idx_noise_signed_shift = None
                n_weight_params += 2
            case 'all':
                idx_noise_scale = -3
                idx_noise_shift = -2
                idx_noise_signed_shift = -1
                n_weight_params += 3
            case None:
                idx_noise_scale = None
                idx_noise_shift = None
                idx_noise_signed_shift = None
            case _:
                raise KeyError(f"Invalid noise tuning method: {noise_tuning}")

    if measurement_fn is None:
        measurement_fn = measure_single_qubit_pauli_z

    match block_type:
        case 'SU':
            n_block_params = -1
            def two_qubit_block(x, wires):
                assert len(wires) == 2
                #assert len(x) == n_block_params
            raise NotImplementedError("SU(4) block not implemented")
        case 'SO':
            n_block_params = 4
            def two_qubit_block(x, wires):
                assert len(wires) == 2
                #assert len(x) == n_block_params
                qml.CNOT(wires=wires)
                qml.RY(x[0], wires=wires[0])
                qml.RY(x[1], wires=wires[1])
                qml.CNOT(wires=wires)
                qml.RY(x[2], wires=wires[0])
                qml.RY(x[3], wires=wires[1])
        case 'SO_full':
            n_block_params = 6
            def two_qubit_block(x, wires):
                assert len(wires) == 2
                #assert len(x) == n_block_params
                qml.RY(x[0], wires=wires[0])
                qml.RY(x[1], wires=wires[1])
                qml.CNOT(wires=wires)
                qml.RY(x[2], wires=wires[0])
                qml.RY(x[3], wires=wires[1])
                qml.CNOT(wires=wires)
                qml.RY(x[4], wires=wires[0])
                qml.RY(x[5], wires=wires[1])
        case 'sparse':
            n_block_params = -1
            def two_qubit_block(x, wires):
                assert len(wires) == 2
                #assert len(x) == n_block_params
            raise NotImplementedError("sparse block not implemented")
        case _:
            raise KeyError(f"Unknown block type {block_type}")

    if include_next_neighbor:
        n_block_params *= 2

    def blocks_layer(xx, wires, reverse=False):
        wire_pairs = list(zip(wires[1:], wires[:-1]))
        if reverse:
            wire_pairs = reversed(wire_pairs)
        if not include_next_neighbor:
            for idx, (j, k) in enumerate(wire_pairs):
                two_qubit_block(xx[idx], wires=(j, k))
        else:
            for idx, (j, k) in enumerate(wire_pairs):
                two_qubit_block(xx[idx, :n_block_params // 2], wires=(j, k))
                i_next_neighbor = j + 1
                if i_next_neighbor not in wires:
                    continue
                two_qubit_block(xx[idx, n_block_params // 2:], wires=(i_next_neighbor, k))


    match noise_enc:
        case 'block':
            noise_enc_fn = blocks_layer
            noise_shape = (blocks_per_layer, n_block_params)
            n_weight_params *= n_block_params
            if idx_noise_scale is not None:
                idx_noise_scale = slice(idx_noise_scale * n_block_params,
                                        idx_noise_shift * n_block_params if idx_noise_shift is not None else None)
            if idx_noise_shift is not None:
                idx_noise_shift = slice(idx_noise_shift * n_block_params,
                                        idx_noise_signed_shift * n_block_params if idx_noise_signed_shift is not None
                                        else None)
            if idx_noise_signed_shift is not None:
                idx_noise_signed_shift = slice(idx_noise_signed_shift * n_block_params, None)

        case 'angle':
            noise_enc_fn = qml.AngleEmbedding
            noise_shape = (n_total_qubits,)
        case _:
            raise KeyError(f"Unknown noise embedding type {noise_enc}")

    def qnode_fn(inputs, weights):
        if not skip_init:
            # initial layer of Hadamard gates to start with a valid FRQI state
            for j in range(n_total_qubits):
                qml.Hadamard(wires=j)

        # initial (first layer) noise if provided
        if isinstance(inputs, tuple):
            inputs, init_inputs = inputs
            noise_enc_fn(init_inputs, wires=range(n_total_qubits))

        for _ in range(reupload + 1):

            # layered circuit (noise re-uploading)
            for i in range(circuit_depth):
                match (block_arrangement):
                    case 'top_down':
                        reverse = False
                    case 'bottom_up':
                        reverse = True
                    case 'mirror':
                        reverse = bool(i % 2)

                qml.Barrier(only_visual=True, wires=range(n_total_qubits))

                # Noise uploading:
                tuned_inputs = inputs
                if noise_tuning is not None and (isinstance(noise_tuning, int) or noise_tuning.isdigit()):
                    n_modes = int(noise_tuning)
                    # assumes the shifts of the unmodulated noise on integers 0, 1, 2, ...
                    # determine modes first:
                    mode_indices = jnp.around(jnp.clip(tuned_inputs, 0., n_modes - 1)).astype(int)
                    shifts = weights[i].T[idx_noise_shift + mode_indices][0]
                    scales = weights[i].T[idx_noise_scale + mode_indices][0]
                    # shift to origin first:
                    tuned_inputs = tuned_inputs - mode_indices
                    # then apply scaling:
                    tuned_inputs = tuned_inputs * scales
                    # then shift by the shifts:
                    tuned_inputs = tuned_inputs + shifts
                else:
                    if idx_noise_signed_shift is not None:
                        tuned_inputs = tuned_inputs + weights[i, :, idx_noise_signed_shift]*jnp.sign(tuned_inputs)
                    if idx_noise_scale is not None:
                        tuned_inputs = tuned_inputs * weights[i, :, idx_noise_scale]
                    if idx_noise_shift is not None:
                        tuned_inputs = tuned_inputs + weights[i, :, idx_noise_shift]
                if noise_enc == 'block':
                    noise_enc_fn(tuned_inputs, wires=range(n_total_qubits), reverse=reverse)
                else:
                    noise_enc_fn(tuned_inputs, wires=range(n_total_qubits))

                qml.Barrier(only_visual=True, wires=range(n_total_qubits))

                # Parameterized blocks:
                blocks_layer(weights[i, :, :n_block_params], wires=range(n_total_qubits), reverse=reverse)

        return measurement_fn(wires=range(n_total_qubits))

    # Shapes of noise inputs and parameters
    init_noise_shape = noise_shape
    params_shape = (circuit_depth,
                    blocks_per_layer + (1 if noise_enc == 'angle' and noise_tuning is not None else 0),
                    n_block_params + n_weight_params)
    # Note that noise tuning with angle noise encoding will result in more parameters than actually used!

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
