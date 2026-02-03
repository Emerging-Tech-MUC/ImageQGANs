import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qml

from qugen.main.generator.measurements.measurement_processes import measure_single_qubit_pauli_z


def get_qnode(circuit_depth, n_qubits, entangling_qnode_getter, entangling_depth=None,
              n_ancilla_qubits=0, measurement_fn=None, noise_tuning=None, uncomputing_mode='adjoint', reupload=0,
              n_color_qubits=1, independent_channels=False, color_as_address_qubits=False):
    if color_as_address_qubits:  # set color qubits as address qubits
        tmp_n_color_qubits = n_color_qubits
        n_color_qubits = 1
    if independent_channels:
        assert n_color_qubits == 3
    if reupload is None:
        reupload = 0
    n_address_qubits = n_qubits - n_color_qubits
    n_total_qubits = n_qubits + n_ancilla_qubits  # n_color_qubits are already included in n_qubits
    n_rotations = n_address_qubits * n_color_qubits
    if entangling_depth is None:
        assert not isinstance(circuit_depth, int)
        circuit_depth, entangling_depth = circuit_depth
    diff_method = "best"
    dev = qml.device("default.qubit", wires=n_total_qubits)

    if measurement_fn is None:
        measurement_fn = measure_single_qubit_pauli_z

    # Get entangling qnodes:
    entangling_qnodes = []
    entangling_qnodes_param_shape = None
    entangling_qnodes_noise_shape = None
    for _ in range(circuit_depth):
        e_qnode, e_noise_shape, _, e_param_shape = entangling_qnode_getter(circuit_depth=entangling_depth,
                                                                           n_qubits=n_address_qubits,
                                                                           noise_tuning=noise_tuning,
                                                                           measurement_fn=(lambda wires: None),
                                                                           skip_init=True)
        e_qnode = qml.workflow.construct_tape(e_qnode._fun)
        entangling_qnodes.append(e_qnode)
        assert entangling_qnodes_noise_shape is None or entangling_qnodes_noise_shape == e_noise_shape
        assert entangling_qnodes_param_shape is None or entangling_qnodes_param_shape == e_param_shape
        entangling_qnodes_noise_shape = e_noise_shape
        entangling_qnodes_param_shape = e_param_shape

    def qnode_fn(inputs, weights):
        # The qnode is first constructed such that the last qubit is the color qubit, will be remapped later

        # initial layer of Hadamard gates to start with a valid FRQI state
        for j in range(n_total_qubits):
            qml.Hadamard(wires=j)

        for _ in range(reupload + 1):
            # layered circuit
            for i in range(circuit_depth):
                qml.Barrier(only_visual=True, wires=range(n_total_qubits))
                weights_i =  weights[i][:-n_rotations].reshape(entangling_qnodes_param_shape)
                for op in entangling_qnodes[i](inputs, weights_i).circuit:
                    qml.apply(op)
                qml.Barrier(only_visual=True, wires=range(n_total_qubits))

                # Apply controlled rotation:
                for j in range(n_rotations):
                    rot_angle_i = weights[i][-j - 1]
                    i_color_qubit = j // n_address_qubits + n_total_qubits - n_color_qubits
                    i_address_qubit = j % n_address_qubits
                    if independent_channels:
                        match i_color_qubit - n_total_qubits + n_color_qubits:
                            case 0:  # red
                                qml.ctrl(qml.RY, control=(i_address_qubit, n_total_qubits - 1, n_total_qubits - 2),
                                         control_values=(1, 0, 0))(rot_angle_i, wires=n_total_qubits - 3)
                            case 1:  # green
                                qml.ctrl(qml.RY, control=(i_address_qubit, n_total_qubits - 1, n_total_qubits - 2),
                                         control_values=(1, 1, 0))(rot_angle_i, wires=n_total_qubits - 3)
                            case 2:  # blue
                                qml.ctrl(qml.RY, control=(i_address_qubit, n_total_qubits - 1, n_total_qubits - 2),
                                         control_values=(1, 0, 1))(rot_angle_i, wires=n_total_qubits - 3)
                    else:
                        qml.CRY(rot_angle_i, wires=(i_address_qubit, i_color_qubit))

            # Un-compute:
            if uncomputing_mode == 'adjoint':
                for i in reversed(range(circuit_depth)):
                    qml.Barrier(only_visual=True, wires=range(n_total_qubits))
                    weights_i =  weights[i][:-n_rotations].reshape(entangling_qnodes_param_shape)
                    for op in entangling_qnodes[i](inputs, weights_i).adjoint().circuit:
                        qml.apply(op)

        return measurement_fn(wires=range(n_total_qubits))

    # Shapes of noise inputs and parameters
    noise_shape = entangling_qnodes_noise_shape
    init_noise_shape = noise_shape
    params_shape = (circuit_depth, int(np.prod(jnp.array(entangling_qnodes_param_shape))) + n_rotations)

    qnode = qml.QNode(
        qnode_fn,
        device=dev,
        diff_method=diff_method,
        interface="jax"
    )

    if color_as_address_qubits:
        # Re-map the wires such that color qubit(s) will be first:
        qnode = qml.map_wires(qnode, dict(zip(range(n_total_qubits),
                                              ((np.arange(n_total_qubits) + tmp_n_color_qubits) % n_total_qubits).tolist())))
        # Fix order of color qubits:
        assert tmp_n_color_qubits == 3
        qnode = qml.map_wires(qnode, {2: 0, 1: 2, 0: 1})
    else:
        # Re-map the wires such that color qubit will be first:
        qnode = qml.map_wires(qnode, dict(zip(range(n_total_qubits),
                                              ((np.arange(n_total_qubits) + n_color_qubits) % n_total_qubits).tolist())))

    # Test
    dummy_noise_inputs = jnp.ones(noise_shape)
    dummy_init_noise_inputs = jnp.ones(init_noise_shape)
    dummy_params = jnp.zeros(params_shape)
    specs = qml.specs(qnode, level="device")(inputs=(dummy_noise_inputs, dummy_init_noise_inputs), weights=dummy_params)
    print("Generator circuit specs:\\n", specs)

    # Compile QNode
    qnode = jax.jit(qnode)

    return qnode, noise_shape, init_noise_shape, params_shape
