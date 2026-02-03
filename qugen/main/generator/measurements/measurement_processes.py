from typing import Callable

import pennylane as qml


def measure_single_qubit_pauli_z(wires):
    return [qml.expval(qml.PauliZ(wires=w)) for w in wires]


def measure_comp_basis_probs(wires):
    # Measure basis state probabilities:
    return qml.probs(wires=wires)


def measure_comp_basis_amps(*_, **__):
    # Measure basis state amplitudes (always uses all wires):
    return qml.state()


# ### Factory methods to obtain measurement process via key: ###

measurement_process_lookup = {
    "n_pauli_z": measure_single_qubit_pauli_z,
    "comp_basis_probs": measure_comp_basis_probs,
    "comp_basis_amps": measure_comp_basis_amps
}


def measurement_process_factory(key: str) -> Callable:
    """
    Factory function retrieves the measurement process, identified by the given key.
    The measurement process defines the quantities to be measured at the end of the generator quantum circuit.

    Args:
        key (str): The identifier for the measurement process.

    Returns (Callable):
        The measurement process.
    """
    try:
        return measurement_process_lookup[key]
    except KeyError:
        raise ValueError(f"Measurement process ({key}) is not recognized. "
                         f"Valid identifiers are {list(measurement_process_lookup.keys())}")
