from functools import partial
from typing import Callable

from qugen.main.generator.quantum_circuits.color_rotations import get_qnode as get_color_rotations_qnode
from qugen.main.generator.quantum_circuits.continuous_circuits import get_qnode as get_layered_rot_strongly_ent_qnode
from qugen.main.generator.quantum_circuits.two_qubit_blocks import get_qnode as get_two_qubit_blocks_qnode

# ### Factory methods to obtain a generator via key: ###

generator_lookup = {
    "layered_rot_strongly_ent": get_layered_rot_strongly_ent_qnode,

    "SU4_blocks": partial(get_two_qubit_blocks_qnode, block_type='SU', noise_enc='block'),
    "SO4_blocks": partial(get_two_qubit_blocks_qnode, block_type='SO', noise_enc='block'),
    "SO4_full_blocks": partial(get_two_qubit_blocks_qnode, block_type='SO_full', noise_enc='block'),
    "sparse_blocks": partial(get_two_qubit_blocks_qnode, block_type='sparse', noise_enc='block'),
    "SU4_blocks_noise_angle": partial(get_two_qubit_blocks_qnode, block_type='SU', noise_enc='angle'),
    "SO4_blocks_noise_angle": partial(get_two_qubit_blocks_qnode, block_type='SO', noise_enc='angle'),
    "SO4_full_blocks_noise_angle": partial(get_two_qubit_blocks_qnode, block_type='SO_full', noise_enc='angle'),
    "sparse_blocks_noise_angle": partial(get_two_qubit_blocks_qnode, block_type='sparse', noise_enc='angle'),
    "SU4_blocks_next_neighbor": partial(get_two_qubit_blocks_qnode, block_type='SU', noise_enc='block', include_next_neighbor=True),
    "SO4_blocks_next_neighbor": partial(get_two_qubit_blocks_qnode, block_type='SO', noise_enc='block', include_next_neighbor=True),
    "SO4_full_blocks_next_neighbor": partial(get_two_qubit_blocks_qnode, block_type='SO_full', noise_enc='block', include_next_neighbor=True),
    "sparse_blocks_next_neighbor": partial(get_two_qubit_blocks_qnode, block_type='sparse', noise_enc='block', include_next_neighbor=True),
    "SU4_blocks_noise_angle_next_neighbor": partial(get_two_qubit_blocks_qnode, block_type='SU', noise_enc='angle', include_next_neighbor=True),
    "SO4_blocks_noise_angle_next_neighbor": partial(get_two_qubit_blocks_qnode, block_type='SO', noise_enc='angle', include_next_neighbor=True),
    "SO4_full_blocks_noise_angle_next_neighbor": partial(get_two_qubit_blocks_qnode, block_type='SO_full', noise_enc='angle', include_next_neighbor=True),
    "sparse_blocks_noise_angle_next_neighbor": partial(get_two_qubit_blocks_qnode, block_type='sparse', noise_enc='angle', include_next_neighbor=True),

    "SU4_blocks_bottom_up": partial(get_two_qubit_blocks_qnode, block_type='SU', noise_enc='block', block_arrangement='bottom_up'),
    "SO4_blocks_bottom_up": partial(get_two_qubit_blocks_qnode, block_type='SO', noise_enc='block', block_arrangement='bottom_up'),
    "SO4_full_blocks_bottom_up": partial(get_two_qubit_blocks_qnode, block_type='SO_full', noise_enc='block', block_arrangement='bottom_up'),
    "sparse_blocks_bottom_up": partial(get_two_qubit_blocks_qnode, block_type='sparse', noise_enc='block', block_arrangement='bottom_up'),
    "SU4_blocks_noise_angle_bottom_up": partial(get_two_qubit_blocks_qnode, block_type='SU', noise_enc='angle', block_arrangement='bottom_up'),
    "SO4_blocks_noise_angle_bottom_up": partial(get_two_qubit_blocks_qnode, block_type='SO', noise_enc='angle', block_arrangement='bottom_up'),
    "SO4_full_blocks_noise_angle_bottom_up": partial(get_two_qubit_blocks_qnode, block_type='SO_full', noise_enc='angle', block_arrangement='bottom_up'),
    "sparse_blocks_noise_angle_bottom_up": partial(get_two_qubit_blocks_qnode, block_type='sparse', noise_enc='angle', block_arrangement='bottom_up'),
    "SU4_blocks_next_neighbor_bottom_up": partial(get_two_qubit_blocks_qnode, block_type='SU', noise_enc='block', block_arrangement='bottom_up', include_next_neighbor=True),
    "SO4_blocks_next_neighbor_bottom_up": partial(get_two_qubit_blocks_qnode, block_type='SO', noise_enc='block', block_arrangement='bottom_up', include_next_neighbor=True),
    "SO4_full_blocks_next_neighbor_bottom_up": partial(get_two_qubit_blocks_qnode, block_type='SO_full', noise_enc='block', block_arrangement='bottom_up', include_next_neighbor=True),
    "sparse_blocks_next_neighbor_bottom_up": partial(get_two_qubit_blocks_qnode, block_type='sparse', noise_enc='block', block_arrangement='bottom_up', include_next_neighbor=True),
    "SU4_blocks_noise_angle_next_neighbor_bottom_up": partial(get_two_qubit_blocks_qnode, block_type='SU', noise_enc='angle', block_arrangement='bottom_up', include_next_neighbor=True),
    "SO4_blocks_noise_angle_next_neighbor_bottom_up": partial(get_two_qubit_blocks_qnode, block_type='SO', noise_enc='angle', block_arrangement='bottom_up', include_next_neighbor=True),
    "SO4_full_blocks_noise_angle_next_neighbor_bottom_up": partial(get_two_qubit_blocks_qnode, block_type='SO_full', noise_enc='angle', block_arrangement='bottom_up', include_next_neighbor=True),
    "sparse_blocks_noise_angle_next_neighbor_bottom_up": partial(get_two_qubit_blocks_qnode, block_type='sparse', noise_enc='angle', block_arrangement='bottom_up', include_next_neighbor=True),

    "SU4_blocks_mirror": partial(get_two_qubit_blocks_qnode, block_type='SU', noise_enc='block', block_arrangement='mirror'),
    "SO4_blocks_mirror": partial(get_two_qubit_blocks_qnode, block_type='SO', noise_enc='block', block_arrangement='mirror'),
    "SO4_full_blocks_mirror": partial(get_two_qubit_blocks_qnode, block_type='SO_full', noise_enc='block', block_arrangement='mirror'),
    "sparse_blocks_mirror": partial(get_two_qubit_blocks_qnode, block_type='sparse', noise_enc='block', block_arrangement='mirror'),
    "SU4_blocks_noise_angle_mirror": partial(get_two_qubit_blocks_qnode, block_type='SU', noise_enc='angle', block_arrangement='mirror'),
    "SO4_blocks_noise_angle_mirror": partial(get_two_qubit_blocks_qnode, block_type='SO', noise_enc='angle', block_arrangement='mirror'),
    "SO4_full_blocks_noise_angle_mirror": partial(get_two_qubit_blocks_qnode, block_type='SO_full', noise_enc='angle', block_arrangement='mirror'),
    "sparse_blocks_noise_angle_mirror": partial(get_two_qubit_blocks_qnode, block_type='sparse', noise_enc='angle', block_arrangement='mirror'),
    "SU4_blocks_next_neighbor_mirror": partial(get_two_qubit_blocks_qnode, block_type='SU', noise_enc='block', block_arrangement='mirror', include_next_neighbor=True),
    "SO4_blocks_next_neighbor_mirror": partial(get_two_qubit_blocks_qnode, block_type='SO', noise_enc='block', block_arrangement='mirror', include_next_neighbor=True),
    "SO4_full_blocks_next_neighbor_mirror": partial(get_two_qubit_blocks_qnode, block_type='SO_full', noise_enc='block', block_arrangement='mirror', include_next_neighbor=True),
    "sparse_blocks_next_neighbor_mirror": partial(get_two_qubit_blocks_qnode, block_type='sparse', noise_enc='block', block_arrangement='mirror', include_next_neighbor=True),
    "SU4_blocks_noise_angle_next_neighbor_mirror": partial(get_two_qubit_blocks_qnode, block_type='SU', noise_enc='angle', block_arrangement='mirror', include_next_neighbor=True),
    "SO4_blocks_noise_angle_next_neighbor_mirror": partial(get_two_qubit_blocks_qnode, block_type='SO', noise_enc='angle', block_arrangement='mirror', include_next_neighbor=True),
    "SO4_full_blocks_noise_angle_next_neighbor_mirror": partial(get_two_qubit_blocks_qnode, block_type='SO_full', noise_enc='angle', block_arrangement='mirror', include_next_neighbor=True),
    "sparse_blocks_noise_angle_next_neighbor_mirror": partial(get_two_qubit_blocks_qnode, block_type='sparse', noise_enc='angle', block_arrangement='mirror', include_next_neighbor=True),
}
# color_rot schemes can use other general ansaetze for the position qubit entanglement
generator_lookup.update({
    f"color_rot_{k}": partial(get_color_rotations_qnode, entangling_qnode_getter=v)
    for k, v in generator_lookup.items()
})
generator_lookup.update({
    f"color_rot_skip_{k}": partial(get_color_rotations_qnode, entangling_qnode_getter=v, uncomputing_mode='skip')
    for k, v in generator_lookup.items()
})
generator_lookup.update({
    f"color3_rot_{k}": partial(get_color_rotations_qnode, entangling_qnode_getter=v, n_color_qubits=3)
    for k, v in generator_lookup.items()
})
generator_lookup.update({
    f"color3_rot_skip_{k}": partial(get_color_rotations_qnode, entangling_qnode_getter=v, uncomputing_mode='skip',
                                    n_color_qubits=3)
    for k, v in generator_lookup.items()
})
generator_lookup.update({
    f"color3i_rot_{k}": partial(get_color_rotations_qnode, entangling_qnode_getter=v, n_color_qubits=3,
                                independent_channels=True)
    for k, v in generator_lookup.items()
})
generator_lookup.update({
    f"color3i_rot_skip_{k}": partial(get_color_rotations_qnode, entangling_qnode_getter=v, uncomputing_mode='skip',
                                    n_color_qubits=3, independent_channels=True)
    for k, v in generator_lookup.items()
})

generator_lookup.update({
    f"color3addr_rot_{k}": partial(get_color_rotations_qnode, entangling_qnode_getter=v, n_color_qubits=3,
                                color_as_address_qubits=True)
    for k, v in generator_lookup.items()
})
generator_lookup.update({
    f"color3addr_rot_skip_{k}": partial(get_color_rotations_qnode, entangling_qnode_getter=v, uncomputing_mode='skip',
                                    n_color_qubits=3, color_as_address_qubits=True)
    for k, v in generator_lookup.items()
})


def generator_factory(key: str) -> Callable:
    try:
        return generator_lookup[key]
    except KeyError:
        raise ValueError(f"Generator ({key}) is not recognized. "
                         f"Valid identifiers are {list(generator_lookup.keys())}")
