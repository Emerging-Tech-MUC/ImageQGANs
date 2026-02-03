from functools import partial
from typing import Callable

import jax.numpy as jnp
import jax.random
import numpy as np


def jax_scaled(jax_rnd_fn, scale=1., shift=0., **kwargs):
    z = jax_rnd_fn(**kwargs)
    z *= scale
    z += shift
    return z


def multi_mode(noise_dist_name, n_modes=2, **kwargs):
    prng_key = kwargs['key']
    shape = kwargs.get('shape', (1,))
    batch_size = shape[0] if len(shape) == 2 else 1
    n_dim = shape[-1]
    multi_var = isinstance(n_modes, int)

    if multi_var:  # along diagonal
        possible_shifts = np.asarray(kwargs.get('shift', np.linspace(0, 2*jnp.pi, n_modes + 1)[:-1]))
        shifts = jnp.tile(jax.random.choice(key=prng_key, a=possible_shifts, shape=(batch_size, 1)), n_dim)
    else:
        if len(n_modes) == 1:  # same number of modes for all dimensions, i.e., independent multivariate dists per dim:
            n_modes = n_modes * n_dim
        assert n_dim == len(n_modes)
        shifts = jnp.zeros(shape)
        for i, m in enumerate(n_modes):
            possible_shifts = np.asarray(kwargs.get('shift', np.linspace(0, 2*jnp.pi, m + 1)[:-1]))
            shifts_i = jax.random.choice(key=prng_key + i, a=possible_shifts, shape=(batch_size,), replace=True)
            shifts = shifts.at[:, i].set(shifts_i)

    noise_sample_fn = noise_sample_fn_factory(noise_dist_name)
    kwargs.pop('shift', None)  # remove shift since resulting noise sample is shifted here
    z = noise_sample_fn(**kwargs)

    z += shifts

    return z.squeeze()


def noise_concat(noise_dist_name, n_concat, **kwargs):
    shape = kwargs['shape']
    if len(shape) != 2:
        raise NotImplementedError(f"Only 2D noise shape dimensions are supported for concatenation (got {len(shape)}). "
                                  f"Shape must describe (batch_size, n_noise_components)")
    batch_size = shape[0]
    z_dim = shape[-1]  # last dim is noise vector dimension
    split_shapes = [(batch_size, z_dim // n_concat)]*(n_concat - 1)
    split_shapes.append((batch_size, z_dim - sum(s[-1] for s in split_shapes)))
    prng_subkeys = jax.random.split(kwargs['key'], n_concat)
    noise_sample_fn = noise_sample_fn_factory(noise_dist_name)

    split_zs = []
    for i in range(n_concat):
        kwargs['shape'] = split_shapes[i]
        kwargs['key'] = prng_subkeys[i]
        split_zs.append(noise_sample_fn(**kwargs))
    z = jnp.concat(split_zs, axis=1)

    assert z.shape == shape, f"Concatenated noise shape {z.shape} does not match expected shape {shape}."

    return z


def rand_x(shift, **kwargs):
    # interpret shift as integer:
    n_flipped_qubits = round(shift)
    n_qubits = kwargs.get('shape', (1,))[-1]
    n_modes = [2] * n_flipped_qubits + [1] * (n_qubits - n_flipped_qubits)
    return multi_mode(noise_dist_name='normal', n_modes=n_modes, **kwargs)


# ### Factory methods to obtain a function via key to sample the noise input vector from: ###

noise_sample_fn_lookup = {
    "normal": partial(jax_scaled, jax_rnd_fn=jax.random.normal),
    "uniform": partial(jax_scaled, jax_rnd_fn=jax.random.uniform),
}
# automatically add multi-mode variants:
noise_sample_fn_lookup.update({
    f"{noise_dist_name}_multi_mode": partial(multi_mode, noise_dist_name=noise_dist_name)
    for noise_dist_name in noise_sample_fn_lookup.keys()
})
noise_sample_fn_lookup.update({
    f"{noise_dist_name}_multi_mode_indep": partial(multi_mode, noise_dist_name=noise_dist_name, n_modes=[2])
    for noise_dist_name in noise_sample_fn_lookup.keys()
})
noise_sample_fn_lookup.update({
    f"{noise_dist_name}_concat_{n_concat}": partial(noise_concat, noise_dist_name=noise_dist_name, n_concat=n_concat)
    for noise_dist_name in noise_sample_fn_lookup.keys()
    for n_concat in [2, 3, 4, 5]
})
noise_sample_fn_lookup.update({
    "rand_x": rand_x,
})


def noise_sample_fn_factory(key: str) -> Callable:
    try:
        return noise_sample_fn_lookup[key]
    except KeyError:
        try:
            const_val = float(key)
            return partial(jax_scaled, jax_rnd_fn=jax.random.normal, scale=0., shift=const_val)
        except ValueError:
            pass
        raise ValueError(f"Noise distribution ({key}) is not recognized. "
                         f"Valid identifiers are {list(noise_sample_fn_lookup.keys())}")


