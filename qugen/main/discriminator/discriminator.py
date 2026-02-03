# This file is a modification of the open‑source 'qugen' project: https://github.com/QutacQuantum/qugen
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Anonymous contributors
# Licensed under the Apache License, Version 2.0: https://www.apache.org/licenses/LICENSE-2.0

import flax.linen as nn_jax


class Discriminator_JAX(nn_jax.Module):
    @nn_jax.compact
    def __call__(self, x):
        x = nn_jax.Dense(
            2 * x.shape[1],
            kernel_init=nn_jax.initializers.variance_scaling(
                scale=10, mode="fan_avg", distribution="uniform"
            ),
        )(x)
        x = nn_jax.leaky_relu(x)
        x = nn_jax.Dense(
            1,
            kernel_init=nn_jax.initializers.variance_scaling(
                scale=10, mode="fan_avg", distribution="uniform"
            ),
        )(x)
        x = nn_jax.leaky_relu(x)
        return nn_jax.sigmoid(x)
