# This file is a modification of the open‑source 'qugen' project: https://github.com/QutacQuantum/qugen
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Anonymous contributors
# Licensed under the Apache License, Version 2.0: https://www.apache.org/licenses/LICENSE-2.0

import flax.linen as nn_jax

class Discriminator(nn_jax.Module):

    is_critic: bool = False
    hidden_layer_dis: int = 64

    @nn_jax.compact
    def __call__(self, x):

        # zero-center input (assumes [0, 1] data interval)
        x = 2*(x - 0.5)

        x = nn_jax.Dense(features=self.hidden_layer_dis)(x)
        x = nn_jax.relu(x)
        x = nn_jax.Dense(features=2*self.hidden_layer_dis)(x)
        x = nn_jax.relu(x)
        x = nn_jax.Dense(features=1)(x)
        if not self.is_critic:  # Only apply sigmoid activation if discriminator does not act as a critic
            x = nn_jax.sigmoid(x)
        return x

