# This file is a modification of the open‑source 'qugen' project: https://github.com/QutacQuantum/qugen
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Anonymous contributors
# Licensed under the Apache License, Version 2.0: https://www.apache.org/licenses/LICENSE-2.0

from itertools import product

import numpy as np


def center_2d(i, j, n):
    ax, bx = 0., 1.
    ay, by = 0., 1.
    return ax + (2 * i + 1) / (2 * n) * (bx - ax), ay + (2 * j + 1) / (2 * n) * (by - ay)


def center(coord, n):
    return np.array(coord) / n + 0.5 / n


def compute_discretization(n_qubits, n_registered):
    format_string = "{:0" + str(n_qubits) + "b}"
    n = 2 ** (n_qubits // n_registered)
    dict_bins = {}
    for k, coordinates in enumerate(product(range(n), repeat=n_registered)):
        dict_bins.update({
            format_string.format(k): [coordinates, center(coordinates, n)]
        })
    return dict_bins
