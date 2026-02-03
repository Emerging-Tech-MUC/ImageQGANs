# This file is a modification of the open‑source 'qugen' project: https://github.com/QutacQuantum/qugen
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Anonymous contributors
# Licensed under the Apache License, Version 2.0: https://www.apache.org/licenses/LICENSE-2.0

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from qugen.main.generator.discrete_qgan_model_handler import DiscreteQGANModelHandler


class TestDiscreteQGANModelHandler:

    def test_build(self):
        assert True

    def test_save(self):
        # Given
        working_folder = Path("./experiments")

        model = DiscreteQGANModelHandler()

        model.build(model_name='model_name', data_set_name='dataset', n_qubits=8, n_registers=2, circuit_depth=1,
                    save_artefacts=False, slower_progress_update=True)
        model.model = MagicMock()

        # When
        model.model.save(working_folder/"model.npy", overwrite=True)

        # Then
        model.model.save.assert_called_once_with(working_folder/"model.npy", overwrite=True)

    def test_reload(self):
        assert True

    def test_train(self):
        # Given
        dataset = np.array([[0.0, 0.0],
                            [1.0, 1.0],
                            [0.2, 0.3],
                            [0.6, 0.7]
                            ])
        model = DiscreteQGANModelHandler()

        # self.train_model(train_dataset, model)
        model.build(model_name='train_discrete', data_set_name='example', n_qubits=8, n_registers=2, circuit_depth=1,
                    transformation='pit', circuit_type='copula', save_artefacts=False, slower_progress_update=True)
        model.train(train_dataset=dataset,
                    n_epochs=2,
                    initial_learning_rate_generator=1e-3,
                    initial_learning_rate_discriminator=1e-3,
                    batch_size=2)

        # # Then
        assert model.generator_weights.shape[0] == 36

    def test_predict(self):
        np.random.seed(0)
        dataset = np.random.rand(10,2)
        print("dataset", dataset)
        model = DiscreteQGANModelHandler()
        model.build(model_name='predict_discrete', data_set_name='example', n_qubits=8, n_registers=2, circuit_depth=1,
                    transformation='pit', circuit_type='copula', save_artefacts=False)
        model.train(train_dataset=dataset,
                    n_epochs=2,
                    initial_learning_rate_generator=1e-3,
                    initial_learning_rate_discriminator=1e-3,
                    batch_size=2)

        predicted_samples = model.predict(100)

        assert(len(predicted_samples) == 100)

