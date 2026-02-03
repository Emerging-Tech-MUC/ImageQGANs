# This file is a modification of the open‑source 'qugen' project: https://github.com/QutacQuantum/qugen
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Anonymous contributors
# Licensed under the Apache License, Version 2.0: https://www.apache.org/licenses/LICENSE-2.0

import warnings

import jax
import matplotlib.pyplot as plt
import numpy as np
from fire import Fire

from qugen.main.data.data_handler import load_data
from qugen.main.generator.continuous_qgan_model_handler import ContinuousQGANModelHandler

DEBUG = False
PLOT_DATA = False
PLOT_SAMPLES = False


def main(data_set_name,
         data_set_kwargs=None,
         model_name="continuous",
         circuit_depth=8,
         n_ancilla_qubits=0,
         noise_distr='normal',
         noise_scale=0.1,
         noise_shift=0.0,
         init_noise_distr=None,
         init_noise_scale=None,
         init_noise_shift=None,
         noise_tuning=None,
         transformation='minmax',
         measurement_scheme='comp_basis_probs',
         decoding_scheme='FRQI',
         save_artifacts="samples_10",
         discriminator_name="continuous_fully_connected",
         generator_name="layered_rot_strongly_ent",
         reupload=0,
         gen_init_distr='normal',
         gen_init_scale=1.0,
         gen_init_shift=0.0,
         single_data_point=None,
         n_epochs=1000,
         initial_learning_rate_generator=1e-1,
         initial_learning_rate_discriminator=1e-3,
         adam_b1=0.9,
         adam_b2=0.999,
         batch_size=20,
         discriminator_training_steps=1,
         gan_method="GAN",
         warm_start=False,
         n_channels=1,
         shots=None,
         **kwargs):

    if len(kwargs) > 0:
        warnings.warn(f"Unrecognized arguments: {kwargs}")

    jax.config.update("jax_enable_x64", True)
    #jax.config.update("jax_disable_jit", DEBUG)
    jax.config.update("jax_debug_nans", True)
    np.seterr(invalid="raise")

    data, data_set_name = _load_data(data_set_name, data_set_kwargs)
    n_pixels = data.shape[1] // n_channels
    shape = tuple([int(np.sqrt(n_pixels))] * 2)

    if single_data_point is not None:
        data = data[single_data_point:single_data_point + 1]
        print(data.shape)

    if PLOT_DATA:
        plt.imshow(data[0].reshape(*shape), cmap=plt.get_cmap("gray"), vmin=0, vmax=1)
        plt.show()

    if decoding_scheme == "amplitude":
        n_qubits = int(np.ceil(np.log2(n_pixels*2))) - 1
    else:
        n_qubits = int(np.ceil(np.log2(n_pixels*n_channels*2)))  # single color qubit for FRQI
    print(f"{n_qubits=}")
    model = ContinuousQGANModelHandler()

    # build a new model:
    model.build(
        model_name=model_name,
        data_set=data_set_name,
        n_qubits=n_qubits,
        n_ancilla_qubits=n_ancilla_qubits,
        circuit_depth=circuit_depth,
        noise_distr=noise_distr,
        noise_scale=noise_scale,
        noise_shift=noise_shift,
        init_noise_distr=init_noise_distr,
        init_noise_scale=init_noise_scale,
        init_noise_shift=init_noise_shift,
        noise_tuning=noise_tuning,
        transformation=transformation,
        measurement_scheme=measurement_scheme,
        decoding_scheme=decoding_scheme,
        save_artifacts=save_artifacts,
        discriminator_name=discriminator_name,
        generator_name=generator_name,
        reupload=reupload,
        gen_init_distr=gen_init_distr,
        gen_init_scale=gen_init_scale,
        gen_init_shift=gen_init_shift,
        gan_method=gan_method,
        n_channels=n_channels,
        shots=shots,
    )

    # Warm start the learning
    if warm_start:
        model.warm_start(prev_depth_decrement=int(warm_start))

    # train a quantum generative model:
    model.train(
        data,
        n_epochs=n_epochs,
        initial_learning_rate_generator=initial_learning_rate_generator,
        initial_learning_rate_discriminator=initial_learning_rate_discriminator,
        adam_b1=adam_b1,
        adam_b2=adam_b2,
        batch_size=batch_size,
        discriminator_training_steps=discriminator_training_steps
    )

    if PLOT_SAMPLES:
        number_samples = 100
        samples = model.predict(number_samples).squeeze().reshape(number_samples, *shape)
        for sample_idx in range(number_samples):
            plt.imshow(samples[sample_idx], cmap=plt.get_cmap("gray"), vmin=0, vmax=1)
            plt.savefig('generated_image.png')
            plt.show()

        # evaluate the performance of the trained model:

        evaluation_df = model.evaluate(data)

        # find the model with the minimum KL divergence:

        minimum_kl_data = evaluation_df.loc[evaluation_df["kl_original_space"].idxmin()]
        minimum_kl_calculated = minimum_kl_data["kl_original_space"]
        print(f"{minimum_kl_calculated=}")

    return model.path_to_models


def _load_data(data_set_name, data_set_kwargs=None):
    # Set specific data set parameters:
    if data_set_kwargs is not None and len(data_set_kwargs) > 0:
        print(f"Original data set name: {data_set_name}")
        data_set_name = data_set_name.format_map(data_set_kwargs)
        print(f"Post-processed data set name: {data_set_name}")
    # Load data:
    data_set_path = f"./training_data/{data_set_name}"
    data, _ = load_data(data_set_path)
    if np.any(data > 1):  # Image pixels exceed expected [0, 1] range
        data = data.astype(float)
        data /= 255.
    return data, data_set_name


if __name__ == "__main__":
    Fire(main)
    # CLI example: python train_image_qgan.py --data_set_name bars_and_stripes_s_noise_0.1_2x2_N_1000
