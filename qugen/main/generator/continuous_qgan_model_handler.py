# This file is a modification of the open‑source 'qugen' project: https://github.com/QutacQuantum/qugen
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Anonymous contributors
# Licensed under the Apache License, Version 2.0: https://www.apache.org/licenses/LICENSE-2.0

import hashlib
import json
import os
import pickle
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pennylane as qml
import yaml
from packaging import version
from tqdm import tqdm

# Get the current installed version of PennyLane
pennylane_version = qml.__version__

import jax
import jax.numpy as jnp
import optax
import numpy as np
from glob import glob

from qugen.main.generator.base_model_handler import BaseModelHandler
from qugen.main.generator.quantum_circuits import generator_factory
from qugen.main.data.helper import CustomDataset
from qugen.main.discriminator import discriminator_factory
from qugen.main.data.data_handler import PITNormalizer, MinMaxNormalizer
from qugen.main.generator.measurements import measurement_process_factory, decoder_factory
from qugen.main.generator.noise_input import noise_sample_fn_factory


jax.config.update("jax_enable_x64", False)
jax.config.update("jax_compilation_cache_dir", "jit_compiled")


class ContinuousQGANModelHandler(BaseModelHandler):
    """
    Parameters:

    """

    def __init__(self):
        """Initialize the parameters specific to this model handler by assigning defaults to all attributes which should immediately be available across all methods."""
        super().__init__()
        self.n_qubits = None
        self.n_ancilla_qubits = None
        self.circuit_depth = None
        self.weights = None
        self.generator_weights = None
        self.discriminator_weights = None
        self.performed_trainings = 0
        self.circuit = None
        self.noise_sample_fn = None
        self.noise_distr = None
        self.noise_scale = None
        self.noise_shift = None
        self.init_noise_sample_fn = None
        self.init_noise_distr = None
        self.init_noise_scale = None
        self.init_noise_shift = None
        self.noise_tuning = None
        self.generator = None
        self.generator_name = None
        self.reupload = None
        self.generator_fn = None
        self.discriminator_name = None
        self.discriminator_cls = None
        self.gen_init_sample_fn = None
        self.gen_init_distr = None
        self.gen_init_scale = None
        self.gen_init_shift = None
        self.num_params = None
        self.params_shape = None
        self.noise_shape = None
        self.init_noise_shape = None
        self.slower_progress_update = None
        self.normalizer = None
        self.measurement_scheme = None
        self.measurement_process = None
        self.decoding_scheme = None
        self.decoder = None
        self.gan_method = None
        self.n_channels = None
        self.shots = None

    def build(
        self,
        model_name: str,
        data_set: str,
        n_qubits: int = 2,
        n_ancilla_qubits: int = 0,
        circuit_depth: int = 1,
        noise_distr='normal',
        noise_scale=0.1,
        noise_shift=0.0,
        init_noise_distr=None,
        init_noise_scale=None,
        init_noise_shift=None,
        noise_tuning=None,
        random_seed: int = 42,
        transformation: str = "pit",
        measurement_scheme: str = "n_pauli_z",
        decoding_scheme: str = "sign_to_unit",
        save_artifacts=True,
        slower_progress_update=False,
        discriminator_name: str = "continuous_fully_connected",
        generator_name: str = "layered_rot_strongly_ent",
        reupload: int = 0,
        gen_init_distr='normal',
        gen_init_scale=2*np.pi,
        gen_init_shift=0.0,
        gan_method='GAN',
        n_channels=1,
        shots=None,
    ) -> BaseModelHandler:
        """Build the continuous qgan model.
        This defines the architecture of the model, including the circuit ansatz, data transformation and whether the artifacts are saved.

        Args:
            model_name (int): The name which will be used to save the data to disk.
            data_set: The name of the data set which gets as part of the model name
            n_qubits (int, optional): Number of qubits. Defaults to 2.
            n_ancilla_qubits (int, optional): Number of ancillary qubits. Defaults to 0.
            circuit_depth (int, optional): Number of repetitions of qml.StronglyEntanglingLayers. Defaults to 1.
            noise_distr (str, optional): Type of noise distribution to use. Defaults to "normal".
            noise_scale (float, optional): Noise distribution parameter. Defaults to 0.1.
            noise_shift (float, optional): Noise distribution parameter. Defaults to 0.0.
            init_noise_distr (str, optional): Type of noise distribution to use initially in generator circuit if desired. Defaults to None.
            init_noise_scale (float, optional): Noise distribution parameter for initial noise in generator circuit if desired. Defaults to None.
            init_noise_shift (float, optional): Noise distribution parameter for initial noise in generator circuit if desired. Defaults to None.
            noise_tuning (str, optional): Determines the learnable parameterization of the noise embedding in the generator. Possible strategies are "scale", "shift" and "both". Defaults to None.
            random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
            transformation (str, optional): Type of normalization, either "minmax" or "pit". Defaults to "pit".
            measurement_scheme (str, optional): Type of measurements to be performed at the end of the generator quantum circuit. Defaults to "n_pauli_z", which complements the "sign_to_unit" decoding scheme.
            decoding_scheme (str, optional): Type of decoding to turn generator measurement outcomes into final samples. Defaults to "sign_to_unit", which complements the "n_pauli_z" measurement scheme.
            save_artifacts (bool | str | iterable, optional): Whether to save the artifacts to disk or detailed specification of artifacts. Defaults to True.
            slower_progress_update (bool, optional): Controls how often the progress bar is updated. If set to True, update every 10 seconds at most, otherwise use tqdm defaults. Defaults to False.
            discriminator_name (str, optional): Name of the discriminator model. Defaults to the fully connected NN.
            generator_name (str, optional): Name of the discriminator model. Defaults to layered angle and strongly entangled QNN.
            reupload (int, optional): Number of reuploading repetitions in the generator circuit. Defaults to 0, i.e., no reuploading.
            gen_init_distr (str, optional): Type of noise distribution to use for initial generator weights. Defaults to "normal".
            gen_init_scale (float, optional): Noise distribution parameter for initial generator weights. Defaults to 2pi.
            gen_init_shift (float, optional): Noise distribution parameter for initial generator weights. Defaults to 0.0.
            gan_method (str, optional): Type of GAN training method to use, either "GAN" or "WGAN-GP". Defaults to "GAN".
            n_channels (int, optional): Number of channels per pixel. Defaults to 1. (grayscale images)
            shots (int, optional): Number of shots per image decoding used. Defaults to None. (exact)
        Returns:
            BaseModelHandler: Return the built model handler. It is not strictly necessary to overwrite the existing variable with this
            since all changes are made in place.
        """
        self.slower_progress_update = slower_progress_update
        self.save_artifacts = save_artifacts
        self.random_key = jax.random.PRNGKey(random_seed)
        self.n_qubits = n_qubits
        self.n_ancilla_qubits = n_ancilla_qubits
        self.circuit_depth = circuit_depth
        self.noise_distr = noise_distr
        self.noise_scale = noise_scale
        self.noise_shift = noise_shift
        self.init_noise_distr = init_noise_distr
        self.init_noise_scale = init_noise_scale
        self.init_noise_shift = init_noise_shift
        self.noise_tuning = noise_tuning
        self.transformation = transformation
        self.measurement_scheme = measurement_scheme
        self.decoding_scheme = decoding_scheme
        self.discriminator_name = discriminator_name
        self.generator_name = generator_name
        self.reupload = reupload
        self.gen_init_distr = gen_init_distr
        self.gen_init_scale = gen_init_scale
        self.gen_init_shift = gen_init_shift
        self.gan_method = gan_method
        self.n_channels = n_channels
        self.shots = shots
        time_str = str(time.time())
        pid_str = str(os.getpid())
        uniq = hashlib.md5((time_str + pid_str).encode("utf-8")).hexdigest()[:8]

        self.data_set = data_set

        circuit_type = "continuous"
        self.model_name = f"{model_name}_{self.data_set}_{self.transformation}_qgan_{uniq}"
        print(f"{pid_str=}: {self.model_name=}")
        self.path_to_models = "experiments/" + self.model_name

        self.metadata = dict(
            {
                "model_name": self.model_name,
                "data_set": self.data_set,
                "n_qubits": self.n_qubits,
                "n_ancilla_qubits": self.n_ancilla_qubits,
                "circuit_type": circuit_type,
                "circuit_depth": self.circuit_depth,
                "transformation": self.transformation,
                "measurement_scheme": self.measurement_scheme,
                "decoding_scheme": self.decoding_scheme,
                "discriminator": "digital",
                "training_data": {},   # information about the training not the training data set
                "discriminator_name": self.discriminator_name,
                "generator_name": self.generator_name,
                "reupload": self.reupload,
                "noise_distr": self.noise_distr,
                "noise_scale": self.noise_scale,
                "noise_shift": self.noise_shift,
                "init_noise_distr": self.init_noise_distr,
                "init_noise_scale": self.init_noise_scale,
                "init_noise_shift": self.init_noise_shift,
                "noise_tuning": self.noise_tuning,
                "gen_init_distr": self.gen_init_distr,
                "gen_init_scale": self.gen_init_scale,
                "gen_init_shift": self.gen_init_shift,
                "gan_method": gan_method,
                "n_channels": self.n_channels,
                "shots": self.shots,
            }
        )
        if save_artifacts:
            os.makedirs(self.path_to_models)
            with open(self.path_to_models + "/" + "meta.json", "w") as fp:
                json.dump(self.metadata, fp)
            with open(self.path_to_models + "/" + "meta.yaml", "w") as fp:
                yaml.dump(self.metadata, fp)
            if isinstance(save_artifacts, str) and len(save_artifacts) > 0:  # If non-empty string, store into list
                self.save_artifacts = [save_artifacts]
        if self.transformation == "minmax":
            self.normalizer = MinMaxNormalizer()
        elif self.transformation == "pit":
            self.normalizer = PITNormalizer()
        else:
            raise ValueError("Transformation value must be either 'minmax' or 'pit'")

        self.noise_sample_fn = noise_sample_fn_factory(key=self.noise_distr)
        if self.init_noise_distr is not None:
            self.init_noise_sample_fn = noise_sample_fn_factory(key=self.init_noise_distr)
        self.measurement_process = measurement_process_factory(self.measurement_scheme)
        self.decoder = decoder_factory(self.decoding_scheme, n_ancilla_qubits=self.n_ancilla_qubits, shots=self.shots)
        self._instantiate_generator()
        if self.discriminator_name is not None:
            self.discriminator_cls = discriminator_factory(self.discriminator_name)
        self.gen_init_sample_fn = noise_sample_fn_factory(key=self.gen_init_distr)

        return self

    def _instantiate_generator(self):
        if self.generator_name is not None:
            self.generator_fn = generator_factory(self.generator_name)
        if self.generator is None:
            generator_kwargs = dict(circuit_depth=self.circuit_depth,
                                    n_qubits=self.n_qubits, n_ancilla_qubits=self.n_ancilla_qubits,
                                    measurement_fn=self.measurement_process, noise_tuning=self.noise_tuning,
                                    reupload=self.reupload)
            try:  # check if generator takes ancilla qubits as argument
                g = self.generator_fn(**generator_kwargs)
            except TypeError:
                generator_kwargs.pop('n_ancilla_qubits')
                g = self.generator_fn(**generator_kwargs)
            print(f"{g=}")
            self.generator, self.noise_shape, self.init_noise_shape, self.params_shape = g
            self.num_params = np.prod(self.params_shape).item()

    def save(self, file_path: Path, overwrite: bool = True) -> BaseModelHandler:
        """Save the generator and discriminator weights to disk.

        Args:
            file_path (Path): The paths where the pickled tuple of generator and discriminator weights will be placed.
            overwrite (bool, optional): Whether to overwrite the file if it already exists. Defaults to True.

        Returns:
            BaseModelHandler: The model, unchanged.
        """
        if overwrite or not os.path.exists(file_path):
            with open(file_path, "wb") as file:
                pickle.dump((self.generator_weights, self.discriminator_weights), file)
        try:  # Store more artifacts if specified (otherwise, artifacts is only a boolean)
            extra_sample_artifacts = ('noise', 'measurement_outputs')
            for artifact_type in self.save_artifacts:
                match artifact_type:
                    case artifact_type if artifact_type.startswith("sample"):
                        # Retrieve iteration
                        filename = os.path.splitext(os.path.basename(file_path))[0]
                        iteration_str = filename[filename.find("iteration="):]

                        # Retrieve number of samples to generate
                        n_samples = 1 if '_' not in artifact_type else int(artifact_type.split('_')[-1])

                        # Sample and save
                        extra_returns = {f"return_{k}": (k in self.save_artifacts) for k in extra_sample_artifacts}
                        samples = self.sample(n_samples, **extra_returns)
                        if any(extra_returns.values()):
                            samples, samples_info = samples
                            for k, v in samples_info.items():
                                np.save(os.path.join(os.path.dirname(file_path), f"{k}_{iteration_str}.npy"),
                                        v, allow_pickle=False)

                        np.save(os.path.join(os.path.dirname(file_path), f"samples_{iteration_str}.npy"),
                                samples, allow_pickle=False)
                    case artifact_type if artifact_type in extra_sample_artifacts:
                        # These artifacts are saved along with the samples
                        if not any(a.startswith("sample") for a in self.save_artifacts):
                            raise ValueError(f"The artifact {artifact_type} requires the artifact 'samples'")
                    case _:
                        raise ValueError(f"Some artifact ({artifact_type}) is unknown and cannot be saved.")
        except TypeError as e:
            print(repr(e))
        return self

    @classmethod
    def scan_for_models(cls, metadata, skip_training_meta_data=False):
        """
        Scans the experiment directory for a model matching the passed metadata (most recent for multiple matches).
        """
        if isinstance(metadata, str):  # Metadata passed as a path
            with open(metadata, "r") as file:
                metadata = json.load(file)
        elif isinstance(metadata, dict):  # Metadata directly provided as dict
            pass  # ready for use
        else:
            raise TypeError("metadata must either be of type str (JSON path) or dict (loaded metadata).")

        other_metadata_paths = glob("experiments/*/meta.json")
        other_metadata_paths.sort(key=os.path.getmtime, reverse=True)  # sort them by lasts modified date
        diff_dicts = []
        model_name = metadata["model_name"]
        del metadata["model_name"]
        for o_metadata_path in other_metadata_paths:
            with open(o_metadata_path, "r") as file:
                o_metadata = json.load(file)
            o_model_name = o_metadata["model_name"]
            if skip_training_meta_data:
                o_metadata['training_data'] = {}
            # Skip if model_name is the same and otherwise drop the model name from the comparison
            if o_model_name == model_name:
                continue
            else:
                del o_metadata["model_name"]

            diff = {}
            for k in metadata.keys():
                if k in o_metadata.keys() and metadata[k] != o_metadata[k]:
                    diff[k] = (metadata[k], o_metadata[k])
            diff_dicts.append(diff)
            if metadata == o_metadata:
                return o_model_name
        diff_dicts.sort(key=len)
        for d in diff_dicts[:3]:
            print("Dictionary:")
            print(d)
        return None

    def warm_start(self, prev_depth_decrement=1):
        orig_model_name = self.model_name
        orig_circuit_depth = self.circuit_depth
        meta_data_warm_start = self.metadata.copy()
        try:
            meta_data_warm_start["circuit_depth"] -= prev_depth_decrement
        except TypeError:  # Tuple with sublayers expected now
            meta_data_warm_start["circuit_depth"] = (meta_data_warm_start["circuit_depth"][0] - prev_depth_decrement,
                                                     *meta_data_warm_start["circuit_depth"][1:])

        meta_data_warm_start_json = self.path_to_models + "/" + "meta_warm_start.json"
        with open(meta_data_warm_start_json, "w+") as file:
            json.dump(meta_data_warm_start, file)
        with open(self.path_to_models + "/" + "meta_warm_start.yaml", "w+") as file:
            yaml.dump(meta_data_warm_start, file)

        warm_start_model_name = self.scan_for_models(metadata=meta_data_warm_start_json, skip_training_meta_data=True)
        if warm_start_model_name is None:
            raise ValueError("No model found for the warm start with the lower depth circuit",
                             meta_data_warm_start["circuit_depth"])
        last_epoch = max(int(s.split("iteration=")[1].split(".pickle")[0]) for s
                         in glob("experiments/" + warm_start_model_name + "/parameters_training_iteration=*.pickle"))
        self.reload(model_name=warm_start_model_name, epoch=last_epoch)

        self.metadata['model_name'], self.model_name = orig_model_name, orig_model_name
        self.metadata['circuit_depth'], self.circuit_depth = orig_circuit_depth, orig_circuit_depth
        self.path_to_models = "experiments/" + self.metadata["model_name"]


        # Only initialize last layer
        self.random_key, subkey1 = jax.random.split(self.random_key)
        generator_weights_last_layer = (
            self.gen_init_sample_fn(key=subkey1, shape=(1, *self.params_shape[1:]),
                                    scale=self.gen_init_scale, shift=self.gen_init_shift)
        )
        # Concatenate the weights:
        self.generator_weights = np.concat([self.generator_weights, generator_weights_last_layer])

        print(f"{self.model_name=}")
        print(f"{self.metadata['circuit_depth']=}")
        print(f"{self.circuit_depth=}")

        #assert self.generator_weights.shape[0] == self.circuit_depth, f"{self.generator_weights.shape[0] }{ self.circuit_depth}"

    def reload(self, model_name: str, epoch: int) -> BaseModelHandler:
        """Reload the model from the artifacts including the parameters for the generator and the discriminator,
        the metadata and the data transformation file (reverse lookup table or original min and max of the training data).

        Args:
            model_name (str): The name of the model to reload.
            epoch (int): The epoch to reload.

        Returns:
            BaseModelHandler: The reloaded model, but changes have been made in place as well.
        """
        weights_file = "experiments/" + model_name + "/" + "parameters_training_iteration={0}.pickle".format(str(epoch))
        meta_file = "experiments/"+ model_name + "/" +  "meta.json"
        reverse_file = "experiments/" + model_name + "/" + 'reverse_lookup.npy'

        with open(weights_file, "rb") as file:
            self.generator_weights, self.discriminator_weights = pickle.load(file)
        with open(meta_file, "r") as file:
            self.metadata = json.load(file)

        assert model_name == self.metadata["model_name"], \
            (f"Name passed for model ({model_name}) reloading is inconsistent with name in metadata "
             f"({self.metadata['model_name']}) of the loaded model.")

        self.reverse_lookup = jnp.load(reverse_file)
        self.data_set = self.metadata["data_set"]
        self.model_name = model_name
        self.n_qubits = self.metadata["n_qubits"]
        self.n_ancilla_qubits = self.metadata["n_ancilla_qubits"]
        self.transformation = self.metadata["transformation"]
        self.measurement_scheme = self.metadata["measurement_scheme"]
        self.decoding_scheme = self.metadata["decoding_scheme"]
        self.generator_name = self.metadata["generator_name"]
        self.discriminator_name = self.metadata["discriminator_name"]
        self.circuit_depth = self.metadata["circuit_depth"]
        self.performed_trainings = len(self.metadata["training_data"])
        self.random_key = jax.random.PRNGKey(2)
        self.path_to_models =  "experiments/" + self.metadata["model_name"]
        self.noise_distr = self.metadata["noise_distr"]
        self.noise_scale = self.metadata["noise_scale"]
        self.noise_shift = self.metadata["noise_shift"]
        self.init_noise_distr = self.metadata["init_noise_distr"]
        self.init_noise_scale = self.metadata["init_noise_scale"]
        self.init_noise_shift = self.metadata["init_noise_shift"]
        self.noise_tuning = self.metadata["noise_tuning"]
        self.gen_init_distr = self.metadata["gen_init_distr"]
        self.gen_init_scale = self.metadata["gen_init_scale"]
        self.gen_init_shift = self.metadata["gen_init_shift"]
        self.gan_method = self.metadata["gan_method"]
        self.n_channels = self.metadata.get("n_channels", 1)
        self.reupload = self.metadata.get("reupload", 0)
        self.shots = self.metadata.get("shots", None)

        if self.normalizer is None:
            if self.transformation == "minmax":
                self.normalizer = MinMaxNormalizer()
            elif self.transformation == "pit":
                self.normalizer = PITNormalizer()
            else:
                raise ValueError("Transformation value must be either 'minmax' or 'pit'")
        self.normalizer.reverse_lookup = self.reverse_lookup

        self.noise_sample_fn = noise_sample_fn_factory(key=self.noise_distr)
        if self.init_noise_distr is not None:
            self.init_noise_sample_fn = noise_sample_fn_factory(key=self.init_noise_distr)
        self.gen_init_sample_fn = noise_sample_fn_factory(key=self.gen_init_distr)

        self.measurement_process = measurement_process_factory(self.measurement_scheme)
        self.decoder = decoder_factory(self.decoding_scheme, n_ancilla_qubits=self.n_ancilla_qubits, shots=self.shots)
        self._instantiate_generator()
        if self.discriminator_name is not None:
            self.discriminator_cls = discriminator_factory(self.discriminator_name)

        return self

    def train(
        self,
        train_dataset_original_space: np.array,
        n_epochs: int,
        initial_learning_rate_generator: float,
        initial_learning_rate_discriminator: float,
        adam_b1: float = 0.9,
        adam_b2: float = 0.999,
        batch_size=None,
        discriminator_training_steps=1
    ) -> BaseModelHandler:
        """Train the continuous QGAN.

        Args:
            train_dataset_original_space (np.array): The training data in the original space.
            n_epochs (int): Technically, we are not passing the number of passes through the training data, but the number of iterations of the training loop.
            initial_learning_rate_generator (float, optional): Learning rate for the quantum generator.
            initial_learning_rate_discriminator (float, optional): Learning rate for the classical discriminator.
            adam_b1 (float, optional): Adam optimizer parameter. (exponential decay rate to track the first moment of past gradients.) Defaults to 0.9.
            adam_b2 (float, optional): Adam optimizer parameter. (exponential decay rate to track the second moment of past gradients.) Defaults to 0.999.
            batch_size (int, optional): Batch size. Defaults to None, and the whole training data is used in each iteration.
            discriminator_training_steps (int, optional): Number of training steps for discriminator per generator training step. Defaults to 1.

        Raises:
            ValueError: Raises ValueError if the training dataset has dimension (number of columns) not equal to 2 or 3.

        Returns:
            BaseModelHandler: The trained model.
        """
        self.batch_size = len(train_dataset_original_space) if batch_size is None else batch_size
        
        if self.performed_trainings == 0:
            self.previous_trained_epochs = 0 
        else:
            self.previous_trained_epochs = sum([self.metadata["training_data"][str(i)]["n_epochs"] for i in range(self.performed_trainings)])

        training_data = {}
        training_data["n_epochs"] = n_epochs
        training_data["batch_size"] = self.batch_size
        training_data[
            "initial_learning_rate_generator"
        ] = initial_learning_rate_generator
        training_data[
            "initial_learning_rate_discriminator"
        ] = initial_learning_rate_discriminator
        training_data["adam_b1"] = adam_b1
        training_data["adam_b2"] = adam_b2
        training_data["discriminator_training_steps"] = discriminator_training_steps
        self.metadata["training_data"][str(self.performed_trainings)] = training_data
        self.performed_trainings += 1

        train_dataset = self.normalizer.fit_transform(train_dataset_original_space)
        self.reverse_lookup = self.normalizer.reverse_lookup

        if self.save_artifacts:
            with open(self.path_to_models + "/" + "meta.json", "w+") as file:
                json.dump(self.metadata, file)
            with open(self.path_to_models + "/" + "meta.yaml", "w+") as file:
                yaml.dump(self.metadata, file)

            jnp.save(self.path_to_models + "/" + "reverse_lookup.npy", self.reverse_lookup)

        X_train = CustomDataset(train_dataset.astype("float32"))
        print(f"{self.gan_method=}")
        is_critic = self.gan_method.startswith("WGAN")  # Wasserstein GAN setting uses a critic instead of discriminator
        D = self.discriminator_cls(is_critic=is_critic, n_channels=self.n_channels)
        epsilon = 1e-10
        if len(gan_method_split := self.gan_method.split("_")) > 1:
            gradient_penalty_coeff = float(gan_method_split[-1])
            D_grad_input = jax.grad(lambda *args, **kwargs: D.apply(*args, **kwargs).squeeze(), argnums=1)
            D_grad_input = jax.vmap(D_grad_input, in_axes=(None, 0))
        else:
            gradient_penalty_coeff, D_grad_input = None, None

        D.apply = jax.jit(D.apply)
        v_qnode = jax.vmap(self.generator, in_axes=(0, None))

        def cost_fn_discriminator(z, X, generator_weights, discriminator_weights, alpha=None):
            G_sample = self.decoder(self.standardize_pennylane_output(v_qnode(z, generator_weights)))
            D_fake = D.apply(discriminator_weights, G_sample)
            D_real = D.apply(discriminator_weights, X)
            if not is_critic:  # log transform if Vanilla GAN, not for Wasserstein GAN
                D_real = -jnp.log(D_real + epsilon)
                D_fake = -jnp.log(1.0 - D_fake + epsilon)
            else:
                D_real = -D_real # important since opposite relative signs are needed
            loss_1 = jnp.mean(D_real)
            loss_2 = jnp.mean(D_fake)
            D_loss = loss_1 + loss_2

            if alpha is not None:  # compute Wasserstein GAN gradient penalty given uniform noise samples
                alpha = alpha.reshape(-1, 1)  # two dims for broadcasting operations
                # gradient penalty code from original implementation of Gulrajani et al. 2017
                differences = G_sample - X
                interpolates = X + (alpha * differences)
                gradients = D_grad_input(discriminator_weights, interpolates)
                slopes = jnp.linalg.norm(gradients, axis=1)
                gradient_penalty = jnp.mean((slopes - 1.) ** 2)
                D_loss += gradient_penalty_coeff * gradient_penalty

            return D_loss


        def cost_fn_generator(z, generator_weights, discriminator_weights):
            G_sample = self.decoder(self.standardize_pennylane_output(v_qnode(z, generator_weights)))
            D_fake = D.apply(discriminator_weights, G_sample)
            if not is_critic:  # log transform if Vanilla GAN, not for Wasserstein GAN
                D_fake = jnp.log(D_fake + epsilon)
            G_loss = -jnp.mean(D_fake)
            return G_loss

        logged_metrics = defaultdict(list)
        it_list = []
        progress = tqdm(
            range(n_epochs), mininterval=10 if self.slower_progress_update else None
        )
        synthetic_transformed_space = 0  # init so its a global var

        self.random_key, subkey1, subkey2, subkey3 = jax.random.split(
            self.random_key, num=4
        )

        # Currently it is not possible that one type of weight is None while the other is not, but keep the
        # if-statements separate for now anyways in case we want to do that in the future.
        if self.generator_weights is None:
            self.generator_weights = (
                self.gen_init_sample_fn(key=subkey1, shape=self.params_shape,
                                        scale=self.gen_init_scale, shift=self.gen_init_shift)
            )
        if self.discriminator_weights is None:
            x = jax.random.uniform(subkey2, shape=X_train.data.shape[1:])  # Dummy input
            self.discriminator_weights = D.init(subkey3, x)

        optimizer_generator = optax.adam(initial_learning_rate_generator, b1=adam_b1, b2=adam_b2)
        optimizer_state_g = optimizer_generator.init(self.generator_weights)

        optimizer_discriminator = optax.adam(initial_learning_rate_discriminator, b1=adam_b1, b2=adam_b2)
        optimizer_state_d = optimizer_discriminator.init(self.discriminator_weights)

        for it in progress:
            if self.save_artifacts and (it % 100 == 0 or it == n_epochs - 1):
                self.save(
                    f"{self.path_to_models}/parameters_training_iteration={it + self.previous_trained_epochs }.pickle",
                    overwrite=False,
                 )


            for _ in range(discriminator_training_steps):

                # Get next training data batch:
                X = X_train.next_batch(self.batch_size)

                # Sample new noise:
                tmp_batch_size = round(self.batch_size / 2 ** self.n_ancilla_qubits)
                self.random_key, subkey = jax.random.split(self.random_key)
                z = self.noise_sample_fn(key=subkey, shape=(tmp_batch_size, *self.noise_shape),
                                         scale=self.noise_scale, shift=self.noise_shift)
                if self.init_noise_distr is not None:
                    self.random_key, subkey = jax.random.split(self.random_key)
                    z_init = self.init_noise_sample_fn(key=subkey, shape=(tmp_batch_size, *self.init_noise_shape),
                                                       scale=self.init_noise_scale, shift=self.init_noise_shift)
                    z = (z, z_init)

                if gradient_penalty_coeff is not None:  # Use critic gradient penalty
                    self.random_key, subkey = jax.random.split(self.random_key)
                    alpha = jax.random.uniform(subkey, shape=(self.batch_size,))
                else:
                    alpha = None

                if self.batch_size != len(train_dataset):
                    self.random_key, subkey = jax.random.split(self.random_key)

                cost_discriminator, grad = jax.value_and_grad(
                    lambda w: cost_fn_discriminator(z, X, self.generator_weights, w, alpha=alpha)
                )(self.discriminator_weights)

                updates, optimizer_state_d = optimizer_discriminator.update(
                    grad, optimizer_state_d
                )
                D_grad_mag = np.linalg.norm(np.asarray(jax.tree_util.tree_leaves(jax.tree.map(jnp.linalg.norm, grad)))).item()
                self.discriminator_weights = optax.apply_updates(
                    self.discriminator_weights, updates
                )

            cost_generator, grad = jax.value_and_grad(
                lambda w: cost_fn_generator(z, w, self.discriminator_weights)
            )(self.generator_weights)
            G_grad_mag = np.linalg.norm(np.array(grad)).mean()

            updates, optimizer_state_g = optimizer_generator.update(
                grad, optimizer_state_g
            )
            self.generator_weights = optax.apply_updates(
                self.generator_weights, updates
            )

            # Update progress bar postfix
            progress.set_postfix(
                loss_generator=cost_generator,
                loss_discriminator=cost_discriminator,
                grad_mag_generator=G_grad_mag,
                grad_mag_discriminator=D_grad_mag,
                major_layer=self.circuit_depth,
                name=self.model_name,
                refresh=False if self.slower_progress_update else None,
            )

            logged_metrics['loss_generator'].append(cost_generator)
            logged_metrics['loss_discriminator'].append(cost_discriminator)
            logged_metrics['grad_mag_generator'].append(G_grad_mag)
            logged_metrics['grad_mag_discriminator'].append(D_grad_mag)
            it_list.append(it)

            if it % 100 == 0 or it == n_epochs - 1:  # Store every 100 epochs and in the final epoch
                df_eval_summary = pd.DataFrame(
                    {"iteration": it_list} | {k: np.array(v).astype(float) for k, v in logged_metrics.items()}
                )
                df_eval_summary = df_eval_summary.sort_values(by=["iteration"]).reset_index(drop=True)
                path_to_eval_csv = f"{self.path_to_models}/evaluation_summary.csv"
                df_eval_summary.to_csv(path_to_eval_csv, mode='w', index=False, header=True)

        if self.save_artifacts:
            self.save(
                f"{self.path_to_models}/parameters_training_iteration={it + self.previous_trained_epochs + 1}.pickle",
                overwrite=False,
            )
        return self

    def predict(
        self,
        n_samples: int = 32,
        **kwargs
    ) -> np.array:
        """Generate samples from the trained model and perform the inverse of the data transformation
        which was used to transform the training data to be able to compute the KL-divergence in the original space.

        Args:
            n_samples (int, optional): Number of samples to generate. Defaults to 32.

        Returns:
            np.array: Array of samples of shape (n_samples, sample_dimension).
        """
        samples_transformed = self.predict_transform(n_samples, **kwargs)

        if self.transformation == "pit":
            self.transformer = PITNormalizer()
        elif self.transformation == "minmax":
            self.transformer = MinMaxNormalizer()
        self.transformer.reverse_lookup = self.reverse_lookup

        try:
            samples = self.transformer.inverse_transform(samples_transformed)
            return samples
        except TypeError:
            samples_transformed, return_info = samples_transformed
            samples = self.transformer.inverse_transform(samples_transformed)
            return samples, return_info

    def predict_transform(
        self,
        n_samples: int = 32,
        **kwargs
    ) -> np.array:
        """Generate samples from the trained model in the transformed space (the n-dimensional unit cube).

        Args:
            n_samples (int, optional): Number of samples to generate. Defaults to 32.

        Returns:
            np.array: Array of samples of shape (n_samples, sample_dimension).
        """
        if self.performed_trainings == 0:
            raise ValueError(
                "Please train the model before trying to generate samples."
            )
        self.random_key, subkey = jax.random.split(self.random_key)
        noise = self.noise_sample_fn(key=subkey, shape=(n_samples, *self.noise_shape),
                                     scale=self.noise_scale, shift=self.noise_shift)
        if self.init_noise_distr is not None:
            self.random_key, subkey = jax.random.split(self.random_key)
            z_init = self.init_noise_sample_fn(key=subkey, shape=(n_samples, *self.init_noise_shape),
                                               scale=self.init_noise_scale, shift=self.init_noise_shift)
            noise = (noise, z_init)

        v_qnode = jax.vmap(lambda inpt: self.generator(inpt, self.generator_weights))
        samples_transformed = self.decoder(measurement_outputs := self.standardize_pennylane_output(v_qnode(noise)))
        samples_transformed = np.asarray(samples_transformed)

        return_info = dict.fromkeys([k[k.find('_') + 1:] for k, v in kwargs.items() if k.startswith("return") and v])
        if len(return_info) > 0:
            for k in set(return_info.keys()):
                match k:
                    case 'noise':
                        if self.init_noise_distr is None:
                            return_info[k] = np.asarray(noise)
                        else:
                            return_info['noise'] = np.asarray(noise[0])
                            return_info['init_noise'] = np.asarray(noise[1])
                    case 'measurement_outputs':
                        return_info[k] = np.asarray(measurement_outputs)
                    case _:
                        raise KeyError(f"Unrecognized key {k} to return additional information from sampling.")

            return samples_transformed, return_info
        else:
            return samples_transformed

    def sample(self, n_samples: int = 32, **kwargs):
        """Generate samples from the trained model.

        Args:
            n_samples (int, optional): Number of samples to generate. Defaults to 32.

        Returns:
            np.array: Array of samples of shape (n_samples, sample_dimension).
        """
        return self.predict(n_samples, **kwargs)

    def standardize_pennylane_output(self, G_sample):
        """ Adapt to new QNode return values with newer Pennylane Versions
            https://docs.pennylane.ai/en/stable/introduction/returns.html
        """
        # Does not only depend on the Pennylane version but also on the measurement type
        # (e.g., Probabilities vs Expectation Values)
        if version.parse(pennylane_version) > version.parse("0.32") and isinstance(G_sample, list):
            res_list = []
            for qubit_output in G_sample:
                res_list.append(qubit_output.reshape(-1, 1))
            G_sample_np = jnp.hstack(res_list)
            return G_sample_np
        else:
            return G_sample
