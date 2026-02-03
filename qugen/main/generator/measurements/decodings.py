from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from .image_encodings import FRQI_decoding, FRQI_RGBa_decoding, amplitude_decoding


def identity_decoding(measurement_outputs, *_, **__):
    return measurement_outputs


def sign_to_unit_decoding(measurement_outputs, *_, **__):
    return (measurement_outputs + 1) / 2


def image_decoding_wrapper(image_decoding_fn: Callable, n_color_qubits: int = 0, shots=None, is_probs: bool = None) -> Callable:
    """
    Wrap the image decoding schemes, which infers certain meta information from the measurement outputs automatically.
    This ensures an interface of solely taking the raw measurement outputs of the generator quantum circuit.
    This interface is expected by the qugen library.

    Args:
        image_decoding_fn (Callable): The image decoding scheme that takes more than the raw measurements as the input.
        n_color_qubits (int): The number of color qubits the decoding scheme uses. If negative, color qubits are assumed to be at the beginning of the circuit (most significant bits), otherwise at the end (least significant bits).
        is_probs (bool, optional): Determines whether the measurement outputs are probabilities or amplitudes.
        None by default to auto-detect depending on the raw measurement output data type.

    Returns (Callable):
        The wrapped image decoding function, which matches the interface expected by qugen
    """

    def wrapped_image_decoding_fn(measurement_outputs: np.ndarray, **kwargs) -> np.ndarray:
        assert measurement_outputs.ndim in (2, 3), \
            (f"Invalid measurement outputs dimension {measurement_outputs.ndim}. "
             f"The measurement outputs must have a dimension of either two or three for flattened images or 2D images, "
             f"respectively. The first dimension determines the batch.")

        n_ancilla_qubits = kwargs.pop("n_ancilla_qubits", 0)
        if n_ancilla_qubits > 0:
            if measurement_outputs.ndim == 3:
                raise NotImplementedError("Ancilla qubits are not supported for non-flattened image measurements")
            n_pixel_data = measurement_outputs.shape[1] // 2 ** n_ancilla_qubits
            measurement_outputs = measurement_outputs.reshape(-1, n_pixel_data, 2 ** n_ancilla_qubits) # separate
            measurement_outputs = measurement_outputs.transpose((0, 2, 1)) # innermost axis holds measures per image
            measurement_outputs = measurement_outputs.reshape(-1, n_pixel_data)  # flatten batch
            # re-normalize (implicitly by marginal probs of ancillas)
            measurement_outputs = measurement_outputs / measurement_outputs.sum(axis=1, keepdims=True)


        batch_size = measurement_outputs.shape[0]
        _is_probs = is_probs

        try:  # If image shape is specified, validate with shape of measurement outputs
            img_shape = kwargs["shape"]
            assert measurement_outputs.size >= batch_size * np.prod(img_shape), \
                (f"Incompatible specified image shape {img_shape} for shape of provided measurement outputs: "
                 f"{measurement_outputs.shape}")
        except KeyError:
            if measurement_outputs.ndim == 2:  # Flattened image
                n_pixels = measurement_outputs.shape[1] // (2 ** abs(n_color_qubits))
                img_size = np.around(np.sqrt(n_pixels)).astype(int)
                assert img_size ** 2 == n_pixels, \
                    (f"Flattened images are assumed to represent square images. "
                     f"However, {n_pixels} pixels cannot constitute a square image (its square root is not integral)")
                img_shape = (img_size, img_size)
            else:  # (measurement_outputs.ndim == 3) -> 2D image
                img_shape = measurement_outputs.shape[1:]
            kwargs["shape"] = img_shape

        # image_decoding_fn expects a flattened image
        measurement_outputs = measurement_outputs.reshape(batch_size, -1)

        # swap order if necessary:
        if n_color_qubits < 0:
            measurement_outputs = measurement_outputs.reshape(batch_size, 2, -1).transpose(0, 2, 1).flatten()

        if _is_probs is None:
            _is_probs = np.isrealobj(measurement_outputs)
        if _is_probs:
            measurement_outputs = jnp.sqrt(measurement_outputs)
        else:
            measurement_outputs = jnp.abs(measurement_outputs).real

        base_key = kwargs.pop("rng_key", jax.random.PRNGKey(0))

        local_shots = kwargs.pop("shots", shots)
        if local_shots is not None:
            # ➋ Split it once into `batch_size` independent keys
            keys = jax.random.split(base_key, batch_size)

            # ➌ Rewrite helper so key is an argument
            def _add_noise(m_out, key):
                p = m_out ** 2
                sample = jax.random.multinomial(key, n=local_shots, p=p) / local_shots
                sample = sample / jnp.sum(sample)
                noise  = jax.lax.stop_gradient(p - sample)
                p = jnp.clip(p - noise, a_min=0) + 1e-10
                p = p / jnp.sum(p)
                return jnp.sqrt(p).flatten()

            # ➍ vmap over *both* measurement_outputs and keys
            measurement_outputs = jax.vmap(_add_noise, in_axes=(0, 0))(
                measurement_outputs.reshape(batch_size, -1),
                keys,
            )

        kwargs["indexing"] = kwargs.get("indexing", 'hierarchical')
        return image_decoding_fn(measurement_outputs, **kwargs).reshape(batch_size, -1)

    return wrapped_image_decoding_fn


# ### Factory methods to obtain a sample decoder via key: ###

decoding_lookup = {
    "identity": identity_decoding,
    "sign_to_unit": sign_to_unit_decoding,
    "amplitude": image_decoding_wrapper(amplitude_decoding),
    "FRQI": image_decoding_wrapper(FRQI_decoding, n_color_qubits=1),
    "FRQI_RGBa": image_decoding_wrapper(FRQI_RGBa_decoding, n_color_qubits=-3),  # IMPORTANT: FRQI_RGBa_decoding uses MSB color qubit ordering!
    "FRQI_msb": image_decoding_wrapper(FRQI_decoding, n_color_qubits=-1),
    "FRQI_RGBa_msb": image_decoding_wrapper(FRQI_RGBa_decoding, n_color_qubits=3)
}


def decoder_factory(key: str, **kwargs) -> Callable:
    """
    Factory function retrieves the decoding function, identified by the given key.
    The decoder decodes the measurement outcomes of the generator quantum circuit into the final generated sample.

    Args:
        key (str): The identifier for the decoding scheme.

    Returns (Callable):
        The decoder implementing the decoding scheme.
    """
    try:
        decoder = decoding_lookup[key]
    except KeyError:
        raise ValueError(f"Decoder ({key}) is not recognized. Valid identifiers are {list(decoding_lookup.keys())}")

    if len(kwargs) > 0:
        decoder = partial(decoder, **kwargs)

    return decoder