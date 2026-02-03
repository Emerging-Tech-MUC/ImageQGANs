import flax.linen as nn_jax

from qugen.main.discriminator.cnn_discriminator import ConvDiscriminator, ConvDiscriminator3x3
from qugen.main.discriminator.discriminator import Discriminator_JAX as FCDiscriminator
from qugen.main.discriminator.discriminator_for_continuous_qgan import Discriminator as ContinuousFCDiscriminator

# ### Factory methods to obtain a discriminator via key: ###

discriminator_lookup = {
    "fully_connected": FCDiscriminator,
    "continuous_fully_connected": ContinuousFCDiscriminator,
    "convolutional": ConvDiscriminator,
    "convolutional3x3": ConvDiscriminator3x3,
}


def discriminator_factory(key: str) -> nn_jax.Module:
    try:
        return discriminator_lookup[key]
    except KeyError:
        raise ValueError(f"Discriminator ({key}) is not recognized. "
                         f"Valid identifiers are {list(discriminator_lookup.keys())}")
