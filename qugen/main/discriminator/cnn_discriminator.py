import flax.linen as nn_jax
import numpy as np


class ConvDiscriminator(nn_jax.Module):

    is_critic: bool = False
    n_channels: int = 1
    hidden_layer_dis: int = 64

    @nn_jax.compact
    def __call__(self, x):

        # transfer the input to proper shape (channel dim and 2D image):
        img_size = round(np.sqrt(x.shape[-1] // self.n_channels))
        x = x.reshape(-1, img_size, img_size, self.n_channels)

        # zero-center input (assumes [0, 1] data interval)
        x = nn_jax.Conv(features=self.hidden_layer_dis, kernel_size=(5, 5), strides=(2, 2))(x)
        x = nn_jax.leaky_relu(x)

        x = nn_jax.Conv(features=2 * self.hidden_layer_dis, kernel_size=(5, 5), strides=(2, 2))(x)
        x = nn_jax.leaky_relu(x)

        x = nn_jax.Conv(features=4 * self.hidden_layer_dis, kernel_size=(5, 5), strides=(2, 2))(x)
        x = nn_jax.leaky_relu(x)

        x = x.reshape((x.shape[0], -1))  # Flatten before into fully connected layer

        x = nn_jax.Dense(features=1)(x)
        if not self.is_critic:  # Only apply sigmoid activation if discriminator does not act as a critic
            x = nn_jax.sigmoid(x)
        return x



class ConvDiscriminator3x3(nn_jax.Module):

    is_critic: bool = False
    hidden_layer_dis: int = 64

    @nn_jax.compact
    def __call__(self, x):

        # transfer the input to proper shape (channel dim and 2D image):
        img_size = round(np.sqrt(x.shape[-1]))
        x = x.reshape(-1, img_size, img_size, 1)

        # zero-center input (assumes [0, 1] data interval)
        x = nn_jax.Conv(features=self.hidden_layer_dis, kernel_size=3, strides=1)(x)
        x = nn_jax.leaky_relu(x)

        x = nn_jax.Conv(features=2 * self.hidden_layer_dis, kernel_size=3, strides=1)(x)
        x = nn_jax.leaky_relu(x)

        x = nn_jax.Conv(features=4 * self.hidden_layer_dis, kernel_size=3, strides=1)(x)
        x = nn_jax.leaky_relu(x)

        x = x.reshape((x.shape[0], -1))  # Flatten before into fully connected layer

        x = nn_jax.Dense(features=1)(x)
        if not self.is_critic:  # Only apply sigmoid activation if discriminator does not act as a critic
            x = nn_jax.sigmoid(x)
        return x

