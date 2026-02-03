import cv2
import numpy as np
from fire import Fire


def main(n_data_samples=-1,
         digit=tuple(range(10)),
         img_size=28,
         flatten_imgs=True,
         normalize_imgs=False,
         n_channels=1):
    # Load data:
    # Download raw data (train_images.npy and train_labels.npy) here: https://github.com/sebastian-lapuschkin/lrp_toolbox/tree/master/data/MNIST
    # Save them as mnist_raw.npy and mnist_labels_raw.npy, respectively
    mnist = np.load("mnist_raw.npy")
    mnist_labels = np.load("mnist_labels_raw.npy")
    assert mnist.shape[0] == mnist_labels.size
    n_pixels = mnist.shape[1]
    mnist_size = round(np.sqrt(n_pixels))
    assert mnist_size ** 2 == n_pixels

    # Re-shape data:
    mnist = mnist.reshape(-1, mnist_size, mnist_size)
    mnist_labels = mnist_labels.flatten()
    print(mnist.shape)
    print(mnist_labels.shape)

    # Filter digits:
    if isinstance(digit, int):  # single digit, otherwise, iterable with multiple desired digits
        digit = [digit]
    digit = np.asarray(digit)
    mnist = mnist[np.vectorize(lambda x: x in digit)(mnist_labels)]
    print(f"{len(mnist)} samples with digits: {digit.squeeze()}")

    # Keep only specified number of samples:
    if n_data_samples is None or n_data_samples < 0:
        n_data_samples = len(mnist)
    mnist = mnist[:n_data_samples]
    print(len(mnist))

    # Downscale images to desired size (with proper interpolation handled by opencv):
    imgs_scaled = np.empty((n_data_samples, img_size, img_size))
    for i, img in enumerate(mnist):
        imgs_scaled[i] = cv2.resize(img, dsize=(img_size, img_size), interpolation=cv2.INTER_LANCZOS4)

    # Duplicate channels if requested:
    if n_channels > 1:
        imgs_scaled = imgs_scaled.reshape(n_data_samples, img_size, img_size, 1)
        imgs_scaled = np.repeat(imgs_scaled, repeats=n_channels, axis=3)

    print("Sample\n", imgs_scaled[0])
    print(imgs_scaled.shape)

    # Flatten images again before saving if desired:
    if flatten_imgs:
        imgs_scaled = imgs_scaled.reshape(n_data_samples, -1)
        print("Flattened shape:", imgs_scaled.shape)

    # Normalize pixel values before saving if desired:
    if normalize_imgs:
        if not flatten_imgs:
            raise NotImplementedError("Image normalization is currently only supported for flattened images")
        imgs_scaled = (imgs_scaled.T / np.sum(imgs_scaled, axis=1) ).T

    # Store final images:
    digits_str = '_'.join(map(str, sorted(digit)))
    save_path = (("" if n_channels == 1 else f"{n_channels}ch_") +
                 f"mnist_{digits_str}_{img_size}x{img_size}_N_{n_data_samples}")
    np.save(save_path, imgs_scaled)


if __name__ == '__main__':
    Fire(main)
