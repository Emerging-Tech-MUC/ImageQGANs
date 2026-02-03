import cv2
import numpy as np
from fire import Fire


def load_mnist(path, kind='train'):
    """
    COPIED FROM https://github.com/zalandoresearch/fashion-mnist/blob/master/utils/mnist_reader.py
    """
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = f'{path}_{kind}_labels_raw.gz'
    images_path = f'{path}_{kind}_raw.gz'

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


def main(n_data_samples=-1,
         digit=tuple(range(10)),
         img_size=28,
         flatten_imgs=True,
         normalize_imgs=False):
    # Load data:
    # Download raw data (train-images-idx3-ubyte.gz and train-labels-idx1-ubyte.gz) here: https://github.com/zalandoresearch/fashion-mnist?tab=readme-ov-file
    # Save them as fashion_mnist_train_raw.gz and fashion_mnist_train_labels_raw.gz, respectively

    mnist, mnist_labels = load_mnist('fashion_mnist', kind='train')
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
    save_path = f"fashion_mnist_{digits_str}_{img_size}x{img_size}_N_{n_data_samples}"
    np.save(save_path, imgs_scaled)


if __name__ == '__main__':
    Fire(main)
