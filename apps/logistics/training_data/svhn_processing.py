import cv2
import numpy as np
import scipy.io
from fire import Fire


def load_svhn(kind='train'):
    """
    See http://ufldl.stanford.edu/housenumbers/
    Information on data layout
    """

    match kind:
        case 'train' | 'extra':
            data_path = f"svhn_{kind}_32x32.mat"
        case _:
            raise ValueError(f"Unknown SVHN data kind: {kind}")

    data_dict = scipy.io.loadmat(data_path)
    x = data_dict['X']
    y = data_dict['y']

    images = x.transpose(3, 0, 1, 2)
    labels = y % 10  # maps the 0 class from label 10 to label 0

    return images, labels


def main(n_data_samples=-1,
         digit=tuple(range(10)),
         img_size=32,
         flatten_imgs=True,
         normalize_imgs=False,
         kind='extra',
         color='RGB'):

    # Load data:
    # Download raw data (extra_32x32.mat) here: http://ufldl.stanford.edu/housenumbers/
    # Save the file as svhn_extra_32x32.mat

    svhn, svhn_labels = load_svhn(kind=kind)
    assert svhn.shape[0] == svhn_labels.size
    n_pixels = round(np.prod(svhn.shape[1:3]))
    svhn_size = round(np.sqrt(n_pixels))
    channels = 3
    assert svhn_size ** 2 == n_pixels

    # Re-shape data:
    svhn = svhn.reshape(-1, svhn_size, svhn_size, channels)
    svhn_labels = svhn_labels.flatten()
    print(svhn.shape)
    print(svhn_labels.shape)

    # Filter digits:
    if isinstance(digit, int):  # single digit, otherwise, iterable with multiple desired digits
        digit = [digit]
    digit = np.asarray(digit)
    svhn = svhn[np.vectorize(lambda x: x in digit)(svhn_labels)]
    print(f"{len(svhn)} samples with digits: {digit.squeeze()}")

    # Keep only specified number of samples:
    if n_data_samples is None or n_data_samples < 0:
        n_data_samples = len(svhn)
    svhn = svhn[:n_data_samples]
    print(len(svhn))

    # Downscale images to desired size (with proper interpolation handled by opencv):
    if img_size != svhn_size:
        imgs_scaled = np.empty((n_data_samples, img_size, img_size))
        for i, img in enumerate(svhn):
            imgs_scaled[i] = cv2.resize(img, dsize=(img_size, img_size), interpolation=cv2.INTER_LANCZOS4)
    else:
        imgs_scaled = svhn

    # Convert color if desired
    if color.upper() != "RGB":
        match color.upper():
            case "HSV":
                for i, img in enumerate(imgs_scaled):
                    imgs_scaled[i] = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
                # convert from degree (0 to 180) to 0 to 255
                imgs_scaled[:,:,:,0] = np.clip(np.around(imgs_scaled[:,:,:,0] / 180 * 255).astype(imgs_scaled.dtype),
                                               0, 255)
            case _:
                raise NotImplementedError(f"Color scheme {color} is not supported.")

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
    color_str = '' if color.upper() == 'RGB' else "_" + color.upper()
    save_path = f"svhn{color_str}_{digits_str}_{img_size}x{img_size}_N_{n_data_samples}"
    print("Save as", save_path)
    np.save(save_path, imgs_scaled)


if __name__ == '__main__':
    Fire(main)
