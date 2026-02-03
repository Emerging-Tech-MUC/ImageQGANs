from functools import partial

import numpy as np
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.mifid import MemorizationInformedFrechetInceptionDistance

_inception_metric_model_class_lookup = {'IS': InceptionScore,
                                       'FID': FrechetInceptionDistance,
                                       'MIFID': MemorizationInformedFrechetInceptionDistance}
_inception_metric_models = {k: {} for k in _inception_metric_model_class_lookup.keys()}  # cache for models


# Template function to compute any inception metrics
def _template_inception_score(real_data, fake_data, inception_metric, feature=None):

    if feature is None:
        feature = 'logits_unbiased' if inception_metric == 'IS' else 2048

    try:  # check if model has been instantiated
        inception_metric_model = _inception_metric_models[inception_metric][feature]
    except KeyError:
        inception_metric_cls = _inception_metric_model_class_lookup[inception_metric]
        inception_metric_model = inception_metric_cls(feature=feature, normalize=True)
        _inception_metric_models[inception_metric][feature] = inception_metric_model

    # convert batch of gray-scale flat images to batch of color 2D images: (n_batch, H*W) -> (n_batch, 3, H, W):
    img_size = round(np.sqrt(real_data.shape[-1]))  # assumes square images
    real_data = real_data.reshape(-1, img_size, img_size)  # 2D gray-scale images (n_batch, H, W)
    fake_data = fake_data.reshape(-1, img_size, img_size)  # 2D gray-scale images (n_batch, H, W)
    real_data = np.repeat(real_data[:, np.newaxis, ...], 3, axis=1)  # copies gray-scale to 3 channels
    fake_data = np.repeat(fake_data[:, np.newaxis, ...], 3, axis=1)  # copies gray-scale to 3 channels

    # convert to torch tensors:
    real_data = torch.from_numpy(np.array(real_data))
    fake_data = torch.from_numpy(np.array(fake_data))

    # Compute score:
    if inception_metric == 'IS':
        inception_metric_model.update(real_data)
    else:
        inception_metric_model.update(real_data, real=True)
        inception_metric_model.update(fake_data, real=False)
    score = inception_metric_model.compute()
    inception_metric_model.reset()

    # if tuple return mean only (first element):
    if isinstance(score, tuple):
        score = score[0].item()
    if isinstance(score, torch.Tensor) and score.ndim == 0:
        score = score.item()

    return score

# Concrete inception metric functions
inception_score = partial(_template_inception_score, inception_metric='IS')
fid = partial(_template_inception_score, inception_metric='FID')
mi_fid = partial(_template_inception_score, inception_metric='MIFID')
