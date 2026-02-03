from functools import partial
from typing import Callable

from .helper import kl_divergence_from_data, kl_div_kde
from .inception_metrics import inception_score, fid, mi_fid
from .mmd import mmd_linear, mmd_poly, mmd_rbf

# ### Factory methods to obtain a metric via key: ###

metrics_lookup = {
    "kl_div_histogram": kl_divergence_from_data,
    "mmd_linear": mmd_linear,
    "mmd_poly": mmd_poly,
    "mmd_rbf": mmd_rbf,
    "kl_div": kl_div_kde,
    "kl_div_scott": partial(kl_div_kde, bandwidth='scott'),
    "kl_div_silverman": partial(kl_div_kde, bandwidth='silverman'),
    "is": inception_score,
    "is_2048": partial(inception_score, feature=2048),
    "is_768": partial(inception_score, feature=768),
    "is_192": partial(inception_score, feature=192),
    "is_64": partial(inception_score, feature=64),
    "fid_2048": partial(fid, feature=2048),
    "fid_768": partial(fid, feature=768),
    "fid_192": partial(fid, feature=192),
    "fid_64": partial(fid, feature=64),
    "mifid_2048": partial(mi_fid, feature=2048),
    "mifid_768": partial(mi_fid, feature=768),
    "mifid_192": partial(mi_fid, feature=192),
    "mifid_64": partial(mi_fid, feature=64),
}


def metrics_factory(key: str, *args, **kwargs) -> Callable:
    try:
        return partial(metrics_lookup[key], *args, **kwargs)
    except KeyError:
        raise ValueError(f"Metric ({key}) is not recognized. "
                         f"Valid identifiers are {list(metrics_lookup.keys())}")
