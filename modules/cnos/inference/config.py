from typing import Tuple
from cnos.model.loss import PairwiseSimilarity

class AttrDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def load_config(
    template_level: int
) -> Tuple[AttrDict, AttrDict, AttrDict]:
    # onboarding config
    onboarding_config = {
        'rendering_type': 'pbr',
        'reset_descriptors': False,
        'level_templates': template_level # 0 is coarse, 1 is medium, 2 is dense
    }
    
    # matching config
    matching_config = {
        'metric': PairwiseSimilarity(metric="cosine", chunk_size=16),
        'aggregation_function': 'avg_5',
        'confidence_thresh': 0.01
    }
    # post processing
    post_processing_config = {
        'mask_post_processing': {
            'min_box_size': 0.05,
            'min_mask_size': 3e-4,
        },
        'nms_thresh': 0.25
    }

    return AttrDict(onboarding_config), AttrDict(matching_config), AttrDict(post_processing_config), 