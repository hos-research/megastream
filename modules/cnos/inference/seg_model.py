# Standard Library
import numpy as np
import time
from pathlib import Path
from typing import Any, Optional

# Third Party
import torch

# CNOS
from cnos.model.fast_sam import FastSAM
from cnos.inference.config import AttrDict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_segmentor_model(
    checkpoint: Path,
    segmentor_width_size: int,
    iou_threshold: Optional[float] = 0.9,
    conf_threshold: Optional[float] = 0.05,
    max_det: Optional[float] = 200,
    verbose: Optional[bool] = False
) -> FastSAM:
    sam_model = FastSAM(
        checkpoint_path=checkpoint,
        segmentor_width_size=segmentor_width_size,
        config=AttrDict({
            'iou_threshold': iou_threshold,
            'conf_threshold': conf_threshold,
            'max_det': max_det,
        })
    )
    sam_model.model.setup_model(device=device, verbose=verbose)
    return sam_model