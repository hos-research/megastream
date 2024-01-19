# Standard Library
import os, sys
import numpy as np
import shutil
from tqdm import tqdm
import time
import json
from pathlib import Path
import glob
from typing import Any, Optional

# Third Party
import torch
from PIL import Image
from omegaconf import DictConfig, OmegaConf
import cv2
import torchvision.transforms as T

# CNOS
from cnos.model.detector import CNOS as CNOS_Detector
from cnos.utils.bbox_utils import CropResizePad, force_binary_mask
from cnos.model.utils import Detections, mask_to_rle
from cnos.model.loss import Similarity
# from cnos.poses.pyrender import render_object_pose
# Load Core Model
from cnos.inference.des_model import load_descriptor_model
from cnos.inference.seg_model import load_segmentor_model
from cnos.inference.config import load_config

# Visualize
from cnos.inference.visualize import visualize

# Logging
# from utils.logging import get_logger

# logger = get_logger(__name__)

class CNOS:
    metric: Similarity
    model: CNOS_Detector
    image_size: int
    features: torch.Tensor
    label: str

    def __init__(self,
        checkpoint: Path,
        image_size: int = 640,
        dinov2_type: Optional[str] = 'dinov2_vitl14',
        template_level: Optional[int] = 0,
        mesh_path: Optional[str] = None,
        template_dir: Optional[Path] = None,
        label: Optional[str] = 'Object',
        radius: Optional[int] = 0.4
    ) -> None:
        self.image_size = image_size
        self.metric = Similarity()
        # segment model
        segmentor_model = load_segmentor_model(
            checkpoint=checkpoint,
            segmentor_width_size=image_size
        )
        # descript model
        descriptor_model = load_descriptor_model(
            model_name=dinov2_type,
            descriptor_width_size=image_size
        )
        # config
        onboarding_config, matching_config, post_processing_config = load_config(
            template_level=template_level
        )

        self.model = CNOS_Detector(
            segmentor_model=segmentor_model,
            descriptor_model=descriptor_model,
            onboarding_config=onboarding_config,
            matching_config=matching_config,
            post_processing_config=post_processing_config
        )

        if mesh_path:
            self.register_object(
                mesh_path=mesh_path,
                template_dir=template_dir,
                label=label,
                radius=radius
            )

    def register_object(
        self,
        mesh_path: Path,
        template_dir: Optional[Path] = None,
        label: Optional[str] = 'Object',
        radius: Optional[int] = 0.4
    ) -> None:
        self.label = label
        if not template_dir:
            template_dir = mesh_path.parent.resolve() / 'template'
        # if not os.path.exists(template_dir):
        #     render_object_pose(
        #         mesh_path=mesh_path,
        #         output_dir=template_dir,
        #     )
        # load template
        assert os.path.exists(template_dir), f"Cannot find template view at {template_dir}"

        template_paths = glob.glob(f'{template_dir}/*.png')
        boxes, templates = [], []
        for path in template_paths:
            image = Image.open(path)
            boxes.append(image.getbbox())

            image = torch.from_numpy(np.array(image.convert("RGB")) / 255).float()
            templates.append(image)
        # process
        templates = torch.stack(templates).permute(0, 3, 1, 2)
        boxes = torch.tensor(np.array(boxes))
        
        processing_config = OmegaConf.create(
            {"image_size": 224,}
        )
        proposal_processor = CropResizePad(processing_config.image_size)
        templates = proposal_processor(images=templates, boxes=boxes).cuda()
        # get features
        self.features = self.model.descriptor_model.compute_features(
            templates, token_name="x_norm_clstoken"
        )
    
    def convert_detections_dict(
        self,
        detections: Detections,
    ) -> dict:
        results = []
        for idx in range(len(detections.boxes)):
            result = {
                'label': self.label,
                'score': detections.scores[idx].item(),
                'bbox': detections.boxes[idx].tolist(),
                'segment': mask_to_rle(
                    force_binary_mask(detections.masks[idx])
                ) 
            }
            results.append(result)
        return results

    def detect(
        self,
        image: np.ndarray, # RGB format is required
        conf_threshold: Optional[float] = 0.5
    ) -> dict:
        rgb = image
        ref_feats = self.features
        # inference
        detections = self.model.segmentor_model.generate_masks(np.array(rgb))
        detections = Detections(detections)
        decriptors = self.model.descriptor_model.forward(np.array(rgb), detections)
        # get scores per proposal
        scores = self.metric(decriptors[:, None, :], ref_feats[None, :, :])
        score_per_detection = torch.topk(scores, k=5, dim=-1)[0]
        score_per_detection = torch.mean(
            score_per_detection, dim=-1
        )
        # get top-k detections
        scores, index = torch.topk(score_per_detection, k=1, dim=-1)
        detections.filter(index)
        # keep only detections with score > conf_threshold
        # detections.filter(scores > conf_threshold)
        detections.add_attribute("scores", scores)
        detections.to_numpy()
        # convert
        detections = self.convert_detections_dict(detections)

        return detections

    @staticmethod
    def visualize(
        rgb: np.ndarray,
        detections: dict,
        save_path: Optional[Path] = None
    ) -> np.ndarray:
        return visualize(
            rgb=rgb,
            detections=detections,
            save_path=save_path
        )

