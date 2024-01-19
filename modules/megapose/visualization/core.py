# Standard Library
import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Union

# Third Party
import numpy as np
import cv2
import torch

# MegaPose
from megapose.inference.core import MegaPose
from megapose.datasets.scene_dataset import ObjectData
# Visualize
from megapose.lib3d.transform import Transform
from megapose.panda3d_renderer import Panda3dLightData
from megapose.panda3d_renderer.panda3d_scene_renderer import Panda3dSceneRenderer, CameraRenderingData
from megapose.utils.conversion import convert_scene_observation_to_panda3d
from megapose.utils.logging import get_logger, set_logging_level
from megapose.visualization.utils import make_contour_overlay, get_mask_from_rgb

class Visualizer:
    """Visualize result of pose estimation."""
    estimator: MegaPose
    renderer: Panda3dSceneRenderer

    def __init__(
        self, 
        estimator: MegaPose
    ) -> None:
        self.estimator = estimator
        camera_data, object_dataset = estimator.camera_data, estimator.object_dataset
        
        # setup renderer
        self.renderer = Panda3dSceneRenderer(object_dataset)
        self.light_datas = [
            Panda3dLightData(light_type="ambient", color=((1.0, 1.0, 1.0, 1))),
        ]
    
    def render(
        self, 
        TCO_dict: dict
    ) -> CameraRenderingData:
        """Render scene from input object data
        """
        if TCO_dict is None:
            return None
        # set camera data and object data
        camera_data = self.estimator.camera_data
        camera_data.TWC = Transform(np.eye(4))
        object_datas = []
        for label in TCO_dict:
            object_data = ObjectData(label=label, TWO=Transform(TCO_dict[label]))
            object_datas.append(object_data)
        
        # convert to scene
        camera_data, object_datas = convert_scene_observation_to_panda3d(camera_data, object_datas)
        
        # render scene
        renderings = self.renderer.render_scene(
            object_datas,
            [camera_data],
            self.light_datas,
        )[0]

        return renderings

    
    def contour_overlay(
        self, 
        frame: np.ndarray,
        renderings: CameraRenderingData
    ) -> np.ndarray:
        if renderings is None:
            return frame
        contour_overlay = make_contour_overlay(
            frame, renderings.rgb, dilate_iterations=1, color=(0, 255, 0)
        )["img"]
        return contour_overlay

    def mesh_overlay(
        self, 
        frame: np.ndarray,
        renderings: CameraRenderingData,
        alpha: float = 1
    ) -> np.ndarray:
        mesh_overlay = np.zeros_like(frame).astype(np.float32)
        if renderings is None:
            return frame

        mask = get_mask_from_rgb(renderings.rgb)
        # overlay
        mesh_overlay[~mask] = frame[~mask] * alpha + 255 * (1 - alpha)
        # bgr 2 rgb
        cv2.cvtColor(mesh_overlay, cv2.COLOR_BGR2RGB, mesh_overlay)
        mesh_overlay[mask] = renderings.rgb[mask]
        cv2.cvtColor(mesh_overlay, cv2.COLOR_RGB2BGR, mesh_overlay)
        mesh_overlay = mesh_overlay.astype(np.uint8)
        return mesh_overlay

    
    def contour_mesh_overlay(
        self, 
        frame: np.ndarray,
        renderings: CameraRenderingData,
        alpha: float = 1
    ) -> np.ndarray:
        fig_contour_overlay = self.contour_overlay(frame, renderings)
        fig_mesh_overlay = self.mesh_overlay(frame, renderings, alpha)
        # concat
        fig_all = np.concatenate((fig_contour_overlay, fig_mesh_overlay), axis=1)
        return fig_all