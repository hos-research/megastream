# Standard Library
import argparse
import json
import os
from pathlib import Path
import time
from typing import List, Tuple, Union, Optional

# Third Party
import numpy as np
import torch
import pandas as pd

# MegaPose
from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
from megapose.datasets.scene_dataset import CameraData, ObjectData
from megapose.inference.types import (
    DetectionsType,
    ObservationTensor,
    PoseEstimatesType,
)
from megapose.utils.load_model import NAMED_MODELS, load_named_model
from megapose.inference.pose_estimator import PoseEstimator
from megapose.inference.utils import make_detections_from_object_data, add_instance_id, make_TCO_from_object_data
from megapose.lib3d.transform import Transform
from megapose.utils.timer import SimpleTimer
from megapose.utils.logging import get_logger

logger = get_logger(__name__)

class MegaPose:
    """Universal inference wrapper for megapose6d."""

    pose_estimator: PoseEstimator
    object_dataset: RigidObjectDataset
    camera_data: CameraData
    n_refiner_iterations: int
    n_pose_hypotheses: int

    def __init__(
        self,
        object_path: Path,
        intrinsic: Union[str, Tuple[int, int]],
        label: Optional[str] = None,
        model_type: Optional[str] = 'megapose-1.0-RGB-multi-hypothesis',
        mesh_units: Optional[str] = 'm'
    ) -> None:
        """Initialize MegaPose instance.
        """
        # set objects
        self.set_object(mesh_path=object_path, label=label, mesh_units=mesh_units)

        # set camera
        if isinstance(intrinsic, str):
            self.set_camera_from_json(Path(intrinsic))
        elif isinstance(intrinsic, tuple):
            assert len(intrinsic) == 2, "intrinsic should be tuple of (width, height)"
            self.set_camera_from_imsize(intrinsic[0], intrinsic[1])
        else:
            raise ValueError("intrinsic should be either str or tuple")

        # init model
        self.load_model(model_name=model_type)
    
    def set_object(
        self, 
        mesh_path: Path, 
        label: str = None,
        mesh_units: str = "m"
    ) -> None:
        """Set novel objects from meshs in the given directory.
        Return the current instance of MegaPose.
        """

        object_mesh = []
        # object_dirs = mesh_dir.iterdir()
        object_id = 0

        # for object_dir in object_dirs:
        #     label = object_dir.name
        #     mesh_path = None
        #     for fn in object_dir.glob("*"):
        #         if fn.suffix in {".obj", ".ply"}:
        #             assert not mesh_path, f"there multiple meshes in the {label} directory"
        #             mesh_path = fn
        #     assert mesh_path, f"couldnt find a obj or ply mesh for {label}"
        #     object_mesh.append(RigidObject(label=label, mesh_path=mesh_path.resolve(), mesh_units=mesh_units))
        if label is None:
            label = str(mesh_path.stem)
        self.label = label
        object_mesh.append(RigidObject(label=label, mesh_path=mesh_path.resolve(), mesh_units=mesh_units))

        self.object_dataset = RigidObjectDataset(object_mesh)
    
    def set_camera_from_json(
        self, 
        filepath: Path
    ) -> None:
        """Set camera parameters from json file.
        Return the current instance of MegaPose.
        """
        logger.info('Setting camera intrinsic from json file.')
        self.camera_data = CameraData.from_json(filepath.read_text())
    
    def set_camera_from_imsize(
        self,
        width: int,
        height: int
    ) -> None:
        """Set coarse camera data from resolution.
        Return the current instance of MegaPose.
        """
        logger.info('Setting coarse camera intrinsic from image size.')
        fxy = float(width if width > height else height)
        self.camera_data = CameraData(
            K=np.array([[fxy, 0.0, width / 2.0],
               [0.0, fxy, height / 2.0],
               [0.0, 0.0, 1.0]]
            ),
            resolution=(height, width),
        )
    
    def load_model(
        self, 
        model_name: str = "megapose-1.0-RGB-multi-hypothesis"
    ) -> None:
        """Init megapose model through input model name.
        Return the current instance of MegaPose.
        """

        logger.info(f"Loading model {model_name}.")
        assert self.object_dataset is not None, "No object set for inference."
        self.pose_estimator = load_named_model(model_name, self.object_dataset).cuda()
        inference_params = NAMED_MODELS[model_name]["inference_parameters"]
        self.n_pose_hypotheses = inference_params["n_pose_hypotheses"]
        self.n_refiner_iterations = inference_params["n_refiner_iterations"]

    def convert_observation(
        self, 
        frame: np.ndarray,
        depth: Optional[np.ndarray] = None
    ) -> ObservationTensor:
        """Load formatted data from image frame
        Return the observation tensor.
        """

        rgb = frame
        K = self.camera_data.K

        assert rgb.shape[:2] == self.camera_data.resolution
        return ObservationTensor.from_numpy(rgb=rgb, depth=depth, K=K)

    def convert_detection(
        self, 
        labeled_detections: List[dict]
    ) -> DetectionsType:
        """Load formatted data from object data
        Format of detections should be:
        {
            label: str,
            box: List[float], # (4, ) array [xmin, ymin, xmax, ymax]
        }

        Return the detection tensor.
        """
        object_data = []
        for ld in labeled_detections:
            d = ObjectData(ld["label"])
            d.bbox_modal = ld["bbox"]
            object_data.append(d)

        detections = make_detections_from_object_data(object_data).cuda()
        return detections
    
    def convert_pose(
        self,
        labeled_poses: List[dict]
    ) -> PoseEstimatesType:
        """Load PoseEstimatesType from pose array
        """
        object_data = []
        for lp in labeled_poses:
            d = ObjectData(lp["label"])
            d.TCO = lp["TCO"]
            object_data.append(d)
        return make_TCO_from_object_data(object_data)

    @torch.no_grad()
    def inference_coarse(
        self, 
        frame: np.ndarray,
        labeled_detections: List[dict],
    ) -> Tuple[PoseEstimatesType, dict]:
        """Runs the coarse pose estimator on the given observation and detections.

        Returns:
            data_TCO_filtered: coarse predictions
            extra_data: dict containing extra data about coarse predictions
        """

        observation = self.convert_observation(frame).cuda()
        detections = self.convert_detection(labeled_detections).cuda()

        timer = SimpleTimer()
        timer.start()

        model: PoseEstimator = self.pose_estimator

        # Ensure that detections has the instance_id column
        assert detections is not None, "No detections provided."
        detections = add_instance_id(detections)

        # Run the coarse estimator using gt_detections
        data_TCO_coarse, coarse_extra_data = model.forward_coarse_model(
            observation=observation,
            detections=detections,
        )

        # Extract top-K coarse hypotheses
        data_TCO_filtered = model.filter_pose_estimates(
            data_TCO_coarse, top_K=self.n_pose_hypotheses, filter_field="coarse_logit"
        )

        timer.stop()

        extra_data: dict = dict()
        extra_data["score"] = data_TCO_filtered.infos["coarse_score"].to_numpy().tolist()
        
        extra_data["time"] = {
            "render" : coarse_extra_data["render_time"],
            "model" : coarse_extra_data["model_time"],
            "total" : timer.elapsed()
        }

        return data_TCO_filtered, extra_data

    @torch.no_grad()
    def inference_refine(
        self, 
        frame: np.ndarray,
        data_TCO_filtered: PoseEstimatesType,
        depth: Optional[np.ndarray] = None
    ) -> Tuple[PoseEstimatesType, dict]:
        """Runs the pose refiner on the given observation and coarse estimates or last refined predictions

        Returns:
            data_TCO_final: refined predictions
            extra_data: dict containing extra data about refined predictions
        """
        
        observation = self.convert_observation(frame, depth).cuda()

        timer = SimpleTimer()
        timer.start()

        model: PoseEstimator = self.pose_estimator

        # Run the refiner
        preds, refiner_extra_data = model.forward_refiner(
            observation,
            data_TCO_filtered,
            n_iterations=self.n_refiner_iterations,
        )
        data_TCO_refined = preds[f"iteration={self.n_refiner_iterations}"]

        # Score the refined poses using the coarse model.
        data_TCO_scored, scoring_extra_data = model.forward_scoring_model(
            observation,
            data_TCO_refined,
        )

        # Extract the highest scoring pose estimate for each instance_id
        data_TCO_final = model.filter_pose_estimates(
            data_TCO_scored, top_K=1, filter_field="pose_logit"
        )

        # if depth is used, run ICP refiner
        depth_refine_time = 0.0
        if depth is not None:
            start_time = time.time()
            data_TCO_final, _ = model.run_depth_refiner(
                observation,
                data_TCO_final,
            )
            depth_refine_time = time.time() - start_time

        timer.stop()

        extra_data: dict = dict()
        extra_data["score"] = scoring_extra_data["scores"].cpu().numpy()[0].tolist()
        extra_data["time"] = {
            "refiner": refiner_extra_data["time"],
            "scoring": scoring_extra_data["time"],
            "total": timer.elapsed(),
        }

        return data_TCO_final, extra_data

    def score(
        self,
        frame: np.ndarray,
        data_TCO: PoseEstimatesType
    ) -> float:
        """Score current pose.

        Returns:
            score: score of the pose
        """
        observation = self.convert_observation(frame).cuda()
        model: PoseEstimator = self.pose_estimator
        # Score current pose using the coarse model.
        data_TCO = data_TCO.cuda()
        data_TCO_scored, scoring_extra_data = model.forward_scoring_model(
            observation=observation,
            data_TCO=data_TCO
        )
        scores = scoring_extra_data["scores"].cpu().numpy()[0].tolist()
        return scores[0]


    
    def estimate(
        self,
        frame: np.ndarray,
        coarse_pose: PoseEstimatesType = None,
        detections: List[dict] = None,
        depth: Optional[np.ndarray] = None
    ) -> Tuple[PoseEstimatesType, dict]:
        """Estimate 6DoF Pose

        Returns:
            data_TCO: refined final predictions
            extra_data: dict containing extra data
        """
        # assert input
        assert coarse_pose is not None or detections is not None, "No detections or coarse pose provided."
        data_TCO : PoseEstimatesType = (None if coarse_pose is None else coarse_pose)
        extra : dict = {}
        # run coarse estimate
        if detections is not None:
            data_TCO, extra_coarse = self.inference_coarse(frame, detections)
            extra["coarse"] = extra_coarse
        # run refine
        data_TCO, extra_refine = self.inference_refine(frame, data_TCO, depth)
        extra["refine"] = extra_refine

        return data_TCO, extra

    def release(self) -> None:
        self.pose_estimator.release()
    
    @staticmethod
    def convert_TCO_dict(
        data_TCO: PoseEstimatesType
    ) -> List[dict]:
        """Convert PoseEstimatesType to list of dict.
        """
        if data_TCO is None: 
            return None
        data_TCO_dict = {}
        for i in range(len(data_TCO)):
            label = data_TCO.infos["label"][i]
            pose = data_TCO.poses[i].cpu().numpy()
            data_TCO_dict[label] = pose
        return data_TCO_dict
    


