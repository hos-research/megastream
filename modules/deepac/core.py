# Standard Library
import os
from pathlib import Path
from typing import Optional, Tuple, Union

# Third Party
import numpy as np
import cv2
import torch
from omegaconf import OmegaConf
import pickle

# DeepAC
from .models import get_model
from .models.deep_ac import calculate_basic_line_data
from .utils.lightening_utils import load_model_weight
from .utils.geometry.wrappers import Pose, Camera
from .utils.utils import (
    project_correspondences_line,
    get_closest_template_view_index,
    get_closest_k_template_view_index,
    get_bbox_from_p2d
)
from .dataset.utils import (
    resize,
    numpy_image_to_torch,
    crop,
    zero_pad,
    get_imgaug_seq
)

class DeepAC:
    fore_learning_rate: float
    back_learning_rate: float
    model: None

    # init data
    reinit_: bool
    total_fore_hist: torch.Tensor
    total_back_hist: torch.Tensor

    def __init__(
        self,
        intrinsic: Union[np.ndarray, Tuple[int, int]],
        template_path: Path,
        config_path: Path,
        checkpoint_path: Path,
        fore_learning_rate: Optional[float] = 0.2,
        back_learning_rate: Optional[float] = 0.2,
    ) -> None:
        # init intrinsic
        self.load_intrinsic(intrinsic=intrinsic)
        # load obj
        self.load_obj(template_path=template_path)
        # load model
        self.load_model(config_path=config_path, checkpoint_path=checkpoint_path)
        # set learning rate
        self.fore_learning_rate = fore_learning_rate
        self.back_learning_rate = back_learning_rate
        # init flag
        self.reinit_ = True

    def load_intrinsic(
        self,
        intrinsic: Union[np.ndarray, Tuple[int, int]]
    ) -> None:
        assert isinstance(intrinsic, (np.ndarray, tuple)), "intrinsic should be either a numpy array or a tuple"
        if isinstance(intrinsic, np.ndarray):
            assert intrinsic.shape == (3, 3), "intrinsic should be a 3x3 matrix"
            self.intrinsic = intrinsic
        else:
            assert len(intrinsic) == 2, "intrinsic should be a tuple of (width, height)"
            f = max(intrinsic[0], intrinsic[1])
            self.intrinsic = np.array([
                [f, 0, intrinsic[0] / 2],
                [0, f, intrinsic[1] / 2],
                [0, 0, 1]
            ])

    def load_obj(
        self,
        template_path: Path,
    ) -> None:
        with open(str(template_path), "rb") as pkl_handle:
            data = pickle.load(pkl_handle)
        head = data['head']
        self.num_sample_contour_points = head['num_sample_contour_point']
        self.template_views = torch.from_numpy(data['template_view']).type(torch.float32)
        self.orientations = torch.from_numpy(data['orientation_in_body']).type(torch.float32)

    def load_model(
        self,
        config_path: Path,
        checkpoint_path: Path
    ) -> None:
        config = OmegaConf.load(config_path)
        data_conf = config.data
        model_conf = config.models

        # get data info
        self.data_conf = data_conf
        self.get_top_k_template_views = data_conf.get_top_k_template_views
        self.skip_template_view = data_conf.skip_template_view

        # load model info
        model = get_model('deep_ac')(model_conf)
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        load_model_weight(model=model, checkpoint=ckpt)
        model.cuda()
        model.eval()

        self.model = model
    

    def preprocess_image(
        self,
        img: np.ndarray,
        bbox2d: np.ndarray,
        camera: Camera
    ) -> Tuple[torch.Tensor, Camera]:
        # crop image
        bbox2d[2:] += self.data_conf.crop_border * 2
        img, camera, bbox = crop(img, bbox2d, camera=camera, return_bbox=True)
        # resize
        scales = (1, 1)
        new_size = self.data_conf.resize
        resize_by = self.data_conf.resize_by
        if isinstance(new_size, int):
            if resize_by == 'max':
                img, scales = resize(img, new_size, fn=max)
            elif (resize_by == 'min' or (resize_by == 'min_if' and min(*img.shape[:2]) < new_size)):
                img, scales = resize(img, new_size, fn=min)
        elif len(new_size) == 2:
            img, scales = resize(img, new_size, fn=min)
        if scales != (1, 1):
            camera = camera.scale(scales)
        
        # padding
        img, = zero_pad(self.data_conf.pad, img)
        img = img.astype(np.float32)
        return numpy_image_to_torch(img), camera

    @torch.no_grad()
    def pack_data(
        self,
        image: np.ndarray,
        init_pose: Pose
    ) -> dict:
        assert init_pose is not None, "pose is not initialized"
        # format data
        height, width = image.shape[:2]
        K = self.intrinsic
        camera = Camera(
            data=torch.Tensor([
                width, height, K[0][0], K[1][1], K[0][2], K[1][2]
            ])
        )

        # closest k template and orientation
        indices = get_closest_k_template_view_index(
            body2view_pose=init_pose,
            orientations_in_body=self.orientations,
            k=self.get_top_k_template_views * self.skip_template_view
        )
        closest_template_views = torch.stack([
            self.template_views[
                ind * self.num_sample_contour_points: (ind + 1) * self.num_sample_contour_points,
                :
            ]
            for ind in indices[::self.skip_template_view]
        ])
        closest_orientations_in_body = self.orientations[indices[::self.skip_template_view]]

        # project correspondences line
        data_lines = project_correspondences_line(
            template_view=closest_template_views[0],
            body2view_pose=init_pose,
            camera=camera
        )
        # bbox 2d
        bbox2d = get_bbox_from_p2d(data_lines['centers_in_image'])

        # preprocess image
        fit_image, fit_camera = self.preprocess_image(
            img=image,
            bbox2d=bbox2d.numpy().copy(),
            camera=camera
        )

        return {
            'image': fit_image[None],
            'camera': fit_camera[None],
            'body2view_pose': init_pose[None],
            'closest_template_views': closest_template_views[None],
            'closest_orientations_in_body': closest_orientations_in_body[None],
        }

    @torch.no_grad()
    def calculate_histogram(
        self,
        image_t: torch.Tensor,
        template_view,
        pose: Pose,
        camera: Camera
    ) -> None:
        return_value = calculate_basic_line_data(
            template_view=template_view,
            body2view_pose_data=pose._data,
            camera_data=camera._data,
            fscale=0, 
            min_continuous_distance=1
        )
        _, _, centers_in_image, centers_valid, normals_in_image, foreground_distance, background_distance, _ = return_value
        total_fore_hist, total_back_hist = \
            self.model.histogram.calculate_histogram(
                image_t,
                centers_in_image,
                centers_valid,
                normals_in_image,
                foreground_distance,
                background_distance, 
                True
            )
        # update value
        return total_fore_hist, total_back_hist

    @torch.no_grad()
    def forward_tracking(
        self,
        image: np.ndarray,
        init_pose: Pose
    ) -> Pose:
        # pack data
        data = self.pack_data(
            image=image, 
            init_pose=init_pose
        )
        # reinit to calculate histogram
        if self.reinit_:
            fore_h, back_h = self.calculate_histogram(
                template_view=data['closest_template_views'][:, 0],
                image_t=data['image'],
                camera=data['camera'],
                pose=init_pose
            )
            # update value
            self.total_fore_hist = fore_h
            self.total_back_hist = back_h
            self.reinit_ = False
        
        data['fore_hist'] = self.total_fore_hist
        data['back_hist'] = self.total_back_hist
        # to cuda
        for key in data:
            data[key] = data[key].cuda()

        # forward model
        pred = self.model._forward(data, visualize=False, tracking=True)
        # update pose
        new_pose = pred['opt_body2view_pose'][-1][0].cpu()

        # update histogram
        index = get_closest_template_view_index(new_pose, self.orientations)
        closet_template_view = self.template_views[
            index * self.num_sample_contour_points : (index + 1) * self.num_sample_contour_points, 
            :
        ]
        fore_h, fore_b = self.calculate_histogram(
            template_view=closet_template_view[None],
            image_t=data['image'].cpu(),
            pose=new_pose,
            camera=data['camera'].cpu()
        )
        self.total_fore_hist = (1 - self.fore_learning_rate) * self.total_fore_hist + self.fore_learning_rate * fore_h
        self.total_back_hist = (1 - self.back_learning_rate) * self.total_back_hist + self.back_learning_rate * fore_b

        return new_pose

    @torch.no_grad()
    def Track(
        self,
        image: np.ndarray,
        last_pose: Union[np.ndarray, Pose],
        reinit: Optional[bool] = False
    ) -> Pose:
        # convert Pose
        if isinstance(last_pose, np.ndarray):
            assert last_pose.shape == (4, 4), "pose should be a 4x4 matrix"
            last_pose = Pose.from_4x4mat(
                torch.from_numpy(last_pose).type(torch.float32)
            )
        # check reinit
        if reinit: self.reinit_ = True
        # forward tracking
        return self.forward_tracking(image, last_pose)

    @staticmethod
    def convert_Pose44_numpy(
        pose: Pose
    ) -> np.ndarray:
        R, T = pose.numpy()
        pose44 = np.identity(4)
        pose44[:3, :3] = R
        pose44[:3, 3] = T
        return pose44.astype(np.float32)
