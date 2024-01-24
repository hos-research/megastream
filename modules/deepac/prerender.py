# Standard Library
import os
import os.path as osp
from pathlib import Path
from typing import Optional

# Third Party
import torch
import imageio
import numpy as np
from tqdm import tqdm
import cv2
import pickle

# DeepAC
from .utils.geometry.body import Body
from .utils.geometry.render_geometry import GenerateGeodesicPoses, RenderGeometry
from .utils.geometry.viewer import Viewer
from .utils.geometry.wrappers import Pose, Camera
from .utils.draw_tutorial import draw_vertices_to_obj


def render_template_view(
    mesh_path: Path,
    sphere_radius: Optional[float] = 0.8,
    maximum_body_diameter: Optional[float] = 0.3,
    geometry_unit_in_meter: Optional[float] = 1.0,
    image_size: Optional[int] = 2000,
    image_border_size: Optional[int] = 20,
    n_divide: Optional[int] = 4,
    num_sample_contour_point: Optional[int] = 200,
    k_min_contour_length: Optional[int] = 15,
    k_contour_normal_approx_radius: Optional[int] = 3,
    normalize_to_origin: Optional[bool] = False,
):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    image_size = torch.tensor([image_size], dtype=torch.int).expand(1).to(device)
    image_border_size = torch.tensor([image_border_size], dtype=torch.int).expand(1).to(device)

    body = Body('body_0', [str(mesh_path)], geometry_unit_in_meter, maximum_body_diameter,
                normalize_to_origin=normalize_to_origin, device=device)
    ggp = GenerateGeodesicPoses(body.maximum_body_diameter, sphere_radius, image_size,
                                image_border_size, n_divide, device=device)
    view2world_matrix = ggp.view2world_matrix.transpose(0, 1)
    view2world_pose = Pose.from_4x4mat(view2world_matrix[0])
    viewer = Viewer((image_size[0].cpu().item(), image_size[0].cpu().item()),
                    view2world_pose, ggp.virtual_camera, device=device)

    render_geometry = RenderGeometry("render eigen", device=device)
    render_geometry.add_body(body)
    render_geometry.add_viewer(body.name, viewer)
    render_geometry.setup_render_context()

    print('start preprocess: ', mesh_path)
    template_views = 0
    orientations = 0
    for i in tqdm(range(0, view2world_matrix.shape[0])):
        view2world_pose = Pose.from_4x4mat(view2world_matrix[i])
        render_geometry.update_viewer_pose(body.name, view2world_pose)
        depths = render_geometry.render_depth()
        depth = depths[body.name].cpu().numpy()

        orientation = (body.world2body_pose @ view2world_pose).R[:, :3, 2].unsqueeze(1).cpu().numpy()
        # orientation = view2world_pose.R[:, :3, 2].unsqueeze(1).cpu().numpy()

        ret, centers_in_body, normals_in_body, foreground_distance, background_distance = \
            render_geometry.generate_point_data(body.name, depths, k_min_contour_length, num_sample_contour_point,
                                                k_contour_normal_approx_radius)

        if not ret:
            import ipdb;
            ipdb.set_trace()

        template_view = np.concatenate((centers_in_body, normals_in_body, np.expand_dims(foreground_distance, axis=-1),
                                        np.expand_dims(background_distance, axis=-1)), axis=-1)

        if i == 0:
            template_views = template_view
            orientations = orientation
        else:
            template_views = np.concatenate((template_views, template_view), axis=1)
            orientations = np.concatenate((orientations, orientation), axis=1)

    obj_path = mesh_path
    obj_name = mesh_path.stem
    output_path = mesh_path.parent / f'{obj_name}.pkl'

    j = 0
    template_view = template_views[j]
    orientation = orientations[j]
    fx = ggp.virtual_camera.f[j, 0].cpu().item()
    fy = ggp.virtual_camera.f[j, 1].cpu().item()
    cx = ggp.virtual_camera.c[j, 0].cpu().item()
    cy = ggp.virtual_camera.c[j, 1].cpu().item()
    head = {'obj_path': str(obj_path),
            'image_size': (image_size[0].cpu().item(), image_size[0].cpu().item()),
            'num_sample_contour_point': num_sample_contour_point,
            'body_normalize_to_origin': normalize_to_origin,
            'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}
    dictionary_data = {'head': head, 'template_view': template_view, 'orientation_in_body': orientation}

    with open(str(output_path), "wb") as pkl_handle:
        pickle.dump(dictionary_data, pkl_handle)

    print('finish preprocess: ', mesh_path)