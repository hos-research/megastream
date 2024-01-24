# Standard Library
import argparse
from pathlib import Path
import logging

from modules.cnos.poses.pyrender import render_object_pose
from modules.deepac.prerender import render_template_view

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh_path", type=str)
    parser.add_argument("--simpled", type=str, default='')

    args = parser.parse_args()
    
    mesh_path = Path(args.mesh_path)
    simpled_mesh_path = mesh_path if args.simpled == '' else Path(args.simpled)

    logging.info(f'Render template view for obejct {mesh_path}')
    
    render_object_pose(mesh_path)
    render_template_view(simpled_mesh_path)