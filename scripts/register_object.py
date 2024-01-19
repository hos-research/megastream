# Standard Library
import argparse
from pathlib import Path
import logging

from modules.cnos.poses.pyrender import render_object_pose

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh_path", type=str)
    args = parser.parse_args()
    mesh_path = Path(args.mesh_path)

    logging.info(f'Render template view for obejct {mesh_path}')
    
    render_object_pose(mesh_path)