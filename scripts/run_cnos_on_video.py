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

# CNOS
from modules.cnos import CNOS
from modules.cnos import visualize as CNOS_Visualize
from modules import CHECKPOINTS_ROOT

# python -m scripts.run_cnos_on_video --input local_data/video_test/input.mp4 --object local_data/video_test/objects/Object/Object.ply --output local_data/video_test/output_cnos.mp4


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--object", type=str)
    parser.add_argument("--output", type=str, default="output.mp4")

    args = parser.parse_args()

    video_path = Path(args.input)
    object_path = Path(args.object)
    output_path = Path(args.output)

    print(f"Input video: {video_path}")
    print(f"Object Path: {object_path}")
    print(f"Output video: {output_path}")

    cap = cv2.VideoCapture(str(video_path))

    # check if video opened successfully
    if not cap.isOpened():
        print(f"Error opening video file '{video_path}'.")
        exit()

    # get video size
    im_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    im_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_stream = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        30,
        (im_width * 2, im_height),
        True
    )

    pipe = CNOS(
        checkpoint=CHECKPOINTS_ROOT / 'fastsam' / 'FastSAM-x.pt'
    )

    pipe.register_object(
        mesh_path=object_path,
    )

    frame_cnt = 0
    print('Start processing video')
    while True:
        # read frame
        ret, frame = cap.read()
        if not ret:
            print("All frames processed.")
            break
        frame_cnt += 1

        res = pipe.detect(frame)
        score = res[0]['score']

        print(f"Frame {frame_cnt}/{total_frames}, ACC: {score}")

        frame = CNOS_Visualize(frame, res)
        
        out_stream.write(frame)
    
    print('release resources')
    cap.release()
    out_stream.release()