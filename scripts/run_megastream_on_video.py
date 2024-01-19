# Standard Library
import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Union
import time
import logging

# Third Party
import numpy as np
import cv2
import torch
from tqdm import tqdm

# MegaStream
from src.megastream import MegaStream

# python -m scripts.run_megastream_on_video --input local_data/video_test/input.mp4 --object local_data/video_test/objects/Object/Object.ply --output local_data/video_test/output_megastream.mp4


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--object", type=str)
    parser.add_argument("--output", type=str, default="output.mp4")
    parser.add_argument("--realtime", action='store_true', default=False)

    args = parser.parse_args()

    video_path = Path(args.input)
    object_path = Path(args.object)
    output_path = Path(args.output)
    real_time = args.realtime

    print(f'Sync mode: {not real_time}')

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
    total_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_stream = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        20,
        (im_width * 2, im_height),
        True
    )

    pipe = MegaStream(
        image_size=(im_width, im_height),
        mesh_path=object_path,
        auto_download=True,
        sync=not real_time,
        log=True
    )

    print('Start processing video')
    first_frame = True
    frame_cnt = 0

    progress = tqdm(total=total_cnt, position=0)
    while True:
        # read frame
        ret, frame = cap.read()
        if not ret:
            break
        pipe.Push(frame=frame)
        progress.update(1)

        pose, score = pipe.Get()
        
        frame = pipe.Render(frame=frame, pose6d=pose)
        out_stream.write(frame)

        time.sleep(3 if first_frame else 1 / 20)
        first_frame = False

    print('Process finished.')
    cap.release()
    out_stream.release()
    pipe.Release()