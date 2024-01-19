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

def add_text_on_image(
    img: np.ndarray,
    text: str,
    color
) -> np.ndarray:
    cv2.putText(
        img,
        text,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--object", type=str)
    parser.add_argument("--depth", type=str, default='')
    parser.add_argument("--output", type=str, default="output.mp4")
    parser.add_argument("--no-sync", action='store_true', default=False)

    args = parser.parse_args()

    video_path = Path(args.input)
    object_path = Path(args.object)
    output_path = Path(args.output)
    depth_path = None if args.depth == '' else Path(args.depth)
    sync = not args.no_sync

    print(f'Sync mode: {sync}')
    print(f"Input video: {video_path}")
    print(f"Object Path: {object_path}")
    print(f"Output video: {output_path}")
    print(f"Depth video: {depth_path}")
    print()

    cap = cv2.VideoCapture(str(video_path))
    cap_d = None
    if depth_path is not None:
        cap_d = cv2.VideoCapture(str(depth_path))

    # check if video opened successfully
    if not cap.isOpened():
        print(f"Error opening video file '{video_path}'.")
        exit()

    # get video size
    im_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    im_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if depth_path is not None:
        total_cnt_d = int(cap_d.get(cv2.CAP_PROP_FRAME_COUNT))
        assert total_cnt == total_cnt_d, 'RGB and Depth frames count mismatch'

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
        sync=sync,
        use_depth=depth_path is not None,
        log=True
    )

    print('Start processing video')
    first_frame = True
    frame_cnt = 0

    progress = tqdm(total=total_cnt, position=0)

    while True:
        frame = None
        frame_d = None
        ###### read frame data ######
        ret, frame = cap.read()
        if depth_path is not None:
            ret, frame_d = cap_d.read()
            if ret and frame_d.shape[2] == 3:
                frame_d = cv2.cvtColor(frame_d, cv2.COLOR_BGR2GRAY)
        if not ret:
            break

        ###### core functions ######
        pipe.Push(frame=frame, depth=frame_d)
        progress.update(1)
        pose, score = pipe.Get()

        ###### render frame ######
        frame = pipe.Render(frame=frame, pose6d=pose)
        add_text_on_image(frame, f'Acc:{score:.2f}', (0, 255, 0) if score >= 0 else (0, 0, 255))
        out_stream.write(frame)

        time.sleep(8 if first_frame else 1 / 20)
        first_frame = False

    print('\nProcess finished.')
    cap.release()
    out_stream.release()
    pipe.Release()