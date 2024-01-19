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
from modules.megapose import MegaPose, Visualizer

# Logging
from modules.utils.logging import get_logger, set_logging_level

logger = get_logger(__name__)

# python -m scripts.run_megapose_on_video --input local_data/video_test/input.mp4 --object local_data/image_test/Object/Object.ply --detection local_data/video_test/detection.json  --output local_data/video_test/output_megapose.mp4

if __name__ == "__main__":
    set_logging_level("info")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--objects", type=str)
    parser.add_argument("--detection", type=str)
    parser.add_argument("--output", type=str, default="output.mp4")

    args = parser.parse_args()

    video_path = Path(args.input)
    detect_path = Path(args.detection)
    object_dir = Path(args.objects)
    output_path = Path(args.output)

    print(f"Input video: {video_path}")
    print(f"Detection data: {detect_path}")
    print(f"Object data: {object_dir}")
    print(f"Output video: {output_path}")

    cap = cv2.VideoCapture(str(video_path))

    # check if video opened successfully
    if not cap.isOpened():
        print(f"Error opening video file '{video_path}'.")
        exit()

    # get video size
    im_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    im_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_stream = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        30,
        (im_width * 2, im_height),
        True
    )

    # load detection data
    detect_data = json.loads(detect_path.read_text())
    detect_box = detect_data[0]["box"]
    detect_max_width = detect_box[2] - detect_box[0]
    detect_max_height = detect_box[3] - detect_box[1]

    # load model
    megapose6d = MegaPose(
        object_dir,
        intrinsic=(im_width, im_height)
    )

    # set visualizer
    visualizer = Visualizer(megapose6d)

    # detection flag
    first_frame = True
    data_TCO = None
    extra = None

    def print_result(extra):
        if "coarse" in extra:
            score = extra["coarse"]["score"][0]
            time = extra["coarse"]["time"]["total"]
            print(f'coarse acc: {score:.2f}, time: {time:.3f}')
        score = extra["refine"]["score"][0]
        time = extra["refine"]["time"]["total"]
        print(f'refine acc: {score:.2f}, time: {time:.3f}')

    # process frame
    while True:
        # read frame
        ret, frame = cap.read()
        if not ret:
            print("All frames processed.")
            break
        
        # run detection on first frame
        if first_frame:
            data_TCO, extra = megapose6d.estimate(frame, detections=detect_data)
            first_frame = False
        else:
            data_TCO, extra = megapose6d.estimate(frame, coarse_pose=data_TCO)
        print_result(extra)

        # check score and decide whether to run coarse
        score = extra["refine"]["score"][0]
        # if score < 0.9:
        #     print("low score, running refine with coarse estimation")
        #     data_TCO_dict = MegaPose.convert_TCO_dict(data_TCO)
        #     label = list(data_TCO_dict.keys())[0]
        #     TCO = data_TCO_dict[label]
        #     # camera coordinate to pixel coordinate
        #     K = megapose6d.camera_data.K
        #     homogeneous = K @ TCO[:3, 3]
        #     u, v, _ = homogeneous / homogeneous[2]
        #     # move detection
        #     bbox = [0,0,0,0]
        #     bbox[0] = max(0, int(u - detect_max_width / 2)) # width
        #     bbox[2] = min(im_width - 1 ,int(u + detect_max_width / 2))
        #     bbox[1] = max(0, int(v - detect_max_height / 2)) # height
        #     bbox[3] = min(im_height - 1, int(v + detect_max_height / 2))
        #     # set flag
        #     detect_data[0]["box"] = bbox
        #     # run with coarse estimation
        #     # data_TCO, extra = megapose6d.estimate(frame, detections=detect_data)
        #     # print_result(extra)
        #     first_frame = True

        # visualize output
        data_TCO_dict = MegaPose.convert_TCO_dict(data_TCO)
        renderings = visualizer.render(data_TCO_dict)
        fig = visualizer.contour_mesh_overlay(frame, renderings)

        # write output
        out_stream.write(fig)

    print('release resources')
    cap.release()
    out_stream.release()
    
    # TODO: Core dump
    # if not released, the render processing is always running 
    megapose6d.release()


