# Standard Library
from pathlib import Path
import logging
import numpy as np
from typing import Union, Optional

# Third Party
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import argparse

# MegaStream
from src.megastream import MegaStream
from src.megastream import PROJECT_DIR

parser = argparse.ArgumentParser()
parser.add_argument("--object", type=str)
parser.add_argument("--log", action='store_true', default=False)
args = parser.parse_args()

# megastream pipe
@st.cache_resource()
def create_pipe() -> MegaStream:
    return MegaStream(
        image_size=(640, 480),
        mesh_path=args.object,
        log=args.log
    )

pipe: MegaStream = create_pipe()

def video_frame_callback(frame: np.ndarray):
    """
    Callback function to be used with webrtc_streamer
    """
    img = frame.to_ndarray(format="bgr24")
    pipe.Push(frame=img)
    pose, _ = pipe.Get()
    img = pipe.Render(frame=img, pose6d=pose)
    return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(key="sample", video_frame_callback=video_frame_callback)
