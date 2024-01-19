# Standard Library
import os, sys
import numpy as np
import time
from pathlib import Path
from typing import Any, Optional

# Third Party
import torch
from PIL import Image
import cv2

# Visualize
from cnos.utils.amg import rle_to_mask
from skimage.feature import canny
from skimage.morphology import binary_dilation
import distinctipy

colors = distinctipy.get_colors(1)

def visualize(
    rgb: np.ndarray, 
    detections: dict, 
    save_path: Optional[Path] = None
):
    rgb = np.uint8(rgb)
    img = rgb.copy()
    img2 = rgb.copy()
    # gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    # img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


    alpha = 0.33

    for mask_idx, det in enumerate(detections):
        mask = rle_to_mask(det["segment"])
        # edge = canny(mask)
        # edge = binary_dilation(edge, np.ones((2, 2)))

        r = int(255*colors[0][0])
        g = int(255*colors[0][1])
        b = int(255*colors[0][2])
        img[mask, 0] = alpha*r + (1 - alpha)*img[mask, 0]
        img[mask, 1] = alpha*g + (1 - alpha)*img[mask, 1]
        img[mask, 2] = alpha*b + (1 - alpha)*img[mask, 2]   
        # img[edge, :] = 255
    
    for _, det in enumerate(detections):
        bbox = det["bbox"]
        x1, y1, x2, y2 = bbox

        r = int(255*colors[0][0])
        g = int(255*colors[0][1])
        b = int(255*colors[0][2])

        img2 = cv2.rectangle(
            img2,
            (x1, y1),
            (x2, y2),
            (r, g, b),
            1
        )

        # cv2.putText(
        #     img,
        #     f"{det['score']:.2f}",
        #     (x1, y1),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.5,
        #     (255, 255, 255),
        #     1
        # )
    
    # concat 
    img = np.concatenate((np.array(img2), np.array(img)), axis=1)
    
    if save_path is not None:
        Image.fromarray(np.uint8(img)).save(save_path)
    return img