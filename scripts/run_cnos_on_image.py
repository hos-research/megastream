# Standard Library
import argparse
import json
import os
from pathlib import Path

# Third Party
import numpy as np
import cv2

# CNOS
from modules.cnos import CNOS
from modules.cnos import visualize as CNOS_Visualize

# Logging
from modules.utils.logging import get_logger, set_logging_level

logger = get_logger(__name__)

# python -m scripts.test_cnos --input local_data/image_test/images/0.png --object local_data/image_test/Object/Object.ply

if __name__ == "__main__":
    set_logging_level("info")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--object", type=str)
    parser.add_argument("--checkpoint", type=str)
    # parser.add_argument("--output", type=str, default="output.mp4")

    args = parser.parse_args()

    detector = CNOS(
        checkpoint=Path(args.checkpoint)
    )

    detector.register_object(
        mesh_path=Path(args.object),
    )

    image = cv2.imread(args.input)
    res = detector.detect(image)
    with open('./result.json', 'w') as f:
        json.dump(res, f)
    mage = CNOS_Visualize(image, res, './result.png')

