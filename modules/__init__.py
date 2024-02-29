from pathlib import Path
import logging
import os

# set basic dirs
PROJECT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINTS_ROOT = PROJECT_DIR / "data" / "checkpoints"
LOGGING_LEVEL = logging.INFO

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["EGL_VISIBLE_DEVICES"] = "0"