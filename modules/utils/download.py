# Standard Library
import os, sys
import logging
from typing import Optional
from pathlib import Path

# Modules
from modules import PROJECT_DIR, CHECKPOINTS_ROOT
from modules.megapose.config import CHECKPOINTS_DIR as MEGAPOSE_CHECKPOINTS_DIR
from modules.utils.logging import get_logger

logger = get_logger(__name__)

RCLONE_CFG_PATH = PROJECT_DIR / "modules/utils/rclone.conf"
RCLONE_ROOT = "inria_data:"


def download_fastsam(
    output_dir: Optional[Path] = CHECKPOINTS_ROOT / 'fastsam',
) -> None:
    os.makedirs(str(output_dir), exist_ok=True)
    # download fastsam for CNOS
    command = f"gdown --no-cookies --no-check-certificate -O '{output_dir}/FastSAM-x.pt' 1m1sjY4ihXBU1fZXdQ-Xdj-mDltW-2Rqv"
    # start download
    logger.info(f"Start downloading FastSAM checkpoints for CNOS: {command}")
    os.system(command)

def download_megapose(
    output_dir: Optional[Path] = MEGAPOSE_CHECKPOINTS_DIR,
) -> None:
    os.makedirs(str(output_dir), exist_ok=True)
    # download megapose
    rclone_path = RCLONE_ROOT + "megapose-models/"
    local_path = output_dir
    command = f'rclone copyto "{rclone_path}" "{local_path}" --exclude *epoch* -P --config "{RCLONE_CFG_PATH}"'
    # start download
    logger.info(f"Start downloading checkpoints for MegaPose: {command}")
    os.system(command)

def auto_download_default(
    fastsam_dir: Optional[Path] = CHECKPOINTS_ROOT / 'fastsam',
    megapose_dir: Optional[Path] = MEGAPOSE_CHECKPOINTS_DIR,
    megapose_type: Optional[str] = "megapose-1.0-RGB-multi-hypothesis"
) -> None:
    # check if fastsam exist
    if os.path.exists(str(fastsam_dir / "FastSAM-x.pt")):
        logger.info(f"Requirement already satisfied: {fastsam_dir / 'FastSAM-x.pt'}, no update.")
    else:
        download_fastsam()

    # check megapose
    from megapose.utils.load_model import NAMED_MODELS
    model_info = NAMED_MODELS[megapose_type]
    check_coarse = os.path.exists(
        str(megapose_dir / model_info['coarse_run_id'])
    ) and os.path.exists(
        str(megapose_dir / model_info['coarse_run_id'] / "checkpoint.pth.tar")
    )
    
    check_refine = os.path.exists(
        str(megapose_dir / model_info['refiner_run_id'])
    ) and os.path.exists(
        str(megapose_dir / model_info['refiner_run_id'] / "checkpoint.pth.tar")
    )

    # checking
    if check_coarse and check_refine:
        logger.info(f"Requirement already satisfied: {megapose_type}, no update.")
    else:
        download_megapose()
    