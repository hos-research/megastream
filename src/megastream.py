# Standard Library
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
import queue
import threading
import time
from tqdm import tqdm

# Modules
from modules import CHECKPOINTS_ROOT
from modules.cnos import CNOS
from modules.megapose import MegaPose, Visualizer
from modules.megapose.inference.types import PoseEstimatesType
from modules.utils.download import auto_download_default
from modules.utils.logging import get_logger

logger = get_logger(__name__)

class MegaStream:
    # Model
    detector: CNOS
    estimator: MegaPose
    visualizer: Visualizer
    # Frame info
    reinit: bool
    pose6d: PoseEstimatesType
    score: float
    threshold_detect: float
    threshold_refine: float
    # threading
    in_queue_: queue.Queue
    in_queue_lock_: threading.Lock
    sync_: bool
    event_: threading.Event
    thread_: threading.Thread
    log_: bool
    ids_: int = 0


    def __init__(
        self,
        image_size: Tuple[int, int], # (width, height)
        mesh_path: Path,
        mesh_units: Optional[str] = 'm',
        mesh_label: Optional[str] = None,
        sync: Optional[bool] = False,
        auto_download: Optional[bool] = False,
        checkpoint_root: Optional[str] = CHECKPOINTS_ROOT,
        megapose_type: Optional[str] = 'megapose-1.0-RGB-multi-hypothesis',
        dinov2_type: Optional[str] = 'dinov2_vitl14',
        log: Optional[bool] = False,
    ) -> None:
        # init val
        self.reinit = True
        self.pose6d = None
        self.score = 0.0
        self.log_ = log
        
        # init threading
        self.in_queue_ = queue.Queue()
        self.in_queue_lock_ = threading.Lock()
        self.sync_ = sync
        self.event_ = threading.Event()
        
        # threshold
        self.thredshold()

        # checkpoints
        if auto_download:
            logger.info('Auto Download Checkpoints at Default Path')
            auto_download_default()

        mesh_path = mesh_path.resolve()
        # log info
        logger.info(f' ==> Pipeline Image Size: {image_size}')
        logger.info(f' ==> Object Registered: {mesh_path}')

        # get label
        if mesh_label is None: mesh_label = str(mesh_path.stem)
        
        # init cnos
        logger.info(' ==> Loading CNOS')
        self.detector = CNOS(
            checkpoint = checkpoint_root / 'fastsam' / 'FastSAM-x.pt',
            image_size = max(image_size[0], image_size[1]),
            dinov2_type = dinov2_type,
            mesh_path=mesh_path,
            label=mesh_label
        )

        # init megapose
        logger.info(' ==> Loading MegaPose')
        megapose6d = MegaPose(
            model_type=megapose_type,
            object_path=mesh_path,
            label=mesh_label,
            mesh_units=mesh_units,
            intrinsic=(image_size[0], image_size[1]),
        )
        self.estimator = megapose6d
        self.visualizer = Visualizer(self.estimator)

        # run stream thread
        self.thread_ = threading.Thread(target=self.work_loop_)
        self.thread_.start()
    
    def thredshold(
        self,
        detect: Optional[float] = 0.5,
        refine: Optional[float] = 0.85
    ) -> None:
        self.threshold_detect = detect
        self.threshold_refine = refine

    def detect(
        self, 
        frame: np.ndarray
    ) -> dict:
        detections = self.detector.detect(frame)
        score = detections[0]['score']
        return detections, score

    def estimate(
        self,
        frame: np.ndarray,
        coarse: Optional[PoseEstimatesType] = None,
        detections: List[dict] = None
    ) -> Tuple[PoseEstimatesType, dict]:
        TCO, extra = self.estimator.estimate(
            frame=frame,
            coarse_pose=coarse,
            detections=detections
        )
        score = extra["refine"]["score"][0]
        return TCO, score

    def iterate(
        self,
        frame: np.ndarray,
    ) -> Tuple[dict, float]:

        coarse = self.pose6d
        detections = None

        if self.reinit or coarse is None:
            # detect
            detections, score = self.detect(frame=frame)
            if score < self.threshold_detect:
                self.reinit = True
                self.score = score
                return None, score

        # coarse and refine
        refined, score = self.estimate(frame=frame, coarse=coarse, detections=detections)
        if score < self.threshold_refine:
            self.reinit = True
            self.score = score
            return None, score
        
        # update pose
        self.pose6d = refined
        self.score = score
        self.reinit = False
        pose6d_dict = MegaPose.convert_TCO_dict(refined)

        return pose6d_dict, score

    def Push(
        self,
        frame: np.ndarray,
    ) -> int:
        with self.in_queue_lock_:
            self.in_queue_.put((self.ids_, frame))
            self.ids_ += 1
            # drop if frame stuck
            if self.in_queue_.qsize() > 240:
                logger.warning('Too many frames in queue, dropping frames')
                while self.in_queue_.qsize() > 60:
                    self.in_queue_.get()
        # clear event
        if self.sync_: 
            self.event_.clear()
        return self.in_queue_.qsize()

    def Get(
        self,
    ) -> Tuple[dict, float]:
        if self.sync_: 
            self.event_.wait()
        pose6d_dict = MegaPose.convert_TCO_dict(self.pose6d)
        return pose6d_dict, self.score

    def Render(
        self,
        frame: np.ndarray,
        pose6d: dict,
    ) -> np.ndarray:
        renderings = self.visualizer.render(pose6d)
        fig = self.visualizer.contour_mesh_overlay(frame, renderings)
        return fig

    def Release(
        self
    ) -> None:
        self.event_.set()
        self.in_queue_.put((None, None))
        self.thread_.join()
        # release MegaPose resources
        self.estimator.release()

    # threading
    def work_loop_(
        self
    ) -> None:
        logger.info(' ==> Stream Thread Started')
        while True:
            id_, frame = self.in_queue_.get()
            if frame is None: break
            # skip frame
            if not self.sync_:
                with self.in_queue_lock_:
                    while not self.in_queue_.empty():
                        id_, frame = self.in_queue_.get()
                if frame is None: break
            # iter
            start = time.time()
            self.iterate(frame=frame)
            end = time.time()
            # log
            if self.log_: tqdm.write(f' [id={id_}] acc={self.score:.2f}, time={(end - start):.3f}')
            # notify event
            if self.sync_: self.event_.set()
        logger.info(' ==> Stream Thread Exited')

