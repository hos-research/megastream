# Standard Library
import numpy as np
from typing import List, Union, Tuple, Optional
from pathlib import Path
import queue
import threading
import time

# Third Party
from tqdm import tqdm
import pandas as pd

# Modules
from modules import PROJECT_DIR, CHECKPOINTS_ROOT
from modules.cnos import CNOS
from modules.megapose import MegaPose, Visualizer
from modules.megapose.inference.types import PoseEstimatesType
from modules.deepac import DeepAC, DeepAC_Pose

# Utils
from modules.utils.download import auto_download_default
from modules.utils.logging import get_logger

logger = get_logger(__name__)

class MegaStream:
    # Model
    detector: CNOS
    estimator: MegaPose
    tracker: DeepAC
    visualizer: Visualizer
    use_depth: bool
    # Frame info
    reinit: bool
    tracking: bool
    pose6d: PoseEstimatesType
    pose_track: DeepAC_Pose
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
        image_size: Tuple[int, int],    # (width, height)
        mesh_path: Union[Path, str],    # path to mesh files (ply/obj)
        # optional args
        sync: Optional[bool] = False,   # synchronize processing each frame
        calib: Optional[Path] = None,   # path to calibration file
        mesh_units: Optional[str] = 'm',    # mesh units, 'm' or 'mm'
        mesh_label: Optional[str] = None,   # mesh label
        auto_download: Optional[bool] = True,   # auto download checkpoints
        checkpoint_root: Optional[str] = CHECKPOINTS_ROOT,  # root dir for checkpoints
        apply_tracking: Optional[bool] = False,             # apply tracking or not
        use_depth: Optional[bool] = False,                  # use depth or not
        dinov2_type: Optional[str] = 'dinov2_vitl14',       # dinov2 model type
        log: Optional[bool] = False,                        # log iteration info
    ) -> None:
        # set megapose model
        self.use_depth = use_depth
        megapose_type = 'megapose-1.0-RGB-multi-hypothesis' if not use_depth else 'megapose-1.0-RGB-multi-hypothesis-icp'

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
            print(' ==> Auto Download Checkpoints at Default Path')
            auto_download_default(
                megapose_type=megapose_type
            )

        mesh_path = Path(mesh_path).resolve()
        # log info
        print(f' ==> Pipeline Image Size: {image_size}')
        print(f' ==> Object Registered: {mesh_path}')

        # get label
        if mesh_label is None: mesh_label = str(mesh_path.stem)
        self.mesh_label = mesh_label
        
        # init cnos
        print(' ==> Loading CNOS')
        self.detector = CNOS(
            checkpoint = checkpoint_root / 'fastsam' / 'FastSAM-x.pt',
            image_size = max(image_size[0], image_size[1]),
            dinov2_type = dinov2_type,
            mesh_path=mesh_path,
            label=mesh_label
        )

        # init megapose
        print(' ==> Loading MegaPose')
        megapose6d = MegaPose(
            model_type=megapose_type,
            object_path=mesh_path,
            label=mesh_label,
            mesh_units=mesh_units,
            intrinsic=image_size if calib is None else calib,
        )
        self.estimator = megapose6d
        self.visualizer = Visualizer(self.estimator)

        # init tracker 
        self.apply_tracking = apply_tracking
        self.tracking = False
        if apply_tracking:
            print(' ==> Loading DeepAC')
            self.tracker = DeepAC(
                intrinsic=image_size,
                template_path=mesh_path.parent / f'{mesh_path.stem}.pkl',
                config_path=checkpoint_root / 'deep_ac' / 'train_cfg.yml',
                checkpoint_path=checkpoint_root / 'deep_ac' / 'model_last.ckpt'
            )

        # run stream thread
        self.thread_ = threading.Thread(target=self.work_loop_)
        self.thread_.start()
    
    def thredshold(
        self,
        detect: Optional[float] = 0.5,
        refine: Optional[float] = 0.75
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
        detections: List[dict] = None,
        depth: Optional[np.ndarray] = None
    ) -> Tuple[PoseEstimatesType, dict]:
        TCO, extra = self.estimator.estimate(
            frame=frame,
            coarse_pose=coarse,
            detections=detections,
            depth=depth if self.use_depth else None
        )
        score = extra["refine"]["score"][0]
        return TCO, score

    def track(
        self,
        frame: np.ndarray,
        last_pose: Union[DeepAC_Pose, np.ndarray]
    ) -> Tuple[DeepAC_Pose, float]:
        pose = self.tracker.Track(
            image=frame,
            last_pose=last_pose,
            reinit=self.reinit
        )
        # convert 
        pose44 = DeepAC.convert_Pose44_numpy(pose)
        labeled_poses = [{
            'label': self.mesh_label,
            'TCO' : pose44
        }]
        # scoring
        score = self.estimator.score(
            frame=frame,
            data_TCO=self.estimator.convert_pose(labeled_poses)
        )
        return pose, score 

    def iterate(
        self,
        frame: np.ndarray,
        depth: Optional[np.ndarray] = None
    ) -> None:

        coarse = self.pose6d
        detections = None

        if self.reinit or coarse is None:
            # detect
            detections, score = self.detect(frame=frame)
            if score < self.threshold_detect:
                self.reinit = True
                self.pose6d = None
                self.score = -score
                return None, -score
        
        if (self.apply_tracking and not self.tracking) or not self.apply_tracking:
            # coarse and refine
            refined, score = self.estimate(frame=frame, coarse=coarse, detections=detections, depth=depth)
            # check score
            self.reinit = (score < self.threshold_refine)
            # update pose
            self.pose6d = refined
            self.score = score
            # init tracking pose
            self.pose_track = MegaPose.convert_TCO_dict(refined)[self.mesh_label]

        elif self.apply_tracking and self.tracking:
            # track
            new_pose, score = self.track(frame=frame, last_pose=self.pose_track)
            # check score
            self.reinit = (score < self.threshold_refine)
            # update pose
            self.pose_track = new_pose
            self.score = score
        
        # set tracking flag
        if self.apply_tracking: 
            self.tracking = not self.reinit
        
        # convert to dict
        # pose6d_dict = MegaPose.convert_TCO_dict(refined)
        # return pose6d_dict, score

    def Push(
        self,
        frame: np.ndarray,
        depth: Optional[np.ndarray] = None
    ) -> int:
        with self.in_queue_lock_:
            self.in_queue_.put((self.ids_, frame, depth))
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
        # tracking or not
        if self.apply_tracking and self.tracking:
            pose_track = self.pose_track
            if isinstance(pose_track, DeepAC_Pose):
                pose_track = DeepAC.convert_Pose44_numpy(pose_track)
            pose6d_dict = {self.mesh_label: pose_track}
        else:
            pose6d_dict = MegaPose.convert_TCO_dict(self.pose6d)
        
        return pose6d_dict, self.score

    def Render(
        self,
        frame: np.ndarray,
        pose6d: dict,
        show_contour: Optional[bool] = False
    ) -> np.ndarray:
        renderings = self.visualizer.render(pose6d)
        fig: np.ndarray = None
        if show_contour:
            fig = self.visualizer.contour_mesh_overlay(frame, renderings)
        else:
            fig = self.visualizer.mesh_overlay(frame, renderings)
        return fig

    def Release(
        self
    ) -> None:
        self.event_.set()
        self.in_queue_.put((None, None, None))
        self.thread_.join()
        # release MegaPose resources
        self.estimator.release()

    # threading
    def work_loop_(
        self
    ) -> None:
        print(' ==> Stream Thread Started')
        while True:
            id_, frame, depth = self.in_queue_.get()
            if frame is None: break
            # skip frame
            if not self.sync_:
                with self.in_queue_lock_:
                    while not self.in_queue_.empty():
                        id_, frame, depth = self.in_queue_.get()
                if frame is None: break
            # iter
            start = time.time()
            self.iterate(frame=frame, depth=depth)
            end = time.time()
            # log
            if self.log_: tqdm.write(f' [id={id_}] acc={self.score:.2f}, time={(end - start):.3f}')
            # notify event
            if self.sync_: self.event_.set()
        print(' ==> Stream Thread Exited')

