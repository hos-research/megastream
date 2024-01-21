# Standard Library
from typing import Optional, Tuple
import numpy as np
import threading

# hl2ss
import modules.hl2ss.hl2ss
import modules.hl2ss.hl2ss_lnm
from .hl2ss import StreamMode, StreamPort, VideoProfile
from .hl2ss_lnm import rx_pv
from .hl2ss_lnm import start_subsystem_pv, stop_subsystem_pv, download_calibration_pv


class HoloLens2:
    client: rx_pv

    def __init__(
        self,
        host: str,
        resolution: Optional[Tuple] = (1280, 720),
        fps: Optional[int] = 30,
        format: Optional[str] = 'bgr24'
    ) -> None:
        mode = StreamMode.MODE_1
        width = resolution[0]
        height = resolution[1]
        self.host = host

        print(f'connecting to hololens2 on {host}')
        print(f'resolution: {width}x{height}')

        start_subsystem_pv(
            host=host,
            port=StreamPort.PERSONAL_VIDEO,
        )

        self.client = rx_pv(
            host=host,
            port=StreamPort.PERSONAL_VIDEO,
            mode=mode,
            width=width,
            height=height,
            framerate=fps,
            divisor=1,
            profile=VideoProfile.H265_MAIN,
            decoded_format=format
        )

        self.client.open()
        print('connected.')
    
    def __del__(self) -> None:
        self.client.close()
        stop_subsystem_pv(
            host=self.host,
            port=StreamPort.PERSONAL_VIDEO,
        )

    def get(self) -> np.ndarray:
        data = self.client.get_next_packet()
        return np.array(data.payload.image)
