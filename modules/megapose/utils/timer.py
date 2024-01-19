"""
Copyright (c) 2022 Inria & NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""



# Standard Library
import time
import datetime
import torch

class Timer:
    def __init__(self):
        self.start_time = None
        self.elapsed = datetime.timedelta()
        self.is_running = False

    def reset(self):
        self.start_time = None
        self.elapsed = 0.0
        self.is_running = False

    def start(self):
        self.elapsed = datetime.timedelta()
        self.is_running = True
        self.start_time = datetime.datetime.now()
        return self

    def pause(self):
        if self.is_running:
            self.elapsed += datetime.datetime.now() - self.start_time
            self.is_running = False

    def resume(self):
        if not self.is_running:
            self.start_time = datetime.datetime.now()
            self.is_running = True

    def stop(self):
        self.pause()
        elapsed = self.elapsed
        self.reset()
        return elapsed

class SimpleTimer:
    def __init__(self) -> None:
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()

    def end(self):
        self.stop()

    def elapsed(self):
        return self.end_time - self.start_time


class CudaTimer:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.elapsed_sec = None

        self.start_called = False
        self.end_called = False

    def start(self) -> None:
        if not self.enabled:
            return

        self.start_called = True
        self.start_event.record()

    def end(self) -> None:
        if not self.enabled:
            return

        self.end_called = True
        self.end_event.record()
        torch.cuda.synchronize()
        self.elapsed_sec = self.start_event.elapsed_time(self.end_event) / 1000.0

    def stop(self) -> None:
        self.end()

    def elapsed(self) -> float:
        """Return the elapsed time (in seconds)."""
        if not self.enabled:
            return 0.0

        if not self.start_called:
            raise ValueError("You must call CudaTimer.start() before querying the elapsed time")

        if not self.end_called:
            raise ValueError("You must call CudaTimer.end() before querying the elapsed time")

        return self.elapsed_sec