# Standard Library
import argparse
import numpy as np
import threading
from pathlib import Path

# Third Party
import cv2
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

# MegaStream
from src.megastream import MegaStream

# HL2SS
from modules.hl2ss import HoloLens2

# Flask
from modules import PROJECT_DIR
app_dir = PROJECT_DIR / 'src'

app = Flask(__name__, template_folder=str(app_dir))
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')


def work_loop(args):
    resolution = (1280, 720)
    fixed_size = (640, 360)

    # init recorder
    # record = cv2.VideoWriter(
    #     args.record,
    #     cv2.VideoWriter_fourcc(*'mp4v'),
    #     30,
    #     fixed_size
    # )

    # init megastream
    # pipe = MegaStream(
    #     image_size=fixed_size,
    #     mesh_path=Path(args.object),
    #     auto_download=True,
    #     log=True
    # )

    # connect to hololens2
    device = HoloLens2(host=args.host, resolution=resolution)

    while True:
        # get rgb frame from hololens2
        frame = device.get()
        frame = cv2.resize(frame, fixed_size)
        # estimate pose6d from megastream
        # pipe.Push(frame)
        # pose, score = pipe.Get()
        # if score < 0: pose = None

        # render and record
        # frame = pipe.Render(frame, pose)
        # record.write(frame)
        # convert to rgba
        frame = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        socketio.emit('image', frame.tobytes())

    record.release()
    # pipe.Release()

def test_loop(args):
    print('in')
    while True:
        # get rgb frame from hololens2
        frame = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
        # convert to rgba
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        socketio.emit('image', frame.tobytes())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="10.129.138.250")
    parser.add_argument("--object", type=str)
    parser.add_argument("--out_dir", type=str, default='./')
    parser.add_argument("--test", action="store_true", default=False)

    args = parser.parse_args()
    
    thread = threading.Thread(
        target=work_loop if not args.test else test_loop,
        args=(args,)
    )
    thread.start()

    socketio.run(app, host='0.0.0.0', debug=True, port=13800)
    