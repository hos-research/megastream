# Standard Library
from pathlib import Path
import logging
import numpy as np
from typing import Union, Optional

# Third Party
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import argparse

# MegaStream
from src.megastream import MegaStream
from src.megastream import PROJECT_DIR

app_dir = Path(__file__).parent

app = Flask(__name__, template_folder=str(app_dir))
app.debug = False
socketio = SocketIO(app)

# megastream pipe
pipe: MegaStream = None

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('data')
def handle_data(data):
    frame = np.frombuffer(data, dtype=np.uint8).reshape((480, 640, 3))
    pipe.Push(frame=frame)
    data, acc = pipe.Get()
    # print(data)
    emit('data', data)

def main(
    mesh_path: Union[str, Path],
    port: Optional[int] = 9800
):

    logging.info(' => initializing megastream pipe')

    pipe = MegaStream(
        image_size=(640, 480),
        mesh_path=mesh_path,
        log=True
    )

    logging.info(' => starting websocket server')

    socketio.run(app, debug=True, use_reloader=False, port=port)



