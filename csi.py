import os
from webcam_onnx import main

os.environ.setdefault("SRC", "csi")
os.environ.setdefault("CAM_W", "1280")
os.environ.setdefault("CAM_H", "720")
os.environ.setdefault("CAM_FPS", "30")        # ép 30 cho mượt
os.environ.setdefault("SENSOR_MODE", "3")
os.environ.setdefault("FLIP", "0")

os.environ.setdefault("ONNX_WEIGHTS", "./model/LP_detector_nano_61.onnx")

# SSH thì để SHOW=0
os.environ.setdefault("SHOW", "1")

os.environ.setdefault("CONF", "0.25")
os.environ.setdefault("IOU", "0.45")
os.environ.setdefault("IMG_SIZE", "640")

main()
