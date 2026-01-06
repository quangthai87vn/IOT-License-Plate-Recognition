import os
from webcam_onnx import main

# CSI default (mượt, đúng bài)
os.environ.setdefault("SRC", "csi")
os.environ.setdefault("CAM_W", "1280")
os.environ.setdefault("CAM_H", "720")
os.environ.setdefault("CAM_FPS", "30")        # ép 30fps
os.environ.setdefault("SENSOR_MODE", "3")     # mode 3 (30fps ổn)
os.environ.setdefault("FLIP", "0")

# ONNX weights (đổi nếu file khác)
os.environ.setdefault("ONNX_WEIGHTS", "./model/LP_detector_nano_61.onnx")

# SHOW=0 nếu chạy SSH/headless
os.environ.setdefault("SHOW", "1")

# Threshold
os.environ.setdefault("CONF", "0.25")
os.environ.setdefault("IOU", "0.45")
os.environ.setdefault("IMG_SIZE", "640")

main()
