import os
from webcam_onnx import main

os.environ.setdefault("SRC", "rtsp")
os.environ.setdefault("RTSP_URL", "")
os.environ.setdefault("RTSP_CODEC", "h264")
os.environ.setdefault("RTSP_LATENCY", "0")

os.environ.setdefault("ONNX_WEIGHTS", "./model/LP_detector_nano_61.onnx")

os.environ.setdefault("SHOW", "1")   # SSH -> SHOW=0
os.environ.setdefault("CONF", "0.25")
os.environ.setdefault("IOU", "0.45")
os.environ.setdefault("IMG_SIZE", "640")

main()
