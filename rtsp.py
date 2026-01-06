import os
from webcam_onnx import main

os.environ.setdefault("SRC", "rtsp")

# BẮT BUỘC: set RTSP_URL khi chạy
# ví dụ:
# RTSP_URL="rtsp://user:pass@ip:554/..." python3 rtsp.py
os.environ.setdefault("RTSP_URL", "")

# nếu camera H265 thì set RTSP_CODEC=h265
os.environ.setdefault("RTSP_CODEC", "h264")
os.environ.setdefault("RTSP_LATENCY", "0")

os.environ.setdefault("ONNX_WEIGHTS", "./model/LP_detector_nano_61.onnx")
os.environ.setdefault("SHOW", "1")      # SSH/headless => SHOW=0

os.environ.setdefault("CONF", "0.25")
os.environ.setdefault("IOU", "0.45")
os.environ.setdefault("IMG_SIZE", "640")

main()
