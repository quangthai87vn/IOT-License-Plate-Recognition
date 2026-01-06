import os

os.environ["SRC"] = "rtsp"
os.environ["RTSP_URL"] = "rtsp://192.168.50.2:8554/mac"
os.environ.setdefault("RTSP_CODEC", "h264")
os.environ.setdefault("RTSP_LATENCY", "0")

os.environ.setdefault("ONNX_WEIGHTS", "./model/LP_detector_nano_61.onnx")
os.environ.setdefault("SHOW", "1")
os.environ.setdefault("IMG_SIZE", "640")
os.environ.setdefault("CONF", "0.25")
os.environ.setdefault("IOU", "0.45")

from webcam_onnx import main   # <-- IMPORT sau khi set env
main()
