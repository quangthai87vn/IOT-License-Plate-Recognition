# rtsp.py
import os
import sys
import runpy

os.environ.setdefault("SRC", "rtsp")
os.environ.setdefault("SHOW", "1")
os.environ.setdefault("IMG_SIZE", "640")
os.environ.setdefault("CONF", "0.25")
os.environ.setdefault("IOU", "0.45")
os.environ.setdefault("RTSP_LATENCY", "200")

# RTSP nhận theo:
# 1) env RTSP_URL="rtsp://...."
# 2) arg: python3 rtsp.py "rtsp://...."
if len(sys.argv) >= 2 and sys.argv[1].strip():
    os.environ["RTSP_URL"] = sys.argv[1].strip()
else:
    os.environ.setdefault("RTSP_URL", "")

if not os.environ["RTSP_URL"]:
    raise RuntimeError('Thiếu RTSP URL. Dùng: python3 rtsp.py "rtsp://..." hoặc set env RTSP_URL')

# Models
os.environ.setdefault("DET_ONNX", "./model/LP_detector_nano_61.onnx")
os.environ.setdefault("OCR_ONNX", "./model/LP_ocr_nano_62.onnx")

runpy.run_path("webcam_onnx.py", run_name="__main__")
