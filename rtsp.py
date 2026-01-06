import os
import sys
import runpy

os.environ.setdefault("SRC", "rtsp")
os.environ.setdefault("SHOW", "1")

# RTSP URL from env OR argv
if "RTSP_URL" not in os.environ or not os.environ["RTSP_URL"].strip():
    if len(sys.argv) >= 2:
        os.environ["RTSP_URL"] = sys.argv[1]
    else:
        print("Usage:")
        print("  RTSP_URL='rtsp://ip:port/stream' python3 rtsp.py")
        print("  python3 rtsp.py rtsp://ip:port/stream")
        sys.exit(1)

# RTSP tuning
os.environ.setdefault("CODEC", "h264")
os.environ.setdefault("RTSP_LATENCY", "200")

# Models
os.environ.setdefault("DET_ENGINE", "./model/LP_detector_nano_61_fp16.engine")
os.environ.setdefault("OCR_ENGINE", "./model/LP_ocr_nano_62_fp16.engine")
os.environ.setdefault("DET_ONNX", "./model/LP_detector_nano_61.onnx")
os.environ.setdefault("OCR_ONNX", "./model/LP_ocr_nano_62.onnx")

# Performance knobs
os.environ.setdefault("SKIP", "0")
os.environ.setdefault("OCR_EVERY", "1")

runpy.run_path("webcam_onnx.py", run_name="__main__")
