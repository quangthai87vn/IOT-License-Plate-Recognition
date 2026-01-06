import os
import runpy

# CSI defaults (you can override by env)
os.environ.setdefault("SRC", "csi")
os.environ.setdefault("SHOW", "1")

# Tune CSI for stable FPS (avoid auto picking 120fps mode)
os.environ.setdefault("CSI_W", "1280")
os.environ.setdefault("CSI_H", "720")
os.environ.setdefault("CSI_FPS", "30")
os.environ.setdefault("CSI_MODE", "3")  # change if your IMX219 mode differs

# Models
os.environ.setdefault("DET_ENGINE", "./model/LP_detector_nano_61_fp16.engine")
os.environ.setdefault("OCR_ENGINE", "./model/LP_ocr_nano_62_fp16.engine")
os.environ.setdefault("DET_ONNX", "./model/LP_detector_nano_61.onnx")
os.environ.setdefault("OCR_ONNX", "./model/LP_ocr_nano_62.onnx")

# Performance knobs
os.environ.setdefault("SKIP", "0")
os.environ.setdefault("OCR_EVERY", "1")

runpy.run_path("webcam_onnx.py", run_name="__main__")
