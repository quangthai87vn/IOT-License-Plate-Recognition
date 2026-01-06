# csi.py
import os
import runpy

# Bạn chỉnh nhanh ở đây hoặc set ENV khi chạy
os.environ.setdefault("SRC", "csi")
os.environ.setdefault("SHOW", "1")
os.environ.setdefault("IMG_SIZE", "640")
os.environ.setdefault("CONF", "0.25")
os.environ.setdefault("IOU", "0.45")

# CSI tuning
os.environ.setdefault("CSI_W", "1280")
os.environ.setdefault("CSI_H", "720")
os.environ.setdefault("CSI_FPS", "30")     # ép 30fps
os.environ.setdefault("CSI_FLIP", "0")

# Nếu nó cứ nhảy qua 120fps mode=5, bạn ép mode khác:
# IMX219 thường: mode 4 ~ 1280x720@60, mode 3 ~ 1640x1232@30, mode 2 ~ 1920x1080@30 (tuỳ board/cam)
# Thử: "3" hoặc "2" nếu muốn 30fps ổn định
os.environ.setdefault("CSI_MODE", "3")

# Models
os.environ.setdefault("DET_ONNX", "./model/LP_detector_nano_61.onnx")
os.environ.setdefault("OCR_ONNX", "./model/LP_ocr_nano_62.onnx")

runpy.run_path("webcam_onnx.py", run_name="__main__")
