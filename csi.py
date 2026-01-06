import os, runpy

os.environ["SRC"] = "csi"
os.environ.setdefault("CAM_W", "1280")
os.environ.setdefault("CAM_H", "720")
os.environ.setdefault("CAM_FPS", "30")
os.environ.setdefault("SENSOR_MODE", "3")
os.environ.setdefault("FLIP", "0")
os.environ.setdefault("SHOW", "1")  # SSH headless thì set SHOW=0

runpy.run_path("webcam_onnx.py", run_name="__main__")
