import os
import runpy

# Force RTSP mode
os.environ["SRC"] = "rtsp"

# RTSP_URL must be set outside or here:
# os.environ["RTSP_URL"] = "rtsp://192.168.50.2:8554/mac"
# os.environ["RTSP_LATENCY"] = "200"

runpy.run_path("webcam_onnx.py", run_name="__main__")
