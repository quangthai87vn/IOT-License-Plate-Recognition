import os, runpy

os.environ["SRC"] = "rtsp"
os.environ.setdefault("SHOW", "1")  # SSH headless thì set SHOW=0

# bắt buộc set RTSP_URL trước khi chạy
# ví dụ: RTSP_URL="rtsp://user:pass@ip/..." python3 rtsp.py
runpy.run_path("webcam_onnx.py", run_name="__main__")
