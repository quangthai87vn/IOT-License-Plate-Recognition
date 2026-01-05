# csi.py - ép cv2.VideoCapture(0) => CSI GStreamer pipeline rồi chạy webcam.py
import os
import runpy
import cv2

def gstreamer_pipeline(width=1280, height=720, fps=30, flip=0):
    return (
        "nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, framerate={fps}/1 ! "
        f"nvvidconv flip-method={flip} ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! "
        "appsink drop=1 sync=false"
    )

CSI_W = int(os.getenv("CSI_W", "1280"))
CSI_H = int(os.getenv("CSI_H", "720"))
CSI_FPS = int(os.getenv("CSI_FPS", "30"))
CSI_FLIP = int(os.getenv("CSI_FLIP", "0"))
FORCE_INDEX = int(os.getenv("FORCE_INDEX", "0"))

_real = cv2.VideoCapture

def patched_VideoCapture(*args, **kwargs):
    if len(args) >= 1 and (args[0] == FORCE_INDEX or args[0] == str(FORCE_INDEX)):
        gst = gstreamer_pipeline(CSI_W, CSI_H, CSI_FPS, CSI_FLIP)
        return _real(gst, cv2.CAP_GSTREAMER)
    return _real(*args, **kwargs)

cv2.VideoCapture = patched_VideoCapture

print(f"[csi.py] CSI {CSI_W}x{CSI_H}@{CSI_FPS} flip={CSI_FLIP}")
runpy.run_path("webcam.py", run_name="__main__")
