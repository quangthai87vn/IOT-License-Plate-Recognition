# rtsp.py - ép cv2.VideoCapture(0) => RTSP URL rồi chạy webcam.py
import os
import sys
import runpy
import cv2

FORCE_INDEX = int(os.getenv("FORCE_INDEX", "0"))

# nhận RTSP theo:
# 1) argument: python3 rtsp.py "rtsp://..."
# 2) env: RTSP_URL="rtsp://..."
rtsp = sys.argv[1].strip() if len(sys.argv) >= 2 else os.getenv("RTSP_URL", "").strip()
if not rtsp:
    raise RuntimeError('Thiếu RTSP URL. Dùng: python3 rtsp.py "rtsp://..." hoặc set env RTSP_URL')

_real = cv2.VideoCapture

def patched_VideoCapture(*args, **kwargs):
    if len(args) >= 1 and (args[0] == FORCE_INDEX or args[0] == str(FORCE_INDEX)):
        return _real(rtsp)
    return _real(*args, **kwargs)

cv2.VideoCapture = patched_VideoCapture

print("[rtsp.py] Using RTSP_URL =", rtsp)
runpy.run_path("webcam.py", run_name="__main__")
