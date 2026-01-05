# csi.py
# Wrapper chạy lại webcam.py nhưng ép nguồn camera = CSI (IMX219) qua GStreamer
# Không cần sửa webcam.py (miễn là webcam.py dùng cv2.VideoCapture(0) hoặc (1))

import os
import runpy
import cv2


def gstreamer_pipeline(
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    flip: int = 0,
) -> str:
    # flip: 0=none, 2=180deg, 4=horizontal, 6=vertical... (tuỳ setup)
    return (
        "nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, framerate={fps}/1 ! "
        f"nvvidconv flip-method={flip} ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink drop=1 sync=false"
    )


# Lấy config từ env (để bạn chỉnh nhanh trong docker-compose)
CSI_W = int(os.getenv("CSI_W", "1280"))
CSI_H = int(os.getenv("CSI_H", "720"))
CSI_FPS = int(os.getenv("CSI_FPS", "30"))
CSI_FLIP = int(os.getenv("CSI_FLIP", "0"))

# Mặc định ép mọi VideoCapture(0) => CSI
FORCE_CSI = os.getenv("FORCE_CSI", "1") == "1"
FORCE_INDEX = int(os.getenv("FORCE_INDEX", "0"))  # webcam.py hay dùng 0

_real_VideoCapture = cv2.VideoCapture


def _patched_VideoCapture(*args, **kwargs):
    """
    Nếu webcam.py gọi cv2.VideoCapture(0) (hoặc index FORCE_INDEX)
    thì thay bằng CSI GStreamer pipeline.
    Còn lại giữ nguyên hành vi cũ.
    """
    if FORCE_CSI and len(args) >= 1:
        src = args[0]
        if src == FORCE_INDEX or src == str(FORCE_INDEX):
            gst = gstreamer_pipeline(CSI_W, CSI_H, CSI_FPS, CSI_FLIP)
            # ưu tiên CAP_GSTREAMER cho CSI
            return _real_VideoCapture(gst, cv2.CAP_GSTREAMER)
    return _real_VideoCapture(*args, **kwargs)


cv2.VideoCapture = _patched_VideoCapture

# Chạy lại webcam.py như đang chạy trực tiếp: python webcam.py
runpy.run_path("webcam.py", run_name="__main__")
