#!/usr/bin/env python3
# rtsp.py - wrapper chạy RTSP nhưng vẫn dùng chung webcam_onnx.py

import os
import sys

def main():
    # hỗ trợ 2 kiểu:
    # 1) python3 rtsp.py "rtsp://ip:8554/xxx"
    # 2) python3 rtsp.py --rtsp "rtsp://ip:8554/xxx" --show 1 ...
    rtsp_url = ""

    argv = sys.argv[1:]
    if len(argv) >= 1 and not argv[0].startswith("-"):
        rtsp_url = argv[0]
        argv = argv[1:]
    else:
        # tìm --rtsp
        if "--rtsp" in argv:
            i = argv.index("--rtsp")
            if i + 1 < len(argv):
                rtsp_url = argv[i + 1]

    if not rtsp_url:
        print('Usage:\n  python3 rtsp.py "rtsp://ip:8554/stream"\n  python3 rtsp.py --rtsp "rtsp://ip:8554/stream" --show 1')
        sys.exit(2)

    args = ["python3", "webcam_onnx.py", "--source", "rtsp", "--rtsp", rtsp_url] + argv
    os.execvp(args[0], args)

if __name__ == "__main__":
    main()
