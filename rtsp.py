#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

def main():
    # usage:
    #   python3 rtsp.py "rtsp://user:pass@ip:554/..." 
    # or set env RTSP_URL
    rtsp_url = ""
    if len(sys.argv) >= 2 and sys.argv[1].startswith("rtsp"):
        rtsp_url = sys.argv[1]
        rest = sys.argv[2:]
    else:
        rtsp_url = os.getenv("RTSP_URL", "")
        rest = sys.argv[1:]

    if not rtsp_url:
        raise RuntimeError('Thiếu RTSP URL. Dùng: python3 rtsp.py "rtsp://..." hoặc set env RTSP_URL')

    os.environ["SRC"] = "rtsp"
    os.environ["RTSP_URL"] = rtsp_url
    os.execvp("python3", ["python3", "webcam_onnx.py"] + rest)

if __name__ == "__main__":
    main()
