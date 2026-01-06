#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import subprocess

def main():
    # Optional: cleanup argus to reduce "cam mở xíu rồi tắt"
    if os.getenv("CLEAR_ARGUS", "0") == "1":
        subprocess.call("pkill -f gst-launch", shell=True)
        subprocess.call("sudo systemctl restart nvargus-daemon", shell=True)

    # Run main
    # You can override env like: IMG_SIZE=640 CONF=0.25 SKIP_OCR=2
    os.environ["SRC"] = "csi"
    os.execvp("python3", ["python3", "webcam_onnx.py"] + sys.argv[1:])

if __name__ == "__main__":
    main()
