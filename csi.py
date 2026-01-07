#!/usr/bin/env python3
# csi.py - wrapper chạy CSI nhưng vẫn dùng chung webcam_onnx.py

import os
import sys

def main():
    # cho phép: python3 csi.py  (hoặc thêm args như --show 1 --csi_w 1280 ...)
    args = ["python3", "webcam_onnx.py", "--source", "csi"] + sys.argv[1:]

    # chạy đúng file trong cùng thư mục (tránh chạy nhầm file cũ)
    os.execvp(args[0], args)

if __name__ == "__main__":
    main()
