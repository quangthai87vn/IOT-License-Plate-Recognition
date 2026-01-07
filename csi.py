#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shortcut chạy CSI.

Ví dụ:
  python3 csi.py
  python3 csi.py --show --csi_w 1280 --csi_h 720 --csi_fps 60

Nó chỉ gọi webcam_onnx.py với --source csi.
"""

import sys
import subprocess


def main():
    cmd = [sys.executable, "webcam_onnx.py", "--source", "csi"] + sys.argv[1:]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
