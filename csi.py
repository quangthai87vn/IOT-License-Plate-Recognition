#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shortcut runner for CSI camera.

Usage (inside container):
  python3 csi.py --show
  python3 csi.py --csi_w 1280 --csi_h 720 --csi_fps 30 --show
"""

import sys


def main():
    from webcam_onnx import main as alpr_main

    argv = ["webcam_onnx.py", "--source", "csi"] + sys.argv[1:]
    sys.argv = argv
    alpr_main()


if __name__ == "__main__":
    main()
