#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shortcut runner for RTSP.

Supports:
  python3 rtsp.py --rtsp rtsp://ip:8554/stream --show
  python3 rtsp.py rtsp://ip:8554/stream --show
"""

import sys


def main():
    from webcam_onnx import main as alpr_main

    args = sys.argv[1:]

    # If user passed URL as positional, convert -> --rtsp URL
    if args and (not args[0].startswith("-")):
        args = ["--rtsp", args[0]] + args[1:]

    argv = ["webcam_onnx.py", "--source", "rtsp"] + args
    sys.argv = argv
    alpr_main()


if __name__ == "__main__":
    main()
