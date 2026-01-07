#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""rtsp.py - wrapper chạy RTSP cho nhanh.

Usage:
  python3 rtsp.py "rtsp://192.168.50.2:8554/mac" --show 1
  python3 rtsp.py "rtsp://..." --latency 200 --tcp 1 --show 1
"""

import argparse
import os
import sys


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('rtsp', help='RTSP url, vd: rtsp://192.168.50.2:8554/mac')
    ap.add_argument('--show', type=int, default=1)
    ap.add_argument('--latency', type=int, default=200)
    ap.add_argument('--tcp', type=int, default=1)
    ap.add_argument('--rtsp_w', type=int, default=1280)
    ap.add_argument('--rtsp_h', type=int, default=720)

    # optional override engine paths
    ap.add_argument('--det_engine', type=str, default="model/LP_detector_nano_61_fp16.engine")
    ap.add_argument('--ocr_engine', type=str, default="model/LP_ocr_nano_62_fp16.engine")

    args = ap.parse_args(argv)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import webcam_onnx

    run_argv = [
        "--source", "rtsp",
        "--rtsp", args.rtsp,
        "--rtsp_latency", str(args.latency),
        "--rtsp_tcp", str(args.tcp),
        "--rtsp_w", str(args.rtsp_w),
        "--rtsp_h", str(args.rtsp_h),
        "--show", str(args.show),
        "--det_engine", args.det_engine,
        "--ocr_engine", args.ocr_engine,
    ]
    return webcam_onnx.main(run_argv)


if __name__ == "__main__":
    raise SystemExit(main())
