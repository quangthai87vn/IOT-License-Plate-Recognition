#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""csi.py - wrapper chạy CSI cho nhanh.

Usage:
  python3 csi.py --show 1
  python3 csi.py --csi_id 0 --csi_w 1280 --csi_h 720 --csi_fps 30 --show 1
"""

import argparse
import os
import sys


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--csi_id', type=int, default=0)
    ap.add_argument('--csi_w', type=int, default=1280)
    ap.add_argument('--csi_h', type=int, default=720)
    ap.add_argument('--csi_fps', type=int, default=30)
    ap.add_argument('--flip', type=int, default=0)
    ap.add_argument('--show', type=int, default=1)

    # optional override engine paths
    ap.add_argument('--det_engine', type=str, default="model/LP_detector_nano_61_fp16.engine")
    ap.add_argument('--ocr_engine', type=str, default="model/LP_ocr_nano_62_fp16.engine")

    args = ap.parse_args(argv)

    # Import main script from same folder
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import webcam_onnx

    run_argv = [
        "--source", "csi",
        "--csi_id", str(args.csi_id),
        "--csi_w", str(args.csi_w),
        "--csi_h", str(args.csi_h),
        "--csi_fps", str(args.csi_fps),
        "--flip", str(args.flip),
        "--show", str(args.show),
        "--det_engine", args.det_engine,
        "--ocr_engine", args.ocr_engine,
    ]
    return webcam_onnx.main(run_argv)


if __name__ == "__main__":
    raise SystemExit(main())
