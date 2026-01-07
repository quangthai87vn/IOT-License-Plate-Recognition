#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shortcut chạy CSI camera cho ALPR.
Ví dụ:
  python3 csi.py --cam 0 --show 1
"""
import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cam", type=int, default=0, help="CSI sensor-id")
    p.add_argument("--csi_w", type=int, default=1640)
    p.add_argument("--csi_h", type=int, default=1232)
    p.add_argument("--csi_fps", type=int, default=30)
    p.add_argument("--flip", type=int, default=0)
    p.add_argument("--out_w", type=int, default=1280)
    p.add_argument("--out_h", type=int, default=720)
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--nms", type=float, default=0.45)
    p.add_argument("--show", type=int, default=1)
    return p.parse_args()


def main():
    a = parse_args()
    script = Path(__file__).resolve().parent / "webcam_onnx.py"
    cmd = [
        sys.executable, str(script),
        "--source", "csi",
        "--cam", str(a.cam),
        "--csi_w", str(a.csi_w),
        "--csi_h", str(a.csi_h),
        "--csi_fps", str(a.csi_fps),
        "--flip", str(a.flip),
        "--out_w", str(a.out_w),
        "--out_h", str(a.out_h),
        "--conf", str(a.conf),
        "--nms", str(a.nms),
        "--show", str(a.show),
    ]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
