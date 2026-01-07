#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shortcut chạy RTSP cho ALPR.

2 kiểu chạy:
  python3 rtsp.py --rtsp "rtsp://192.168.50.2:8554/mac" --show 1
hoặc:
  python3 rtsp.py "rtsp://192.168.50.2:8554/mac" --show 1
"""
import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("rtsp_url", nargs="?", default="", help="RTSP url (positional)")
    p.add_argument("--rtsp", type=str, default="", help="RTSP url (flag)")
    p.add_argument("--latency", type=int, default=100)
    p.add_argument("--udp", action="store_true")
    p.add_argument("--out_w", type=int, default=1280)
    p.add_argument("--out_h", type=int, default=720)
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--nms", type=float, default=0.45)
    p.add_argument("--show", type=int, default=1)
    return p.parse_args()


def main():
    a = parse_args()
    url = a.rtsp or a.rtsp_url
    if not url:
        print("ERROR: thiếu RTSP url. Ví dụ: python3 rtsp.py rtsp://IP:PORT/...", file=sys.stderr)
        raise SystemExit(2)

    script = Path(__file__).resolve().parent / "webcam_onnx.py"
    cmd = [
        sys.executable, str(script),
        "--source", "rtsp",
        "--rtsp", url,
        "--rtsp_latency", str(a.latency),
        "--out_w", str(a.out_w),
        "--out_h", str(a.out_h),
        "--conf", str(a.conf),
        "--nms", str(a.nms),
        "--show", str(a.show),
    ]
    if a.udp:
        cmd.append("--rtsp_udp")

    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
