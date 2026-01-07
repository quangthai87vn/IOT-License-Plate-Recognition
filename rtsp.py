#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import subprocess

def main():
    # cho phép gọi: python3 rtsp.py "rtsp://...." --show 1
    url = None
    extra = []
    for a in sys.argv[1:]:
        if a.startswith("rtsp://"):
            url = a
        else:
            extra.append(a)

    if not url:
        print('Usage: python3 rtsp.py "rtsp://ip:port/xxx" --show 1')
        sys.exit(1)

    cmd = ["python3", "webcam_onnx.py", "--source", "rtsp", "--rtsp", url] + extra
    raise SystemExit(subprocess.call(cmd))

if __name__ == "__main__":
    main()
