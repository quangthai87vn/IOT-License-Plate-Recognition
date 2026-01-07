#!/usr/bin/env python3
import subprocess
import sys

def main():
    # Usage:
    #   python3 rtsp.py rtsp://ip:8554/xxx
    # or:
    #   python3 rtsp.py --rtsp rtsp://ip:8554/xxx
    args = sys.argv[1:]
    cmd = ["python3", "webcam_onnx.py", "--source", "rtsp"]

    if len(args) >= 1 and not args[0].startswith("-"):
        # positional RTSP
        cmd += ["--rtsp", args[0]] + args[1:]
    else:
        cmd += args

    raise SystemExit(subprocess.call(cmd))

if __name__ == "__main__":
    main()
