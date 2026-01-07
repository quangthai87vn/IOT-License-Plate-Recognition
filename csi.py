#!/usr/bin/env python3
import subprocess
import sys

def main():
    # pass-through args to webcam_onnx.py
    cmd = ["python3", "webcam_onnx.py", "--source", "csi"]
    cmd += sys.argv[1:]
    raise SystemExit(subprocess.call(cmd))

if __name__ == "__main__":
    main()
