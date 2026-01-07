#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess

def main():
    # default: cam 0 show 1
    cmd = ["python3", "webcam_onnx.py", "--source", "csi", "--cam", "0", "--show", "1"]
    raise SystemExit(subprocess.call(cmd))

if __name__ == "__main__":
    main()
