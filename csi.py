#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from webcam_onnx import main

if __name__ == "__main__":
    raise SystemExit(main(["--source", "csi", "--cam", "0", "--show", "1"]))
