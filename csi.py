#!/usr/bin/env python3
import os
import sys
import runpy


# CSI wrapper
os.environ["ALPR_SOURCE"] = "csi"
runpy.run_path("webcam_onnx.py", run_name="__main__")
