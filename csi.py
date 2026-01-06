import os
import runpy

# Force CSI mode
os.environ["SRC"] = "csi"

# Optional defaults (you can override when running)
# os.environ["CSI_FPS"] = "30"
# os.environ["CSI_MODE"] = "3"
# os.environ["OUT_W"] = "1280"
# os.environ["OUT_H"] = "720"
# os.environ["SHOW"] = "1"

runpy.run_path("webcam_onnx.py", run_name="__main__")
