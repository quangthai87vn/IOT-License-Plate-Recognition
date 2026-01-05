
import os, runpy
import torch

# --- Patch torch.hub.load để tương thích nhiều bản YOLOv5 (path/weights) ---
_real_hub_load = torch.hub.load

def hub_load_compat(repo_or_dir, model, *args, **kwargs):
    # Ép dùng yolov5 local để khỏi lấy nhầm torch hub cache
    if repo_or_dir == "yolov5":
        repo_or_dir = "./yolov5"
    kwargs["source"] = "local"

    try:
        return _real_hub_load(repo_or_dir, model, *args, **kwargs)
    except TypeError as e:
        msg = str(e)

        # Case 1: custom() không nhận path= -> đổi sang weights=
        if "unexpected keyword argument 'path'" in msg and "path" in kwargs:
            kwargs["weights"] = kwargs.pop("path")
            return _real_hub_load(repo_or_dir, model, *args, **kwargs)

        # Case 2: custom() không nhận weights= -> đổi sang path=
        if "unexpected keyword argument 'weights'" in msg and "weights" in kwargs:
            kwargs["path"] = kwargs.pop("weights")
            return _real_hub_load(repo_or_dir, model, *args, **kwargs)

        raise

torch.hub.load = hub_load_compat
# --- end patch ---

runpy.run_path("webcam.py", run_name="__main__")