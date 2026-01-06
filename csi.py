import os
import cv2
import torch
import time

# ---------- YOLOv5 torch.hub compat (path/weights) ----------
_real_hub_load = torch.hub.load

def hub_load_compat(repo_or_dir, model, *args, **kwargs):
    if repo_or_dir == "yolov5":
        repo_or_dir = "./yolov5"
    kwargs["source"] = "local"

    try:
        return _real_hub_load(repo_or_dir, model, *args, **kwargs)
    except TypeError as e:
        msg = str(e)
        # custom() không nhận path= -> đổi sang weights=
        if "unexpected keyword argument 'path'" in msg and "path" in kwargs:
            kwargs["weights"] = kwargs.pop("path")
            return _real_hub_load(repo_or_dir, model, *args, **kwargs)
        # custom() không nhận weights= -> đổi sang path=
        if "unexpected keyword argument 'weights'" in msg and "weights" in kwargs:
            kwargs["path"] = kwargs.pop("weights")
            return _real_hub_load(repo_or_dir, model, *args, **kwargs)
        raise

torch.hub.load = hub_load_compat
# -----------------------------------------------------------

def gstreamer_pipeline(sensor_id=0, capture_width=1280, capture_height=720,
                       display_width=1280, display_height=720,
                       framerate=30, flip_method=0):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=1"
    )

def load_models():
    lp_det = torch.hub.load("yolov5", "custom", path="model/LP_detector_nano_61.pt")
    lp_ocr = torch.hub.load("yolov5", "custom", path="model/LP_ocr_nano_62.pt")

    # tuỳ chỉnh nhẹ cho nhanh/ổn định
    lp_det.conf = 0.35
    lp_det.iou  = 0.45
    lp_det.max_det = 5

    lp_ocr.conf = 0.25
    lp_ocr.iou  = 0.45
    lp_ocr.max_det = 20

    return lp_det, lp_ocr

def main():
    cv2.setNumThreads(1)

    pipeline = gstreamer_pipeline(sensor_id=0, capture_width=1280, capture_height=720,
                                  display_width=1280, display_height=720,
                                  framerate=30, flip_method=0)
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("❌ Không mở được CSI camera. Check mount /tmp/argus_socket + --privileged + display.")
        print("Pipeline:", pipeline)
        return

    print("✅ CSI camera opened")

    lp_det, lp_ocr = load_models()
    t0 = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("⚠️ Frame None / read failed -> kiểm tra camera đang bị app khác chiếm (nvargus) hoặc pipeline.")
            break

        # YOLOv5 nhận BGR cũng được (nó tự xử), cứ để vậy
        det_rs = lp_det(frame, size=640)  # size nhỏ cho Nano đỡ đuối
        det_df = det_rs.pandas().xyxy[0]

        # vẽ bbox + OCR từng biển
        for _, r in det_df.iterrows():
            x1, y1, x2, y2 = map(int, [r.xmin, r.ymin, r.xmax, r.ymax])
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            ocr_rs = lp_ocr(crop, size=320)
            ocr_df = ocr_rs.pandas().xyxy[0]

            # ghép text đơn giản theo x (tuỳ dataset bạn mà tinh chỉnh)
            if len(ocr_df):
                ocr_df = ocr_df.sort_values("xmin")
                text = "".join([str(c) for c in ocr_df["name"].tolist()])
            else:
                text = "?"

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, text, (x1, max(0, y1-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        frame_count += 1
        if frame_count % 30 == 0:
            dt = time.time() - t0
            fps = frame_count / max(dt, 1e-6)
            print(f"FPS ~ {fps:.2f}")

        cv2.imshow("CSI ALPR", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
