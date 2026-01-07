#!/usr/bin/env bash
set -e

# Usage:
#   ./run_alpr.sh csi
#   ./run_alpr.sh rtsp rtsp://192.168.50.2:8554/mac
#   ./run_alpr.sh webcam 0
# Extra args:
#   ./run_alpr.sh csi --show --csi_w 1280 --csi_h 720 --csi_fps 60

MODE="${1:-csi}"
shift || true

pkill -f gst-launch || true
sudo systemctl restart nvargus-daemon || true

export DISPLAY=${DISPLAY:-:0}
xhost +local:root >/dev/null 2>&1 || true

IMG="iot-license-plate-recognition:jetson-lpr"
PROJ_DIR="$(pwd)"

PY_ARGS=("--source" "$MODE")
if [[ "$MODE" == "rtsp" ]]; then
  if [[ "${1:-}" == rtsp://* ]]; then
    PY_ARGS+=("--rtsp" "$1")
    shift
  fi
elif [[ "$MODE" == "webcam" ]]; then
  if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
    PY_ARGS+=("--cam" "$1")
    shift
  fi
fi

if [[ " $* " != *" --show "* ]]; then
  PY_ARGS+=("--show")
fi

PY_ARGS+=("$@")

exec docker run --rm -it \
  --runtime nvidia \
  --network host \
  --ipc=host \
  --privileged \
  -e DISPLAY="$DISPLAY" \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /tmp/argus_socket:/tmp/argus_socket \
  -v /dev:/dev \
  -v "$PROJ_DIR":/workspace \
  -w /workspace \
  "$IMG" \
  python3 webcam_onnx.py "${PY_ARGS[@]}"
