#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-csi}"          # csi | rtsp
RTSP_URL_ARG="${2:-}"     # nếu mode=rtsp có thể truyền ở đây

# ====== Config ======
export IMG_SIZE="${IMG_SIZE:-640}"
export CONF="${CONF:-0.25}"
export IOU="${IOU:-0.45}"
export SHOW="${SHOW:-1}"
export SKIP="${SKIP:-0}"
export CODEC="${CODEC:-h264}"
export RTSP_LATENCY="${RTSP_LATENCY:-200}"

# RTSP URL ưu tiên: tham số > env
if [[ "$MODE" == "rtsp" ]]; then
  export RTSP_URL="${RTSP_URL_ARG:-${RTSP_URL:-}}"
  if [[ -z "${RTSP_URL}" ]]; then
    echo "[ERR] RTSP_URL is empty. Example:"
    echo "  ./run_alpr.sh rtsp rtsp://192.168.50.2:8554/mac"
    exit 1
  fi
fi

echo "[INFO] MODE=$MODE IMG_SIZE=$IMG_SIZE CONF=$CONF IOU=$IOU SHOW=$SHOW SKIP=$SKIP"

# ====== Clear camera processes ======
sudo pkill -f gst-launch || true
sudo systemctl restart nvargus-daemon || true

# ====== X11 for docker GUI ======
export DISPLAY="${DISPLAY:-:0}"
xhost +local:docker >/dev/null 2>&1 || true

# ====== Run container and execute ======
docker run --rm -it \
  --runtime nvidia \
  --network host \
  --ipc=host \
  --privileged \
  -e DISPLAY="$DISPLAY" \
  -e QT_X11_NO_MITSHM=1 \
  -e IMG_SIZE -e CONF -e IOU -e SHOW -e SKIP \
  -e CODEC -e RTSP_LATENCY \
  ${RTSP_URL:+-e RTSP_URL="$RTSP_URL"} \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /tmp/argus_socket:/tmp/argus_socket \
  -v /dev:/dev \
  -v "$PWD":/workspace \
  -w /workspace \
  iot-license-plate-recognition:jetson-lpr \
  bash -lc "python3 ${MODE}.py"
