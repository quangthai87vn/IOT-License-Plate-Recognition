
FROM dustynv/jetson-inference:r32.7.1

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace


RUN apt-get update && apt-get install -y --no-install-recommends \
    git nano curl ca-certificates \
    python3-opencv \
    python3-pip python3-dev \
    build-essential cmake \
    protobuf-compiler libprotobuf-dev \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    x11-apps \
 && rm -rf /var/lib/apt/lists/*



RUN python3 -m pip install --upgrade pip setuptools wheel

# Cài requirements riêng cho repo trungdinh22 (KHÔNG cài lại torch/torchvision/opencv-python)
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

CMD ["bash"]
