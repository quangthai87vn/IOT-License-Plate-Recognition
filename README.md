# Vietnamese License Plate Recognition

This repository provides you with a detailed guide on how to training and build a Vietnamese License Plate detection and recognition system. This system can work on 2 types of license plate in Vietnam, 1 line plates and 2 lines plates.

## Installation

```bash
 
  # install dependencies using pip 
  pip install -r ./requirement.txt


  # Dùng môi trường Python 3.9 trên MacOS M2
  pip uninstall -y torch torchvision torchaudio
  # chọn 1 bộ version ổn (torch < 2.6)
  pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1


```

- **Pretrained model** provided in ./model folder in this repo 

- **Download yolov5 (old version) from this link:** [yolov5 - google drive](https://drive.google.com/file/d/1g1u7M4NmWDsMGOppHocgBKjbwtDA-uIu/view?usp=sharing)

- Copy yolov5 folder to project folder

## Run License Plate Recognition

```bash
  # run inference on webcam (15-20fps if there is 1 license plate in scene)
  python webcam.py 

  # run inference on image
  python lp_image.py -i test_image/3.jpg

  # run LP_recognition.ipynb if you want to know how model work in each step
```

## Result
![Demo 1](result/image.jpg)

![Vid](result/video_1.gif)

## Vietnamese Plate Dataset

This repo uses 2 sets of data for 2 stage of license plate recognition problem:

- [License Plate Detection Dataset](https://drive.google.com/file/d/1xchPXf7a1r466ngow_W_9bittRqQEf_T/view?usp=sharing)
- [Character Detection Dataset](https://drive.google.com/file/d/1bPux9J0e1mz-_Jssx4XX1-wPGamaS8mI/view?usp=sharing)

Thanks [Mì Ai](https://www.miai.vn/thu-vien-mi-ai/) and [winter2897](https://github.com/winter2897/Real-time-Auto-License-Plate-Recognition-with-Jetson-Nano/blob/main/doc/dataset.md) for sharing a part in this dataset.

## Training

**Training code for Yolov5:**

Use code in ./training folder
```bash
  training/Plate_detection.ipynb     #for LP_Detection
  training/Letter_detection.ipynb    #for Letter_detection
```






## Cài Docker (nếu máy chưa có)

```bash
#Jetson thường cài sẵn/hoặc cài nhanh:

sudo apt update
sudo apt install -y docker.io docker-compose
sudo usermod -aG docker $USER
newgrp docker
docker --version
docker compose version
docker build --no-cache -t iot-license-plate-recognition:jetson-lpr .
```

## Chạy Dự đoán

```bash
#(Nếu có HDMI) bật quyền hiển thị cửa sổ - Trên Jetson (ngoài docker):
xhost +local:docker
python3 rtsp.py "rtsp://192.168.50.2:8554/mac"


docker run --rm -it \
  --runtime nvidia --network host \
  -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /tmp/argus_socket:/tmp/argus_socket \
  -v "$PWD":/workspace -w /workspace \
  --device /dev/video0 \
  iot-license-plate-recognition:jetson-lpr \
  python3 csi.py




xhost +local:docker
docker run --rm -it \
  --runtime nvidia \
  --network host \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -e RTSP_URL="rtsp://192.168.50.2:8554/mac" \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$PWD":/workspace \
  -w /workspace \
  iot-license-plate-recognition:jetson-lpr \
  python3 rtsp.py

```