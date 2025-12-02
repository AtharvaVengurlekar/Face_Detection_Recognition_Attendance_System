# Face Detection & Recognition – Attendance System

A real-time face detection and recognition system designed for attendance tracking using **YOLO**, **ArcFace**, and **Streamlit**.  
Supports **Docker GPU deployment** for high-performance inference.

🔹 YOLO (You Only Look Once)

A fast, single-shot object detection model that processes the entire image in one pass.
Used here to detect faces in real time, providing high FPS and accurate bounding boxes.

🔹 ArcFace

A state-of-the-art face recognition model that learns highly discriminative facial embeddings using additive angular margin loss.
Used to match detected faces with saved identities for accurate attendance marking.

This project includes two face-recognition models that I trained on a custom dataset of 48,000 images (16k + 32k) to achieve high-accuracy embeddings for the attendance system.

1. trained_on_16k+32k_images.pt (PyTorch Model)
A) A full-precision model trained using the complete dataset.
B) Best used for further training, fine-tuning, or standard inference
C) Provides highest accuracy
D) Suitable for research, experimentation, or re-training

2. trained_on_16k+32k_images_int8.engine (TensorRT INT8 Model)
A) An INT8-quantized TensorRT engine built from the trained PyTorch model.
B) Designed for real-time GPU deployment
C) 4×–6× faster than FP32 with much lower memory usage
D) Small accuracy drop but ideal for production, Docker GPU containers, and edge devices

---

## Requirements
| Component | Version |
|----------|---------|
| Python | **3.10.12 / 3.10.13** |
| CUDA | Installed on system for GPU Docker usage |

---

## Create Environment 
Using Conda (Recommended)
```bash
conda create -n face_env python=3.10.13
conda activate face_env
```

## Run Application Directly (Without Docker)
Make sure dependencies are installed:
```bash
pip install -r requirements.txt
```

## Start the Streamlit application:
```bash
streamlit run app.py
```

## Run Using Docker (GPU Deployment)
Build Docker Image
```bash
sudo docker build -t face_dr_gpu_py310 .
```
Run Docker Container
```bash
docker run -it --gpus all --shm-size=10G -p 8501:8501 --name fdfr_streamlit_gpu face_dr_gpu_py310
```

## Alternative Execution (Without Streamlit UI)

Run using Python directly:
```bash
python face_detection_recognitaion_rtsp.py
```
