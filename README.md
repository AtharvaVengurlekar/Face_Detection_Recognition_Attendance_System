# Face Detection & Recognition – Attendance System

A real-time face detection and recognition system designed for attendance tracking using **YOLO**, **ArcFace**, and **Streamlit**.  
Supports **Docker GPU deployment** for high-performance inference.

---

## Requirements
| Component | Version |
|----------|---------|
| Python | **3.10.12 / 3.10.13** |
| CUDA | Installed on system for GPU Docker usage |

---

## Run Application Directly (Without Docker)
Make sure dependencies are installed:
```bash
pip install -r requirements.txt
```

## Start the Streamlit application:
```bash
streamlit run attandance_app.py
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
