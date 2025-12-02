# Use NVIDIA PyTorch image (Python 3.10 + CUDA 12.x)
FROM nvcr.io/nvidia/pytorch:24.10-py3

# Install system deps
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libglvnd0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . /workspace/

# Install Python packages
RUN pip install --upgrade pip setuptools wheel

# Install your requirements (now it will install correct OpenCV)
RUN pip install -r requirements.txt

# Expose Streamlit port
EXPOSE 8502

CMD ["streamlit", "run", "attandance_app.py", "--server.port=8502", "--server.address=0.0.0.0"]