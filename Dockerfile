# AD-HyFormer Docker image
# Base: official PyTorch image with CUDA 12.1 + cuDNN pre-installed,
# matched to typical RTX 30-series driver support.
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# ---- System dependencies ----
# SimpleITK/OpenCV need these system libs to build/run correctly in a
# slim base image (libGL for cv2, libgomp for SimpleITK's OpenMP backend).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# ---- Python dependencies ----
# Copied and installed separately from the rest of the code so Docker
# caches this layer and doesn't reinstall ~2GB of packages every time
# you edit a .py file.
COPY requirements.txt .
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# ---- Project code ----
# In development this gets overridden by the docker-compose volume
# mount; this COPY just makes the image self-contained for deployment.
COPY . .

# Streamlit dashboard port + TensorBoard port
EXPOSE 8501 6006

CMD ["bash"]
