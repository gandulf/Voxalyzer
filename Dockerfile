# 1. Use NVIDIA CUDA as the base
FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04

# 2. Set environment variables to avoid prompts during install
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 3. Install System Dependencies Python,Pip
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*


# 4. Install project and dependencies
RUN pip3 install --no-cache-dir --break-system-packages numpy mutagen fastapi uvicorn python-multipart onnx onnxruntime-gpu numba librosa pydub

# Copy project files
COPY models/*.onnx ./models/
COPY *.py ./

# Expose the port the app runs on
EXPOSE 8000

# Run the server
ENTRYPOINT ["python3", "voxalyzer.py"]
