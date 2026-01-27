# 1. Use NVIDIA CUDA as the base
FROM nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04

# 2. Set environment variables to avoid prompts during install
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 3. Install System Dependencies for Essentia & Audio
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    ffmpeg \
    libyaml-dev \
    libfftw3-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libsamplerate0-dev \
    libtag1-dev \
    libchromaprint-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 6. Install Essentia
# Use essentia-tensorflow if you want access to the pre-trained models
RUN pip3 install essentia-tensorflow

RUN ln -s /usr/local/lib/python3.8/dist-packages/essentia/_essentia.cpython-310-x86_64-linux-gnu.so /usr/local/lib/python3.8/dist-packages/essentia/_essentia.so

RUN ldconfig

RUN pip3 install fastapi uvicorn python-multipart mutagen numpy

# Copy project files
COPY models/ ./models/
COPY main.py mp3.py server.py ./

# Expose the port the app runs on
EXPOSE 8000

# Run the server
ENTRYPOINT ["python3", "main.py"]
