FROM nvcr.io/nvidia/pytorch:19.12-py3
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get -y install sudo dialog apt-utils && rm -rf /var/lib/apt/lists/*
USER root

# Install some basic utilities
RUN apt-get update && apt-get install -y \    
    curl \
    ca-certificates \
    sudo \
    unzip \
    htop \
    wget \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/
ENV HOME=/workspace/
RUN chmod 777 /workspace/

RUN pip install easydict
RUN pip install opencv-python

RUN pip install tensorboardX
CMD ["python3"]

WORKDIR /workspace/
USER root