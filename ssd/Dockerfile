FROM nvcr.io/nvidia/pytorch:19.12-py3
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

RUN apt-get update -y 

# # Set the default command to python3
CMD ["python3"]

# Create a working directory
RUN mkdir -p /workspace
RUN chmod 777 /workspace
ENV HOME=/workspace
WORKDIR /workspace

RUN pip install torch-summary

# # This is for marvinpytorch:1.8

