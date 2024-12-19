FROM nvidia/cuda:12.0.0-base-ubuntu22.04

ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV PYTHONPATH="${PYTHONPATH}:/clevrskills"

RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt install --yes --no-install-recommends \
        git \
        less \
        python-is-python3 \
        python3 \
        python3-pip \
        python3-dev \
        vim \
        nano \
        curl \
        build-essential \
        libvulkan1 \
        libegl1 \
        libsm6 \
        libxext6 \
        libjpeg-dev \
        libpng-dev  \
        libxrender-dev \
        libglib2.0-0 \
        libglvnd-dev \
        x11vnc \
        xvfb \
        && \
    rm -rf /var/lib/apt/lists/*
 
RUN curl https://bootstrap.pypa.io/get-pip.py | python3

RUN git clone https://github.com/haosulab/ManiSkill.git /ManiSkill2 \
    && cd /ManiSkill2 \
    && git checkout v0.5.3 \
    && pip install -r requirements.txt \
    && pip install -e . \
    && rm -rf /ManiSkill2/.git

RUN git config --global --add safe.directory /ManiSkill2

COPY docker/requirements.txt requirements.txt

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir --upgrade --requirement requirements.txt

# https://github.com/haosulab/ManiSkill/issues/9
COPY docker/deps/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY docker/deps/nvidia_layers.json /etc/vulkan/implicit_layer.d/nvidia_layers.json
COPY docker/deps/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

COPY . /clevrskills/