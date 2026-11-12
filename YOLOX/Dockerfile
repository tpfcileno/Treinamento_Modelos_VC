
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6 8.9+PTX"

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        ninja-build \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/YOLOX

COPY YOLOX/requirements.txt ./requirements.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY YOLOX/ ./
# Fail the image build if the YOLOX-L fine-tuning weights were not included.
RUN test -s weights/yolox_l.pth
# --no-build-isolation: o setup.py importa torch para compilar a op CUDA (FastCOCOEval);
# sem isso o pip cria um ambiente de build isolado sem torch e a compilacao falha.
RUN pip install -v -e . --no-build-isolation

VOLUME ["/workspace/YOLOX/datasets", "/workspace/YOLOX/YOLOX_outputs"]

CMD ["/bin/bash"]
