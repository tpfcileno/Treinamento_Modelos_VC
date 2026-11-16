# Variante "runtime" (bem menor que a "devel"): o YOLOX so compila uma extensao C++ pura
# (FastCOCOEval, CppExtension), que precisa do g++ (build-essential) e nao do nvcc/CUDA toolkit.
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libglib2.0-0 \
        ninja-build \
        wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/YOLOX

# Dependencias primeiro (camada cacheada enquanto requirements.txt nao mudar).
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY . ./
# --no-build-isolation: o setup.py importa torch para compilar a op C++ (FastCOCOEval);
# sem isso o pip cria um ambiente de build isolado sem torch e a compilacao falha.
RUN pip install -v -e . --no-build-isolation \
    && python -c "import yolox, cv2, numpy, pycocotools; print('yolox', yolox.__version__, '| numpy', numpy.__version__, '| cv2', cv2.__version__)"

# Falha rapido: se os testes quebrarem, a imagem nao e gerada (rodam em CPU, sem GPU no build).
# Para ver o log completo de uma falha: docker compose build --progress plain yolox
RUN python -m unittest discover -s tests -t . -v 2>&1 \
    || { echo ""; \
         echo "================================================================"; \
         echo " BUILD ABORTADO: testes falharam."; \
         echo " Veja acima as linhas 'FAIL:' / 'ERROR:' com o teste e o motivo."; \
         echo " Log completo: docker compose build --progress plain yolox"; \
         echo "================================================================"; \
         exit 1; }

# Dataset, pesos e saidas vem do host (ver docker-compose.yml).
VOLUME ["/workspace/YOLOX/datasets", "/workspace/YOLOX/weights", "/workspace/YOLOX/YOLOX_outputs"]

CMD ["/bin/bash"]
