#!/usr/bin/env bash
# Baixa os pesos YOLOX pre-treinados no COCO (release oficial Megvii) para YOLOX/weights/.
#
# Uso (a partir da pasta YOLOX/):
#   bash scripts/download_weights.sh          # baixa s, m e l
#   bash scripts/download_weights.sh m l      # baixa so os escolhidos (nano tiny s m l x)
set -euo pipefail

BASE_URL="https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0"
WEIGHTS_DIR="$(cd "$(dirname "$0")/.." && pwd)/weights"
MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=(s m l)

mkdir -p "$WEIGHTS_DIR"

for m in "${MODELS[@]}"; do
    case "$m" in
        nano|tiny|s|m|l|x) ;;
        *) echo "Modelo invalido: '$m' (use nano, tiny, s, m, l ou x)" >&2; exit 1 ;;
    esac
    file="yolox_${m}.pth"
    dest="$WEIGHTS_DIR/$file"
    if [ -s "$dest" ]; then
        echo "[ok]  $file ja existe, pulando"
        continue
    fi
    echo "[get] $file"
    if command -v wget >/dev/null 2>&1; then
        wget -q --show-progress -O "$dest.part" "$BASE_URL/$file"
    else
        curl -fL --progress-bar -o "$dest.part" "$BASE_URL/$file"
    fi
    mv "$dest.part" "$dest"
done

echo "Pesos em: $WEIGHTS_DIR"
ls -lh "$WEIGHTS_DIR"
