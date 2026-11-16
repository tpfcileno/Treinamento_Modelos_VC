# Treinamento de Modelos de Visão Computacional — Integra360

Repositório de treino de detectores de objetos para o Integra360. Cada modelo fica em
uma pasta própria e autocontida (Dockerfile, `docker-compose.yml`, dependências e código).

| Pasta | Modelo |
|---|---|
| [`YOLOX/`](YOLOX/) | YOLOX (S / M / L) — fork do [Megvii YOLOX 0.3.0](https://github.com/Megvii-BaseDetection/YOLOX) |

---

## YOLOX

### Pré-requisitos (máquina de treino Linux)

- GPU NVIDIA com driver instalado (`nvidia-smi` funcionando)
- Docker + Docker Compose v2
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

Teste rápido: `docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi`

### Estrutura

```
YOLOX/
├── Dockerfile, docker-compose.yml
├── requirements.txt            # dependências de treino (versões travadas p/ a imagem)
├── requirements-export.txt     # opcional: export ONNX
├── scripts/download_weights.sh # baixa os pesos pré-treinados no COCO
├── exps/example/custom/        # experimentos Integra: yolox_{s,m,l}_integra.py
├── tools/                      # train.py, eval.py, eval_save.py, demo/export
├── analises/                   # métricas por confiança, tabelas PNG e dashboard
├── yolox/                      # biblioteca YOLOX
├── tests/                      # executados durante o docker build
├── datasets/                   # (fora do git) dataset no formato COCO
├── weights/                    # (fora do git) pesos pré-treinados
└── YOLOX_outputs/              # saídas de treino/avaliação
```

`datasets/`, `weights/` e `YOLOX_outputs/` são montados como volume no container, ou
seja, tudo que o treino gera fica salvo no host.

### 1. Clonar e preparar o dataset

```bash
git clone <url-do-repo> && cd Treinamento_Modelos_VC/YOLOX
```

Copie o dataset (formato COCO) para `YOLOX/datasets/integra/`, com as anotações
renomeadas assim:

```
datasets/integra/
├── annotations/
│   ├── instances_train.json
│   ├── instances_val.json
│   └── instances_test.json   (opcional: usado com --test e nas análises)
├── train/   (imagens de treino)
├── val/     (imagens de validação)
└── test/    (imagens de teste)
```

O dataset não vai pelo git; transfira com `scp`/`rsync`, por exemplo:
`rsync -avP datasets/integra/ usuario@maquina:~/Treinamento_Modelos_VC/YOLOX/datasets/integra/`

### 2. Baixar os pesos pré-treinados

Pesos oficiais do YOLOX treinados no COCO, usados como ponto de partida do fine-tuning:

```bash
bash scripts/download_weights.sh        # baixa yolox_s, yolox_m e yolox_l para weights/
bash scripts/download_weights.sh l      # ou só os escolhidos (nano tiny s m l x)
```

### 3. Build da imagem

```bash
docker compose build yolox
```

O build roda os testes (`tests/`) no final; se algum falhar, a imagem não é gerada.
Por padrão o Docker esconde a saída da etapa que falhou; para ver o motivo completo:

```bash
docker compose build --progress plain yolox
```

> A imagem final tem cerca de 12 GB. Garanta espaço livre no disco do Docker antes do
> build (`docker system df`). Se o build cair com `failed to solve: ... EOF`, normalmente
> o disco do Docker encheu.

### 4. Treinar

**Direto pelo compose** — usa o comando definido no `docker-compose.yml`, configurável
por variáveis de ambiente (ou por um arquivo `.env` na pasta `YOLOX/`):

| Variável | Padrão |
|---|---|
| `EXP` | `exps/example/custom/yolox_m_integra.py` |
| `CKPT` | `weights/yolox_m.pth` |
| `BATCH` | `16` |
| `DEVICES` | `1` (nº de GPUs) |

```bash
docker compose up yolox                                   # YOLOX-M com os padrões

EXP=exps/example/custom/yolox_l_integra.py \
CKPT=weights/yolox_l.pth BATCH=8 \
docker compose up yolox                                   # YOLOX-L, batch 8
```

**Ou por shell interativo** (todos os comandos abaixo rodam em `/workspace/YOLOX`):

```bash
docker compose run --rm yolox bash

python3 tools/train.py \
  -f exps/example/custom/yolox_m_integra.py \
  -d 1 -b 16 --fp16 -o \
  -c weights/yolox_m.pth
```

> Flags: `-d` nº de GPUs, `-b` batch size total, `--fp16` precisão mista,
> `-o` reserva a memória da GPU antes, `-c` checkpoint inicial,
> `--cache` guarda as imagens em RAM, `--resume` retoma de `latest_ckpt.pth`.
> Se faltar memória de GPU, reduza `-b` (ex.: 8 para o L).

Saídas em `YOLOX_outputs/<nome_do_exp>/`: `best_ckpt.pth` (melhor AP50:95 de validação),
`last_epoch_ckpt.pth`, `epoch_N_ckpt.pth` (se `save_history_ckpt = True`),
`train_log.txt` e os logs do tensorboard.

Acompanhar pelo tensorboard (no host):
`tensorboard --logdir YOLOX/YOLOX_outputs`

#### Configuração dos experimentos

Os hiperparâmetros ficam em `exps/example/custom/yolox_{s,m,l}_integra.py`. Pontos
específicos deste projeto:

- **`num_classes = 14`**: precisa bater com as categorias do JSON. Se divergir, o treino
  falha logo no início com uma mensagem indicando as categorias encontradas.
- **Classe `ruido`** (`noise_class_name = "ruido"`): as caixas dessa classe **não** são
  alvo de treino nem GT de avaliação. No treino, a região é borrada (Gaussian blur) para o
  modelo não aprender com o conteúdo dela; na avaliação a imagem fica intacta. A classe
  continua existindo na saída do modelo (índices inalterados), mas nunca é aprendida.
  Use `noise_class_name = None` para desativar. Implementação em
  `yolox/data/datasets/coco_noise.py`.
- **Early stopping** (`early_stop_patience`): para o treino após N avaliações seguidas
  sem melhora do AP50:95 de validação (`0` desativa; `early_stop_min_delta` define o
  ganho mínimo).

### 5. Avaliar um checkpoint

```bash
# AP/AR COCO no split de validação
python3 tools/eval.py \
  -f exps/example/custom/yolox_m_integra.py \
  -c YOLOX_outputs/yolox_m_integra/best_ckpt.pth \
  -b 16 -d 1 --fp16

# no split de teste (instances_test.json), salvando as predições
python3 tools/eval_save.py \
  -f exps/example/custom/yolox_m_integra.py \
  -c YOLOX_outputs/yolox_m_integra/best_ckpt.pth \
  -b 16 -d 1 --fp16 --test
```

O `eval_save.py` grava `YOLOX_outputs/<exp>/predictions.json` e imprime AP/AR por classe.

### 6. Análises de métricas

A partir do `predictions.json`, rodando de dentro de `YOLOX/` (no container ou no host
com `numpy`, `Pillow` e `pycocotools`):

```bash
# precisão / recall / F1 / AP por classe e por limiar de confiança + matrizes de confusão
python3 analises/metrics_inferences.py \
  --predictions YOLOX_outputs/yolox_m_integra/predictions.json \
  --annotations datasets/integra/annotations/instances_test.json \
  --output-dir  YOLOX_outputs/yolox_m_integra/metrics

# converte os CSVs gerados em tabelas PNG
python3 analises/csv_tables_to_png.py --input-dir YOLOX_outputs/yolox_m_integra/metrics

# dashboard web local (http://127.0.0.1:8765)
python3 analises/dashboard/run.py --input-dir YOLOX_outputs/yolox_m_integra/metrics
```

> Essas análises usam o JSON de anotações original, então a classe `ruido` aparece
> com GT e sem predições — desconsidere essa linha.

### 7. Exportar para ONNX (opcional)

```bash
pip install -r requirements-export.txt
python3 tools/export_onnx.py \
  -f exps/example/custom/yolox_m_integra.py \
  -c YOLOX_outputs/yolox_m_integra/best_ckpt.pth \
  --output-name YOLOX_outputs/yolox_m_integra/yolox_m_integra.onnx
```

### Observações

- Arquivos criados pelo container nos volumes pertencem ao `root` no host Linux. Para
  recuperar a posse: `sudo chown -R $USER YOLOX/YOLOX_outputs`.
- `shm_size: 8gb` no compose é necessário para o DataLoader com vários workers; se o
  host tiver pouca RAM, reduza junto com `data_num_workers` no exp.
