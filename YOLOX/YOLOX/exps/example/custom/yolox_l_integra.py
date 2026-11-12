#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()

        # ---------------- model config ---------------- #
        # YOLOX-L  (depth=1.0, width=1.0)
        self.depth = 1.0   # multiplicador de profundidade (nº de blocos/camadas do backbone+FPN); 1.0 = variante L
        self.width = 1.0   # multiplicador de largura (nº de canais por camada); 1.0 = variante L
        # nome do experimento: deriva do nome do arquivo (ex.: "yolox_l_integra"). Define a pasta de saída em YOLOX_outputs/
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # As categorias nos JSONs COCO sao: capacete, botas, pessoa,
        # torso_reflexivo, calca, luvas e manga_longa.
        self.num_classes = 14

        # ---------------- dataset config ---------------- #
        # Layout esperado dentro de data_dir:
        #   datasets/integra/
        #     annotations/
        #       instances_train.json
        #       instances_val.json
        #       instances_test.json
        #     train/   (imagens de treino)
        #     val/     (imagens de validacao)
        #     test/    (imagens de teste)
        self.data_dir = "datasets/integra"              # raiz do dataset (relativa ao cwd do treino, /workspace/YOLOX)
        self.train_ann = "instances_train.json"    # arquivo de anotações COCO de treino (dentro de data_dir/annotations/)
        self.val_ann = "instances_val.json"        # arquivo de anotações COCO de validação (dentro de data_dir/annotations/)
        self.test_ann = "instances_test.json"      # arquivo de anotações COCO de teste
        self.train_name = "train"
        self.val_name = "val"
        self.test_name = "test"

        # ---------------- training config ---------------- #
        self.max_epoch = 300          # nº total de épocas de treino
        self.data_num_workers = 4     # processos paralelos do DataLoader para carregar/aumentar imagens
        self.eval_interval = 1        # roda avaliação no conjunto de val a cada N épocas

        # early stopping: para o treino apos N avaliacoes sem melhora do AP50:95 de val.
        # Com eval_interval=1, N equivale a epocas. 0 desativa.
        self.early_stop_patience = 45
        self.early_stop_min_delta = 0.0  # ganho minimo de AP para contar como melhora

        self.multiscale_range = 5     # variação de escala multi-resolução: input_size +/- 5*32 px por iteração
        self.mosaic_prob = 1.0        # probabilidade de aplicar augmentation Mosaic (junta 4 imagens); 1.0 = sempre
        self.mixup_prob = 1.0         # probabilidade de aplicar MixUp (sobrepõe 2 imagens); 1.0 = sempre
        self.hsv_prob = 1.0           # probabilidade de aplicar jitter de cor HSV (matiz/saturação/brilho)
        self.flip_prob = 0.5          # probabilidade de espelhar a imagem horizontalmente
        self.degrees = 5           # rotação aleatória máxima (graus) na augmentation afim
        self.translate = 0.1          # translação aleatória máxima (fração da imagem) na augmentation afim
        self.mosaic_scale = (0.5, 1.5)  # faixa de zoom aplicada às imagens do Mosaic (min, max)
        self.enable_mixup = True      # liga/desliga o MixUp
        self.mixup_scale = (0.5, 1.5)   # faixa de escala usada no MixUp (min, max)
        self.shear = 2.0              # cisalhamento (shear) aleatório máximo (graus) na augmentation afim
        self.warmup_epochs = 5        # épocas iniciais de "aquecimento" com LR subindo gradualmente
        self.warmup_lr = 0            # learning rate inicial no começo do warmup
        self.min_lr_ratio = 0.05      # LR mínimo ao fim do cosine = basic_lr * min_lr_ratio
        self.basic_lr_per_img = 0.00015625  # LR base por imagem; LR efetivo = basic_lr_per_img * batch_size
        self.scheduler = "yoloxwarmcos"  # política de LR: warmup linear + decaimento cosseno
        self.no_aug_epochs = 15       # nº de épocas finais SEM augmentation pesada (Mosaic/MixUp desligados)
        self.ema = True               # usa Exponential Moving Average dos pesos (estabiliza/melhora a avaliação)
        self.weight_decay = 5e-4      # regularização L2 (weight decay) do otimizador
        self.momentum = 0.9           # momentum do otimizador SGD
        self.save_history_ckpt = True   # salva checkpoint de cada época (não só o último/melhor)
        self.test_conf = 0.01         # limiar de confiança mínimo para considerar uma detecção na avaliação
        self.nmsthre = 0.65           # limiar de IoU do NMS (suprime caixas sobrepostas acima desse valor)

        # input/test size (multiplos de 32)
        self.input_size = (640, 640)  # resolução (altura, largura) das imagens no treino
        self.test_size = (640, 640)   # resolução (altura, largura) das imagens na avaliação/inferência
