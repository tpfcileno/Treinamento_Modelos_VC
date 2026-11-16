#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
COCODataset com tratamento de uma classe de "ruido" (Integra360).

As caixas anotadas com a classe de ruido NAO viram alvo de treino nem GT de avaliacao:
  - sao removidas dos labels e do COCO em memoria (o evaluator usa dataset.coco como GT);
  - opcionalmente (treino), a regiao da caixa e borrada com Gaussian blur, para o modelo
    nao aprender a partir do conteudo dessas regioes.

A categoria continua em `cats`/`class_ids`, entao os indices de classe do modelo nao mudam.
Implementado como subclasse (e nao monkeypatch) para continuar serializavel com
`--cache` + multi-GPU e para valer igualmente em train.py, eval.py e eval_save.py.
"""

import cv2
import numpy as np
from loguru import logger

from .coco import COCODataset

__all__ = ["COCONoiseDataset", "blur_regions"]


def blur_regions(img, boxes, ksize=(31, 31), sigma=0):
    """
    Aplica Gaussian blur nas regioes `boxes` (Nx4: x1, y1, x2, y2) e retorna uma copia.
    Nunca altera `img` in-place (importante com --cache, que compartilha o array em memoria).
    """
    out = img.copy()
    if boxes is None or len(boxes) == 0:
        return out

    h, w = out.shape[:2]
    for x1, y1, x2, y2 in np.asarray(boxes)[:, :4]:
        x1 = max(int(np.floor(x1)), 0)
        y1 = max(int(np.floor(y1)), 0)
        x2 = min(int(np.ceil(x2)), w)
        y2 = min(int(np.ceil(y2)), h)
        if x2 <= x1 or y2 <= y1:
            continue
        out[y1:y2, x1:x2] = cv2.GaussianBlur(out[y1:y2, x1:x2], ksize, sigma)
    return out


class COCONoiseDataset(COCODataset):
    def __init__(
        self,
        *args,
        noise_class_name="ruido",
        blur_noise=True,
        blur_ksize=(31, 31),
        blur_sigma=0,
        **kwargs,
    ):
        self.noise_class_name = noise_class_name
        self.blur_noise = blur_noise
        self.blur_ksize = tuple(blur_ksize)
        self.blur_sigma = blur_sigma
        super().__init__(*args, **kwargs)
        self.noise_boxes = self._remove_noise_annotations()

    def _remove_noise_annotations(self):
        empty = [np.zeros((0, 4)) for _ in self.annotations]

        # Busca pelo category_id (nao pela posicao em `_classes`, que segue a ordem do JSON
        # e pode divergir de `class_ids`, que e ordenado e define o indice do label).
        noise_cat_ids = [c["id"] for c in self.cats if c["name"] == self.noise_class_name]
        if not noise_cat_ids:
            logger.warning(
                f"Classe de ruido '{self.noise_class_name}' nao existe em {self.json_file}; "
                "nenhuma anotacao sera removida/borrada."
            )
            return empty
        noise_cat_id = noise_cat_ids[0]
        noise_cls = self.class_ids.index(noise_cat_id)

        noise_boxes, filtered = [], []
        for labels, img_info, resized_info, file_name in self.annotations:
            is_noise = labels[:, 4].astype(int) == noise_cls
            noise_boxes.append(labels[is_noise, :4].copy())
            filtered.append((labels[~is_noise].copy(), img_info, resized_info, file_name))
        self.annotations = filtered

        coco_dataset = self.coco.dataset
        before = len(coco_dataset["annotations"])
        coco_dataset["annotations"] = [
            ann for ann in coco_dataset["annotations"] if ann["category_id"] != noise_cat_id
        ]
        self.coco.createIndex()

        logger.info(
            f"{self.json_file}: {before - len(coco_dataset['annotations'])} anotacoes de "
            f"'{self.noise_class_name}' removidas dos alvos/GT "
            f"(blur={'on' if self.blur_noise else 'off'})."
        )
        return noise_boxes

    def pull_item(self, index):
        img, label, img_info, img_id = super().pull_item(index)
        if self.blur_noise and len(self.noise_boxes[index]):
            img = blur_regions(img, self.noise_boxes[index], self.blur_ksize, self.blur_sigma)
        return img, label, img_info, img_id
