import cv2
import numpy as np

__all__ = ["blur_noise_regions", "enable_noise_blur"]


def blur_noise_regions(img, labels, noise_class_id, ksize=(31, 31), sigma=0,
                       remove_noise_boxes=True):
    """
    Aplica Gaussian blur nas regiões anotadas com a classe de ruído.

    img:    np.ndarray HxWxC
    labels: np.ndarray Nx5 -> [x1, y1, x2, y2, class_id]
    Retorna (img_borrada, labels). Não altera a imagem original (importante
    com --cache, que compartilha o array em memória).
    """
    out = img.copy()
    if labels is None or len(labels) == 0:
        return out, labels

    h, w = out.shape[:2]
    is_noise = labels[:, 4].astype(int) == noise_class_id

    for x1, y1, x2, y2 in labels[is_noise, :4]:
        x1 = max(int(np.floor(x1)), 0)
        y1 = max(int(np.floor(y1)), 0)
        x2 = min(int(np.ceil(x2)), w)
        y2 = min(int(np.ceil(y2)), h)
        if x2 <= x1 or y2 <= y1:
            continue
        out[y1:y2, x1:x2] = cv2.GaussianBlur(out[y1:y2, x1:x2], ksize, sigma)

    if remove_noise_boxes:
        labels = labels[~is_noise]

    return out, labels


def enable_noise_blur(exp, noise_class_name="ruido", ksize=(31, 31), sigma=0,
                      remove_noise_boxes=True):
    """
    Faz o dataset de treino aplicar o blur em todo pull_item.
    Funciona tanto se exp.dataset já existir (ex.: --cache) quanto se ele
    for criado depois, dentro de get_data_loader.
    """

    def _patch(dataset):
        noise_id = list(dataset._classes).index(noise_class_name)
        original_pull_item = dataset.pull_item

        def pull_item(index):
            img, label, img_info, img_id = original_pull_item(index)
            img, label = blur_noise_regions(
                img, label, noise_id, ksize, sigma, remove_noise_boxes
            )
            return img, label, img_info, img_id

        dataset.pull_item = pull_item
        return dataset

    if getattr(exp, "dataset", None) is not None:
        _patch(exp.dataset)
    else:
        original_get_dataset = exp.get_dataset

        def get_dataset(*args, **kwargs):
            return _patch(original_get_dataset(*args, **kwargs))

        exp.get_dataset = get_dataset