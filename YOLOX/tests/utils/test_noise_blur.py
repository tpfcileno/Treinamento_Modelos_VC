import json
import os
import pickle
import tempfile
import unittest

import cv2
import numpy as np

from yolox.data.datasets.coco_noise import COCONoiseDataset, blur_regions


def _write_tiny_coco(root, categories):
    """2 imagens 32x32 em `root/train` + annotations/instances_train.json."""
    os.makedirs(os.path.join(root, "annotations"))
    os.makedirs(os.path.join(root, "train"))
    rng = np.random.RandomState(0)
    for name in ("a.png", "b.png"):
        cv2.imwrite(os.path.join(root, "train", name), rng.randint(0, 255, (32, 32, 3), np.uint8))

    coco = {
        "images": [
            {"id": 1, "file_name": "a.png", "width": 32, "height": 32},
            {"id": 2, "file_name": "b.png", "width": 32, "height": 32},
        ],
        "categories": categories,
        "annotations": [
            # img 1: uma pessoa + um ruido
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 14, "bbox": [16, 16, 12, 12], "area": 144, "iscrowd": 0},
            # img 2: so ruido
            {"id": 3, "image_id": 2, "category_id": 14, "bbox": [2, 2, 8, 8], "area": 64, "iscrowd": 0},
        ],
    }
    with open(os.path.join(root, "annotations", "instances_train.json"), "w") as f:
        json.dump(coco, f)


class TestCOCONoiseDataset(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def _dataset(self, categories=None, **kwargs):
        if categories is None:
            # Fora de ordem de id de proposito: `_classes` segue o JSON, `class_ids` e ordenado.
            categories = [{"id": 14, "name": "ruido"}, {"id": 1, "name": "pessoa"}]
        _write_tiny_coco(self.root, categories)
        return COCONoiseDataset(
            data_dir=self.root, json_file="instances_train.json", name="train",
            img_size=(32, 32), **kwargs,
        )

    def test_removes_noise_from_targets_and_coco_gt(self):
        ds = self._dataset()
        pessoa_idx = ds.class_ids.index(1)

        self.assertEqual(ds.class_ids, [1, 14])  # indices do modelo preservados
        self.assertEqual([a["category_id"] for a in ds.coco.dataset["annotations"]], [1])
        self.assertEqual(set(ds.coco.anns), {1})
        self.assertEqual(ds.annotations[0][0][:, 4].tolist(), [pessoa_idx])
        self.assertEqual(ds.annotations[1][0].shape, (0, 5))
        self.assertEqual(len(ds.noise_boxes[0]), 1)
        self.assertEqual(len(ds.noise_boxes[1]), 1)

    def test_blurs_only_noise_regions_in_train(self):
        ds = self._dataset(blur_noise=True)
        raw = ds.read_img(0)
        img, labels, _, _ = ds.pull_item(0)

        self.assertEqual(labels[:, 4].tolist(), [ds.class_ids.index(1)])
        self.assertFalse(np.array_equal(img[16:28, 16:28], raw[16:28, 16:28]))  # ruido borrado
        np.testing.assert_array_equal(img[0:10, 0:10], raw[0:10, 0:10])        # pessoa intacta

    def test_eval_mode_does_not_blur(self):
        ds = self._dataset(blur_noise=False)
        img, labels, _, _ = ds.pull_item(1)
        np.testing.assert_array_equal(img, ds.read_img(1))
        self.assertEqual(labels.shape, (0, 5))

    def test_missing_noise_class_is_noop(self):
        ds = self._dataset(categories=[{"id": 1, "name": "pessoa"}, {"id": 14, "name": "outra"}])
        self.assertEqual(len(ds.coco.dataset["annotations"]), 3)
        self.assertEqual(ds.annotations[0][0].shape, (2, 5))

    def test_is_picklable(self):
        # Necessario para --cache + multi-GPU (o Exp e serializado para os processos).
        pickle.dumps(self._dataset())

    def test_blur_regions_does_not_modify_input(self):
        img = np.random.RandomState(1).randint(0, 255, (20, 20, 3), np.uint8)
        original = img.copy()
        out = blur_regions(img, np.array([[0, 0, 10, 10]]))
        np.testing.assert_array_equal(img, original)
        self.assertFalse(np.array_equal(out, original))


if __name__ == "__main__":
    unittest.main()
