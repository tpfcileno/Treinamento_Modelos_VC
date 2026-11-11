import unittest
from unittest.mock import patch

import numpy as np

from tools.utils.noise_blur import enable_noise_blur


class FakeCOCO:
    def __init__(self, annotations):
        self.dataset = {"annotations": annotations}
        self.createIndex()

    def createIndex(self):
        self.anns = {annotation["id"]: annotation for annotation in self.dataset["annotations"]}


class FakeDataset:
    def __init__(self):
        annotations = [
            {"id": 1, "category_id": 1},
            {"id": 2, "category_id": 14},
            {"id": 3, "category_id": 14},
        ]
        self.coco = FakeCOCO(annotations)
        self._classes = ("pessoa", "ruido")
        self.class_ids = [1, 14]
        self.annotations = [
            (
                np.array([[0, 0, 8, 8, 0], [8, 8, 16, 16, 1]], dtype=float),
                (16, 16),
                (16, 16),
                "first.jpg",
            ),
            (
                np.array([[1, 1, 10, 10, 1]], dtype=float),
                (16, 16),
                (16, 16),
                "second.jpg",
            ),
        ]

    def pull_item(self, index):
        labels, img_info, _, _ = self.annotations[index]
        return np.ones((16, 16, 3), dtype=np.uint8), labels.copy(), img_info, index


class TestNoiseBlur(unittest.TestCase):
    def test_removes_noise_annotations_but_blurs_saved_regions(self):
        dataset = FakeDataset()
        enable_noise_blur(type("Exp", (), {"dataset": dataset})())

        self.assertEqual(
            [annotation["category_id"] for annotation in dataset.coco.dataset["annotations"]],
            [1],
        )
        self.assertEqual(set(dataset.coco.anns), {1})
        self.assertEqual(dataset.class_ids, [1, 14])
        self.assertEqual(dataset.annotations[0][0][:, 4].tolist(), [0])
        self.assertEqual(dataset.annotations[1][0].shape, (0, 5))

        with patch(
            "tools.utils.noise_blur.cv2.GaussianBlur",
            side_effect=lambda region, _, __: region.copy(),
        ) as gaussian_blur:
            _, first_labels, _, _ = dataset.pull_item(0)
            _, second_labels, _, _ = dataset.pull_item(1)

        self.assertEqual(first_labels[:, 4].tolist(), [0])
        self.assertEqual(second_labels.shape, (0, 5))
        self.assertEqual(gaussian_blur.call_count, 2)

    def test_can_blur_without_removing_noise_labels(self):
        dataset = FakeDataset()
        enable_noise_blur(
            type("Exp", (), {"dataset": dataset})(),
            remove_noise_boxes=False,
        )

        with patch(
            "tools.utils.noise_blur.cv2.GaussianBlur",
            side_effect=lambda region, _, __: region.copy(),
        ):
            _, labels, _, _ = dataset.pull_item(0)

        self.assertIn(1, labels[:, 4])
        self.assertEqual(len(dataset.coco.dataset["annotations"]), 3)


if __name__ == "__main__":
    unittest.main()
