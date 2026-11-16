"""Garante que os arquivos de experimento Integra carregam e montam o modelo (roda em CPU)."""
import os
import unittest

import torch

from yolox.exp import check_exp_value, get_exp

EXPS_DIR = os.path.join(os.path.dirname(__file__), "..", "exps", "example", "custom")
INTEGRA_EXPS = ["yolox_s_integra.py", "yolox_m_integra.py", "yolox_l_integra.py"]


class TestIntegraExps(unittest.TestCase):
    def test_exps_load_and_build_model(self):
        for exp_file in INTEGRA_EXPS:
            with self.subTest(exp=exp_file):
                exp = get_exp(os.path.join(EXPS_DIR, exp_file))
                check_exp_value(exp)
                self.assertEqual(exp.num_classes, 14)
                self.assertEqual(exp.noise_class_name, "ruido")

                model = exp.get_model().eval()
                with torch.no_grad():
                    out = model(torch.zeros(1, 3, 64, 64))
                # saida: [batch, anchors, 4 bbox + 1 obj + num_classes]
                self.assertEqual(out.shape[-1], 5 + exp.num_classes)


if __name__ == "__main__":
    unittest.main()
