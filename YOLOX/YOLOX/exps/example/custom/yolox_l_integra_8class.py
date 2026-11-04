#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super().__init__()
        self.depth = 1.0
        self.width = 1.0
        self.num_classes = 8
        self.exp_name = "yolox_l_integra_8class_inference"
        self.test_size = (640, 640)
        self.nmsthre = 0.50
