#!/usr/bin/env python
# -*- coding: utf-8 -*-
#===============================================================================
#
# Copyright (c) 2025 <> All Rights Reserved
#
#
# File: /c/Users/Administrator/courses/DeepLearning/PyTorch/pytorch_get_started/src/001.using_the_multipretrained_weight_api.py
# Author: Hai Liang Wang
# Date: 2025-01-19:16:00:14
#
#===============================================================================

"""
Sample code from
https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/
"""
__copyright__ = "Copyright (c) 2020 . All Rights Reserved"
__author__ = "Hai Liang Wang"
__date__ = "2025-01-19:16:00:14"

import os, sys
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curdir)

if sys.version_info[0] < 3:
    raise RuntimeError("Must be using Python 3")
else:
    unicode = str

# Get ENV
from env import ENV
from common.utils import console_log

# ROOT_DIR = os.path.join(curdir, os.pardir)
# DATA_ROOT_DIR = ENV.str("DATA_ROOT_DIR", None)

from PIL import Image
import torchvision as P

def main():
    img = Image.open(os.path.join(curdir, os.pardir, "assets/encode_jpeg/grace_hopper_517x606.jpg"))

    # Initialize model
    weights = P.models.ResNet50_Weights.IMAGENET1K_V2
    model = P.models.resnet50(weights=weights)
    model.eval()

    # Initialize inference transforms
    preprocess = weights.transforms()

    # Apply inference preprocessing transforms
    batch = preprocess(img).unsqueeze(0)
    prediction = model(batch).squeeze(0).softmax(0)

    # Make predictions
    label = prediction.argmax().item()
    score = prediction[label].item()

    # Use meta to get the labels
    category_name = weights.meta['categories'][label]
    print(f"{category_name}: {100 * score}%")

if __name__ == '__main__':
    main()
