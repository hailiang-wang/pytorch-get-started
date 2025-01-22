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
Tensor 基本操作2 理解 tensor.max 操作，沿着给定的 dim 是什么意思 | PyTorch 深度学习实战
https://chatopera.blog.csdn.net/article/details/145297647
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

import torch
from PIL import Image

def main():
    torch.manual_seed(777)

    '''
    max
    '''
    a = torch.randn(1, 3)
    print(a)
    b = torch.max(a)
    print(b)

    print("*"*8)
    a = torch.randn(4, 3, 2, 5)
    print(a)
    max, max_indices = torch.max(a, 1)
    print(max)
    print(max_indices)


if __name__ == '__main__':
    main()
