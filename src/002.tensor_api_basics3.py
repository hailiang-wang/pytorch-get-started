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
Tensor 基本操作3 理解 shape, stride, storage, view，is_contiguous 和 reshape 操作 | PyTorch 深度学习实战
https://chatopera.blog.csdn.net/article/details/145305367
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
    stride, shape, storage
    '''
    a = [[2, 3], [4, 5], [6, 7]]
    a_t = torch.tensor(a)

    print("Tensor: ", a_t)
    print("is_contiguous: ", a_t.is_contiguous())

    
    print("Shape: ", a_t.shape)
    print("Stride: ", a_t.stride())
    print("Storage: ", a_t.storage())

   

    a_t.storage()[3] = 8
    print(a_t)

    print(a_t.shape)
    b_t = a_t.view(2, 3)
    print(b_t)
    print(b_t.shape)


if __name__ == '__main__':
    main()
