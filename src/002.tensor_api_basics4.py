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
Tensor 基本操作4 理解 indexing，加减乘除和 broadcasting 运算 | PyTorch 深度学习实战
https://blog.csdn.net/samurais/article/details/145314174
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
    indexing
    '''
    print("*" * 8, " a")
    a = torch.randn(5,4,3)
    print(a)

    print("*" * 8, " b")
    b = a[1,]
    print(b)

    print("*" * 8, " c")
    c = a[1:]
    print(c)

    print("*" * 8, " d")
    d = a[1:, 1]
    print(d)

if __name__ == '__main__':
    main()
