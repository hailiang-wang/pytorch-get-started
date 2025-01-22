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
理解 unsqueeze, squeeze, softmax
https://chatopera.blog.csdn.net/article/details/145244874
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
    '''
    unsqueeze
    '''
    data = [[1, 2],[3, 4]]
    x_data = torch.tensor(data)
    print("x_data")
    print(x_data)

    x2_data = x_data.unsqueeze(-1)
    print("x_data>> unsqueeze -1")
    print(x2_data)

    x2_data = x_data.unsqueeze(0)
    print("x_data>> unsqueeze 0")
    print(x2_data)

    x2_data = x_data.unsqueeze(1)
    print("x_data>> unsqueeze 1")
    print(x2_data)

    x2_data = x_data.unsqueeze(2)
    print("x_data>> unsqueeze 2")
    print(x2_data)


    '''
    squeeze
    '''
    data = [[1], [2],[3], [4]]
    x_data = torch.tensor(data)
    print("x_data")
    print(x_data)

    x2_data = x_data.squeeze()
    print("x_data>> squeeze")
    print(x2_data)

    x2_data = x_data.squeeze(1)
    print("x_data>> squeeze 1")
    print(x2_data)

    '''
    softmax
    '''
    data = torch.tensor([1,2,3], dtype=torch.float)
    x_data = torch.softmax(data, 0)
    print("x_data")
    print(x_data)

    data = torch.tensor([[1],[2],[3]], dtype=torch.float)
    x_data2 = torch.softmax(data, 1)
    print("x_data2")
    print(x_data2)


if __name__ == '__main__':
    main()
