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
Tensor 基本操作5 device 管理，使用 GPU 设备 | PyTorch 深度学习实战
https://chatopera.blog.csdn.net/article/details/145314362
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
    device
    '''
    
    #device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    device_gpu = torch.device('cuda')
    points_gpu = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]], device=device_gpu)
    print(points_gpu)

    print("*" * 8)
    device_cpu = torch.device('cpu')
    points_cpu = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]], device=device_cpu)
    print(points_cpu)
   
    print("*" * 8)
    device = torch.get_default_device() # cpu
    print(device)
    points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
    print(points)
   
    print("*" * 8)
    torch.set_default_device(device_gpu)  # https://pytorch.org/docs/stable/generated/torch.set_default_device.html
    points_default = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
    print(points_default)

    points2 = points.to(device_gpu)
    print(points2)

    point3 = points2 + points


if __name__ == '__main__':
    main()
