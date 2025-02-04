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
使用线性回归模型逼近目标模型 | PyTorch 深度学习实战
https://blog.csdn.net/samurais/article/details/145436751

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
torch.manual_seed(777)


'''
Linear Regression
'''
import torch
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(777)# for reproducibility为了重复使用

# X and Y data
x_data = [[65., 80., 75.],
          [89., 88., 93.],
          [80., 91., 90.],
          [30., 98., 100.],
          [50., 66., 70.]]
y_data = [[152.],
          [185.],
          [189.],
          [196.],
          [142.]]

x=torch.autograd.Variable(torch.Tensor(x_data))
y=torch.autograd.Variable(torch.Tensor(y_data))

# Our hypothesis XW+b
model=torch.nn.Linear(3,1,bias=True)

# cost criterion
criterion=torch.nn.MSELoss()

# Minimize
optimizer=torch.optim.SGD(model.parameters(),lr=1e-7)

# cost_h=[]
epochs=200
cost_h=np.zeros(epochs)

# Train the model
for step in range(epochs):
    optimizer.zero_grad()
    hypothesis=model(x) # Our hypothesis
    cost=criterion(hypothesis,y)
    cost.backward()
    optimizer.step()
    # cost_h.append(cost.data.numpy())
    cost_h[step]=cost.data.numpy()
    print(step,'Loss:',cost.data.numpy(),'\nPredict:\n',hypothesis.data.numpy())


for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)

plt.plot(cost_h)
plt.show()