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
学习率调整策略 | PyTorch 深度学习实战
https://chatopera.blog.csdn.net/article/details/145488189
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
Learning rate scheduler
'''
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset # 构造数据集加载器
from torch.utils.data import random_split # 划分数据集

torch.manual_seed(777)# for reproducibility为了重复使用

############################
# 生成数据
############################

# 定义函数
def f(x,y):
    return x**2 + 2*y**2

# 定义初始值
num_samples = 1000 # 1000 个样本点
X = torch.rand(num_samples) # 均匀分布
Y = torch.rand(num_samples)
Z = f(X,Y) + 3 * torch.randn(num_samples)


# Concatenates a sequence of tensors along a new dimension.
# All tensors need to be of the same size.
# https://pytorch.org/docs/stable/generated/torch.stack.html
dataset = torch.stack([X,Y,Z], dim=1)
# print(dataset.shape) # torch.Size([1000, 3])

# split data, 按照 7:3 划分数据集
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset=dataset, lengths=[train_size, test_size])

# 将数据封装为数据加载器
# narrow 函数对数据进行切片操作，
# 
train_dataloader = DataLoader(TensorDataset(train_dataset.dataset.narrow(1,0,2), train_dataset.dataset.narrow(1,2,1)), batch_size=32, shuffle=False)
test_dataloader = DataLoader(TensorDataset(test_dataset.dataset.narrow(1,0,2), test_dataset.dataset.narrow(1,2,1)), batch_size=32, shuffle=False)

############################
# 模型定义
############################

# 定义一个简单的模型
class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 8)
        self.output = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        return self.output(x)


############################
# 模型训练
############################

# 超参数
num_epochs = 100
learning_rate = 0.1 # 学习率，调大一些更直观

# 定义损失函数
loss_fn = nn.MSELoss()

# 通过两次训练，对比有无调节器的效果
for with_scheduler in [False, True]:

    # 定义训练和测试误差数组
    train_losses = []
    test_losses = []

    # 初始化模型
    model = Model()

    # 定义优化器
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

    # 定义学习率调节器
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.99)

    # 迭代训练
    for epoch in range(num_epochs):
        # 设定模型工作在训练模式
        model.train()
        train_loss = 0

        # 遍历训练集
        for inputs, targets in train_dataloader:
            # 预测、损失函数、反向传播
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # 记录 loss
            train_loss += loss.item()

        # 计算 loss 并记录到训练误差
        train_loss /= len(train_dataloader)
        train_losses.append(train_loss)

        # 在测试数据集上评估
        model.eval()
        test_loss = 0

        with torch.no_grad():
            # 遍历测试集
            for inputs, targets in test_dataloader:
                # 预测、损失函数
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                # 记录 loss
                test_loss += loss.item()

            # 计算 loss 并记录到测试误差
            test_loss /= len(test_dataloader)
            test_losses.append(test_loss)

        
        # 是否更新学习率
        if with_scheduler:
            scheduler.step()

    # 绘制训练和测试误差曲线
    plt.figure(figsize= (8, 4))
    plt.plot(range(num_epochs), train_losses, label="Train")
    plt.plot(range(num_epochs), test_losses, label="Test")
    plt.title("{0} lr_scheduler".format("With " if with_scheduler else "Without"))
    plt.legend()
    # plt.ylim((1,2))
    plt.show()


# 常见学习率调节器

## 学习率衰减，例如每训练 100 次就将学习率降低为原来的一半
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=100, gamma=0.5)
## 指数衰减法，每次迭代将学习率乘上一个衰减率
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer,gamma=0.99)
## 余弦学习率调节，optimizer 初始学习率为最大学习率，eta_min 是最小学习率，T_max 是最大的迭代次数
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=100, eta_min=0.00001)
## 自定义学习率，通过一个 lambda 函数自定义实现学习率调节器
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.99 ** epoch)
## 预热
warmup_steps = 20
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda t: min(t/warmup_steps, 0.001))