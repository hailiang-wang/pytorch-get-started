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
CNN 卷积神经网络处理图片任务 | PyTorch 深度学习实战
https://blog.csdn.net/samurais/article/details/145493782
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

'''
CNN Model
'''
import torch
import torchvision.datasets as ds
import torchvision.transforms as ts
from torch.utils.data import DataLoader
from torch.autograd import Variable
import random

torch.manual_seed(777)# reproducibility

# parameters
batch_size=100
learning_rate=0.001
epochs=2

# MNIST dataset
ds_train=ds.MNIST(root='../../../DATA/MNIST_data',
                  train=True,
                  transform=ts.ToTensor(),
                  download=True)
ds_test=ds.MNIST(root='../../../DATA/MNIST_data',
                 train=False,
                 transform=ts.ToTensor(),
                 download=True)
# dataset loader
dl=DataLoader(dataset=ds_train,batch_size=batch_size,shuffle=True)

# CNN Model (2 conv layers)
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        # L1 ImgIn shape=(?, 28, 28, 1)
        #    Conv     -> (?, 28, 28, 32)
        #    Pool     -> (?, 14, 14, 32)
        self.layer1=torch.nn.Sequential(
            torch.nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1),#padding=1进行0填充
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2,stride=2)
        )
        # L2 ImgIn shape=(?, 14, 14, 32)
        #    Conv      ->(?, 14, 14, 64)
        #    Pool      ->(?, 7, 7, 64)
        self.layer2=torch.nn.Sequential(
            torch.nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2,stride=2)
        )
        # Final FC 7x7x64 inputs -> 10 outputs
        self.fc=torch.nn.Linear(7*7*64,10)
        torch.nn.init.xavier_uniform(self.fc.weight)
    def forward(self,x):
        out=self.layer1(x)
        out=self.layer2(out)
        out=out.view(out.size(0),-1)# Flatten them for FC
        out=self.fc(out)
        return out

# instantiate CNN model
model=CNN()

# define cost/loss & optimizer
criterion=torch.nn.CrossEntropyLoss()# Softmax is internally computed.
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)

# train my model
print('Learning started. It takes sometime.')
for epoch in range(epochs):
    avg_cost=0
    total_batch=len(ds_train)//batch_size
    for step,(batch_xs,batch_ys) in enumerate(dl):
        x=Variable(batch_xs)#[100, 1, 28, 28] image is already size of (28x28), no reshape
        y=Variable(batch_ys)#[100] label is not one-hot encoded

        optimizer.zero_grad()
        h=model(x)
        cost=criterion(h,y)
        cost.backward()
        optimizer.step()

        avg_cost+=cost/total_batch
    print(epoch+1,avg_cost.item())
print('Learning Finished!')

# Test model and check accuracy
model.eval()#！！将模型设置为评估/测试模式 set the model to evaluation mode (dropout=False)

# x_test=ds_test.test_data.view(len(ds_test),1,28,28).float()
x_test=ds_test.test_data.view(-1,1,28,28).float()
y_test=ds_test.test_labels

pre=model(x_test)

print("pre.data=")
print(pre.data)
print("*"*3)

pre=torch.max(pre.data,1)[1].float()
acc=(pre==y_test.data.float()).float().mean()
print("acc", acc)

r=random.randint(0,len(x_test)-1)
x_r=x_test[r:r+1]
y_r=y_test[r:r+1]
pre_r=model(x_r)

# IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)
# https://discuss.pytorch.org/t/indexerror-dimension-out-of-range-expected-to-be-in-range-of-1-0-but-got-1/54267/12
print("pre_r.data=")
print(pre_r.data)
print("*"*3)

pre_r=torch.max(pre_r.data,-1)[1].float()
print('pre_r')
print(pre_r)

acc_r=(pre_r==y_r.data).float().mean()
print(acc_r)