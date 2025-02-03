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
计算图 Compute Graph 和自动求导 Auto | PyTorch 深度学习实战
https://blog.csdn.net/samurais/article/details/145319886

Code sample from https://zachcolinwolpe.medium.com/pytorchs-dynamic-graphs-autograd-96ecb3efc158
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
autograd
'''
import plotly.graph_objects as go
import plotly.express as px
from torch import nn
import numpy as np
import torch
import math

# data generating process
X  = torch.tensor(np.linspace(-10, 10, 1000))
y  = 1.5 * torch.sin(X) + 1.2 * torch.cos(X/4)
yt = y + np.random.normal(0, 1, 1000)

# vis
def plotter(X, y, yhat=None, title=None):
    with torch.no_grad():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X, y=y, mode='lines',    name='y'))
        fig.add_trace(go.Scatter(x=X, y=yt, mode='markers', marker=dict(size=4), name='yt'))
        if yhat is not None: fig.add_trace(go.Scatter(x=X, y=yhat, mode='lines', name='yhat'))
        fig.update_layout(template='none', title=title)
        fig.show()

plotter(X, y, title='Data Generating Process')

# build model
def fit_model(theta:torch.tensor=torch.rand(3, requires_grad=True)):
    return theta[0] * X + theta[1] * torch.sin(X) + theta[2] * torch.cos(X/4)

# params
theta = torch.randn(3, requires_grad=True)

# optimization proce
loss_fn  = nn.MSELoss()                         # MSE loss
optimizer = torch.optim.SGD([theta], lr=0.01)   # build optimizer 

# run training
epochs = 500
for i in range(epochs):
    yhat = fit_model(theta)
    loss = loss_fn(y, yhat)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if i % (epochs/10) == 0:
        msg = f"loss: {loss.item():>7f} theta: {theta.detach().numpy()}"
        yhat = fit_model(theta)
        plotter(X, y, yhat.detach(), title=f"loss: {loss.item():>7f} theta: {theta.detach().numpy().round(3)}")