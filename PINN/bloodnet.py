# scitific cal
import numpy as np
from scipy.spatial.distance import pdist, squareform
import copy
import math
# plotting
import pandas as pd
import matplotlib.pylab as plt
import matplotlib.ticker as ticker
# system
from time import time
import sys
import os
import gc
import subprocess # Call the command line
from subprocess import call
import pdb
# torch import
import torch
import torch.nn as nn
import torchvision.transforms as transforms

import torch.nn.functional as F
from torch.nn.functional import normalize  # noqa: F401
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.distributions import Gamma
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Swish(nn.Module):
        def __init__(self, inplace=True):
            super(Swish, self).__init__()
            self.inplace = inplace

        def forward(self, x):
            if self.inplace:
                x.mul_(torch.sigmoid(x))
                return x
            else:
                return x * torch.sigmoid(x)
class FloodNet(torch.nn.Module):
    def __init__(self, n_feature, n_hidden):
        super(FloodNet, self).__init__()
        self.args = list()
        self.kwargs = {"n_feature": n_feature, "nhidden": n_hidden}
#         self.mu = args.mu
#         self.sigma = args.sigma
#         self.scale = args.scale
#         self.nu = args.nu
#         self.rho = args.rho
#         self.rInlet = args.rInlet
#         self.xStart = args.xStart
#         self.xEnd = args.xEnd
#         self.dP = args.dP
#         self.L = args.L
        self.mu = 0.5
        self.sigma = 0.1
        self.scale = 0.005
        self.nu = 4.1e-3
        self.rho = 1.037
        # self.rInlet = 0.05
        self.xStart = 29
        self.xEnd = 67
        self.dP = 1
        self.L = 10 #mm
        self.u_scale = 10 #cm/s
        self.p = 0.75
        self.euler = 1/self.rho*self.p/(self.u_scale**2)
        self.re = 259.1

        
        self.linear = torch.nn.Linear(n_feature, n_hidden)
        self.weights = torch.nn.Linear(n_hidden, n_hidden)
        self.attn = torch.nn.Linear(n_hidden,n_hidden)
        self.predict = torch.nn.Linear(n_hidden, 3)
        # self.conv1 = nn.Conv2d(n_feature, n_hidden,kernel_size=k, stride=s, padding=p) 
        # self.conv2 = nn.Conv2d(n_hidden, n_hidden*2,kernel_size=k, stride=s, padding=p)
        # self.conv3 = nn.Conv2d(n_hidden*2, n_hidden*2*2,kernel_size=k, stride=s, padding=p)
        # self.predict = nn.Conv2d(n_hidden*2*2, 3,kernel_size=1, stride=1)

        
    def forward(self, x):
        out = Swish()(self.linear(x))
        w = Swish()(self.weights(Swish()(self.linear(x))))
        h = torch.mul(out,w)
        w = Swish()(self.weights(h))
        h = torch.mul(out,w)
        w = Swish()(self.weights(h))
        h = torch.mul(out,w)
        w = Swish()(self.weights(h))
        h = torch.mul(out,w)
        w = Swish()(self.weights(h))
        h = torch.mul(out,w)
        output = self.predict(h)
        return output

    
    def reset_parameters(self, verbose=False):
        for module in self.modules():
            if isinstance(module, self.__class__):
                continue
        if 'reset_parameters' in dir(module):
            if callable(module.reset_parameters):


                module.reset_parameters()
            if verbose:
                print("Reset parameters in {}".format(module))
def init_normal(m):
        if type(m) == nn.Linear:
          nn.init.kaiming_normal_(m.weight)    
