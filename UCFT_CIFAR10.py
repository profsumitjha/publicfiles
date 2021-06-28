from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import sys
import time
import argparse
import datetime

from torch.autograd import Variable

from typing import Optional
from art.classifiers import PyTorchClassifier

from resnet import ResNet18 as resnet18



#code to connect to evaluation script
def get_art_model(
    model_kwargs: dict, wrapper_kwargs: dict, weights_path: Optional[str] = None
) -> PyTorchClassifier:

    model = resnet18() 
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(weights_path))
    model.to(DEVICE)
    cudnn.benchmark = True
    wrapped_model = PyTorchClassifier(
        model,
        loss=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-100),
        input_shape=(32, 32, 3),
        nb_classes=10,
        clip_values=(0.0, 1.0),
        **wrapper_kwargs,
    )    
    return wrapped_model

# x = x.permute(0, 3, 1, 2)
