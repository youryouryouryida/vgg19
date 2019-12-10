import torch
import torchvision
import numpy as np
from torchvision.datasets import ImageFolder
from torchvision import transforms
import matplotlib.pyplot as plt
import time
from torch.autograd import Variable

model=torchvision.models.resnet50(pretrained=True)