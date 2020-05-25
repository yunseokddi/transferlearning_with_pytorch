import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn
from model import *
from data_loader import *

classes = ('cat', 'dog')

dataiter = iter(test_set)
imagges, labels = dataiter.next()

net = Net()
net.load_state_dict(torch.load('trained_net'))

