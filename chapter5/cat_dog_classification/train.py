import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn
from model import *
from data_loader import *
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import torchvision

writer = SummaryWriter('runs/experiment1')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = Net()
net.to(device)

total_batch = len(train_set)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

epochs = 100

for epoch in range(epochs):
    running_loss = 0.0
    acc = 0.
    correct = 0
    total = 0

    for i, data in enumerate(train_set, 0):
        inputs, labels = data
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, prediction = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (prediction == labels).sum().item()

        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.6f acc : %.6f' % (
            epoch + 1, i + 1, running_loss / 2000, 100 * correct / total))
            running_loss = 0.0

            writer.add_scalar('training loss',
                              running_loss / 100,
                              epoch * len(train_set) + i)


PATH = './trained_net'
torch.save(net.state_dict(), PATH)







