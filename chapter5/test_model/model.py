import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64,128,3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer4 = nn.Sequential(
            nn.Linear(17*17*128, 512),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(-1, 17*17*128)
        out = self.layer4(out)

        return out

