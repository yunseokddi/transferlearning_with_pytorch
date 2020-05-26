import numpy as np
import pandas as pd
import os
import torch
import torchvision
import torch.nn as nn
import torch.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

all_labels_df = pd.read_csv('./data/labels.csv')

breeds = all_labels_df.breed.unique()
breed2idx = dict((breed, idx) for idx, breed in enumerate(breeds))
idx2breed = dict((idx, breed) for idx, breed in enumerate(breeds))

all_labels_df["label_idx"] = [breed2idx[b] for b in all_labels_df.breed]


class DogDataset(Dataset):
    def __init__(self, label_df, img_path, transform=None):
        self.label_df = label_df
        self.img_path = img_path
        self.transform = transform

    def __len__(self):
        return self.label_df.shape[0]

    def __getitem__(self, idx):
        label = self.label_df.label_idx[idx]
        id_img = self.label_df.id[idx]
        img_P = os.path.join(self.img_path, id_img) + ".jpg"
        img = Image.open(img_P)

        if self.transform:
            img = self.transform(img)

        return img, label


IMG_SIZE = 224
BATCH_SIZE = 256
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]
CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")

train_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize(IMG_MEAN, IMG_STD)
])

val_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMG_MEAN, IMG_STD)
])

dataset_name = ["train", "valid"]
stra_splt = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
train_index, val_index = next(iter(stra_splt.split(all_labels_df.id, all_labels_df.breed)))
train_df = all_labels_df.iloc[train_index, :].reset_index()
val_df = all_labels_df.iloc[val_index, :].reset_index()

img_transforms = {"train": train_transform, "valid": val_transform}

train_dataset = DogDataset(train_df, "./data/train", transform=img_transforms["train"])
valid_dataset = DogDataset(val_df, "./data/train", transform=img_transforms["valid"])
image_dataset = {"train": train_dataset, "valid": valid_dataset}

image_dataloader = {x: DataLoader(image_dataset[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4) for x in
                    dataset_name}
datasize = {x: len(image_dataset[x]) for x in dataset_name}

# Define model
model_ft = models.resnet50(pretrained=True)

for param in model_ft.parameters():
    param.requires_grad = False

# fc 수정
num_fc_ftr = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_fc_ftr, len(breeds))
model_ft = model_ft.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([{"params": model_ft.fc.parameters()}], lr=0.001)


def train(model, device, train_loader, epoch):
    model.train()

    for batch_idx, data in enumerate(train_loader):
        x, y = data
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()

    print('Train Epoch: {}\t Loss: {:.6f}'.format(epoch, loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_hat = model(x)
            test_loss += criterion(y_hat, y).item()
            pred = y_hat.max(1, keepdim=True)[1]
            correct += pred.eq(y.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(valid_dataset),
        100. * correct / len(valid_dataset)))


for epoch in range(1, 9):
    train(model_ft, DEVICE, image_dataloader["train"], epoch)
    test(model_ft, DEVICE, image_dataloader["valid"])

PATH = './trained_parameter.pth'
torch.save(model_ft.state_dict(), PATH)
