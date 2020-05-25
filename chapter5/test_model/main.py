import os
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from model import *
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class CatDogDataset(Dataset):
    def __init__(self, file_list, dir, mode='train', transform=None):
        self.file_list = file_list
        self.dir = dir
        self.mode = mode
        self.transform = transform
        if self.mode == 'train':
            if 'dog' in self.file_list[0]:
                self.label = 1
            else:
                self.label = 0

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.dir, self.file_list[idx]))
        if self.transform:
            img = self.transform(img)
        if self.mode == 'train':
            img = img.numpy()
            return img.astype('float32'), self.label
        else:
            img = img.numpy()
            return img.astype('float32'), self.file_list[idx]


test_dir = './data/test1'
test_files = os.listdir(test_dir)

filename_path = 'trained_net.pth'

test_transform = transforms.Compose([
        transforms.Resize((150,150)),
        transforms.ToTensor()
])

testset = CatDogDataset(test_files, test_dir, mode='test', transform=test_transform)
testloader = DataLoader(testset, batch_size=40, shuffle=False, num_workers=4)

device = 'cuda'
model = Net().to(device)

model.eval()
fn_list = []
pred_list = []


for x, fn in testloader:
    with torch.no_grad():
        x = x.to(device)
        output = model(x)
        pred = torch.argmax(output, dim =1)
        fn_list += [n[:-4] for n in fn]
        pred_list += [p.item() for p in pred]

submission = pd.DataFrame({"id":fn_list, "label":pred_list})
submission.to_csv('preds_densenet121.csv', index=False)

samples, _ = iter(testloader).next()
samples = samples.to(device)
fig = plt.figure(figsize=(24, 16))
fig.tight_layout()
output = model(samples[:24])
pred = torch.argmax(output, dim=1)
pred = [p.item() for p in pred]
ad = {0:'cat', 1:'dog'}
for num, sample in enumerate(samples[:24]):
    plt.subplot(4,6,num+1)
    plt.title(ad[pred[num]])
    plt.axis('off')
    sample = sample.cpu().numpy()
    plt.imshow(np.transpose(sample, (1,2,0)))