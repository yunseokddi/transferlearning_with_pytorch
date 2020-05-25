import os
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image
from torchvision import transforms


train_dir = './data/train/'
train_files = os.listdir(train_dir)

class CatDogDataset(Dataset):
    def __init__(self, file_list, dir, mode='train', transform=None):
        self.file_list = file_list
        self.dir = dir
        self.mode = mode
        self.transform = transform

        if self.mode == 'train':
            if 'dog' in self.file_list[0]:
                self.label =1

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


data_transform = transforms.Compose([
    transforms.Resize((150,150)),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip()
])

cat_files = [tf for tf in train_files if 'cat' in tf]
dog_files = [tf for tf in train_files if 'dog' in tf]

cats = CatDogDataset(cat_files, train_dir, transform = data_transform)
dogs = CatDogDataset(dog_files, train_dir, transform = data_transform)

catdogs = ConcatDataset([cats, dogs])

dataloader = DataLoader(catdogs, batch_size=32, shuffle=True, num_workers=4)
