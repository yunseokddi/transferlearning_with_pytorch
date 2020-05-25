import torchvision.transforms as transforms
import os.path
import torchvision.datasets as datasets
import torch.utils.data as data

traindir = os.path.join('./data/data_set', 'training')
testdir = os.path.join('./data/data_set', 'testing')


class TrainImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        filename = self.imgs[index]

        if (filename[0].split('/')[4]).split('.')[0] == 'cat':
            label = 0
        else:
            label = 1

        return super(TrainImageFolder, self).__getitem__(index)[0], label


class TestImageFolder(datasets.ImageFolder):

    def __getitem__(self, index):
        filename = self.imgs[index]
        real_idx = int(filename[0].split('/')[4].split('.')[0])
        return super(TestImageFolder, self).__getitem__(index)[0], real_idx


train_set = data.DataLoader(
    TrainImageFolder(traindir,
                     transforms.Compose([
                         transforms.Resize((150, 150)),
                         transforms.ToTensor()
                     ])),
    batch_size=4,
    shuffle=True,
    num_workers=4,
    pin_memory=True)

test_set = data.DataLoader(
    TestImageFolder(testdir,
                    transforms.Compose([
                        transforms.Resize((150, 150)),
                        transforms.ToTensor()
                    ])),
    batch_size=4,
    shuffle=False,
    num_workers=1,
    pin_memory=False)
