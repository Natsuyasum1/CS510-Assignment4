from turtle import down
import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

class MNIST_Dataset:
    def __init__(self, root: str, data_size: int):
        transform = transforms.ToTensor()

        self.train_set = MyMNIST(root=root, train=True, download=True, transform=transform, data_size=data_size)
        # print(len(self.train_set))
        self.test_set = MNIST(root=root, train=False, download=True)
    
    def loader(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers:int = 0) -> (DataLoader, DataLoader):
        trainloader = DataLoader(self.train_set, batch_size=batch_size, shuffle=shuffle_train)
        return trainloader
    
    def get_full_dataset(self):
        return self.train_set.getitems()

    def get_test_data(self):
        transform = transforms.ToTensor()
        data, targets = [], []
        for index in range(len(self.test_set.data)):
            if len(targets) == 10:
                break
            target = self.test_set.targets[index]
            if target not in targets:
                temp = self.test_set.data[index]
                img = Image.fromarray(temp.numpy(), mode='L')
                img = transform(img)
                data.append(img)
                targets.append(target)
        # data = np.asarray(data)
        # targets = np.asarray(targets)
        return data
        

class MyMNIST(MNIST):
    def __init__(self, root: str, train: bool = True, transform = None, target_transform = None, download: bool = False, data_size: int = 100) -> None:
        super().__init__(root, train, transform, target_transform, download)

        if data_size > 0:
            data, targets, counter = [], [], [0 for i in range(10)]
            for index in range(len(self.data)):
                target = self.targets[index]
                if counter[target] < data_size:
                    data.append(np.asarray(self.data[index]))
                    targets.append(int(target))
                    counter[target] += 1
            self.data = np.asarray(data)
            self.targets = np.asarray(targets)
        else:
            self.data = np.asarray(self.data)
            self.targets = np.asarray(self.targets)

    def __getitem__(self, index: int):
        # return super().__getitem__(index)
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def getitems(self):
        return self.data, self.targets