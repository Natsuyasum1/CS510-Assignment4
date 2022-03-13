import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import MSELoss
import numpy as np
import cv2
import random

class Trainer:
    def __init__(self, model, device) -> None:
        self.model = model
        self.device = device

    def train(self, train_loader, lr, epochs):
        start_time = time.time()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = MSELoss()

        self.model.train()
        for epoch in range(epochs):
            epoch_start_time = time.time()
            loss_epoch = 0.0
            n_batches = 0

            for data, _ in train_loader:
                data = data.to(self.device)
                data = data.to(torch.float)
                optimizer.zero_grad()
                loss = criterion(self.model(data), data)
                loss.backward()
                optimizer.step()

                loss_epoch += loss.item()
            n_batches += 1
            epoch_train_time = time.time() - epoch_start_time
            print('Epoch {}/{}\t Time: {:.3f}\t Loss: {:.6f}'
                .format(epoch+1, epochs, epoch_train_time, loss_epoch/n_batches))
        train_time = time.time() - start_time
        print('Training time: %.3f' % train_time)
        print('Finished training.')

    def extract_feature(self, train_loader):
        self.model.eval()
        
        features, targets = [], []

        with torch.no_grad():
            for data, target in train_loader:
                data = data.to(self.device)

                f = self.model.encoder(data)
                f = f.cpu().detach().data.numpy()
                for i in range(len(f)):
                    features.append(f[i])
                    targets.append(target[i])
        features = np.asarray(features)
        targets = np.asarray(targets)
        return features, targets

    def train_denoising(self, train_loader, lr, epochs, num_noise):
        start_time = time.time()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = MSELoss()

        self.model.train()
        for epoch in range(epochs):
            epoch_start_time = time.time()
            loss_epoch = 0.0
            n_batches = 0

            for data, _ in train_loader:
                original_pic = data.detach()
                original_pic = original_pic.to(self.device)
                original_pic = original_pic.to(torch.float)
                for index in range(len(data)):
                    for i in range(num_noise):
                        # print(data[index].shape)
                        x = random.randint(0, 27)
                        y = random.randint(0, 27)
                        data[index][0][x][y] = 1.0
                data = data.to(self.device)
                data = data.to(torch.float)
                optimizer.zero_grad()
                loss = criterion(self.model(data), original_pic)
                loss.backward()
                optimizer.step()

                loss_epoch += loss.item()
            n_batches += 1
            epoch_train_time = time.time() - epoch_start_time
            print('Epoch {}/{}\t Time: {:.3f}\t Loss: {:.6f}'
                .format(epoch+1, epochs, epoch_train_time, loss_epoch/n_batches))
        train_time = time.time() - start_time
        print('Training time: %.3f' % train_time)
        print('Finished training.')

    def test_denoising(self, test_data, num_noise):
        noised_pic, outputs = [], []
        self.model.eval()
        with torch.no_grad():
            for data in test_data:
                for i in range(num_noise):
                    x = random.randint(0, 27)
                    y = random.randint(0, 27)
                    data[0][x][y] = 1.0
                noised_pic.append(data.cpu().detach().data.numpy().reshape(28, 28))
                data = data.to(self.device)
                data = data.to(torch.float)
                outputs.append(self.model(data.reshape(1, 1, 28, 28)).cpu().detach().data.numpy().reshape(28, 28))
        noised_pic = np.asarray(noised_pic)
        outputs = np.asarray(outputs)
        return noised_pic, outputs