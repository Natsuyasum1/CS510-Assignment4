import torch
import numpy as np
from datasets.mnist import MNIST_Dataset
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
from trainer import Trainer
from networks.Autoencoder import Autoencoder
import matplotlib.pyplot as plt

device = torch.device("cuda")
# data_size = 100


# ---------------------------- question 1 ---------------------------------
# data_size = 100
# dataset = MNIST_Dataset('./datasets', data_size = data_size)
# data, targets = dataset.get_full_dataset()
# data = data.reshape(data.shape[0], -1)

# # kmeans = KMeans(n_clusters=10).fit(data)
# # print(kmeans.labels_)
# kmeans = KMeans(n_clusters=10, n_init=50, max_iter=3000).fit_predict(data)

# results = [[0 for i in range(10)] for j in range(10)]
# for i in range(len(kmeans)):
#     results[targets[i]][kmeans[i]] += 1
# df = pd.DataFrame(data=results)
# print(df)

# ---------------------------- question 2.a ---------------------------------
# data_size = 1000
# batch_size = 128
# epochs = 200
# lr = 1e-4
# dataset = MNIST_Dataset('./datasets', data_size = data_size)
# train_loader = dataset.loader(batch_size=batch_size)

# model = Autoencoder()
# model = model.to(device)

# trainer = Trainer(model, device)
# trainer.train(train_loader=train_loader, lr=lr, epochs=epochs)
# features, targets = trainer.extract_feature(train_loader=train_loader)
# features = features.reshape(features.shape[0], -1)
# print(features.shape)

# kmeans = KMeans(n_clusters=10, n_init=50, max_iter=3000).fit_predict(features)

# results = [[0 for i in range(10)] for j in range(10)]
# for i in range(len(kmeans)):
#     results[targets[i]][kmeans[i]] += 1
# df = pd.DataFrame(data=results)
# print(df)

# ---------------------------- question 2.b ---------------------------------
# data_size = 1000
# batch_size = 128
# epochs = 200
# lr = 1e-4
# dataset = MNIST_Dataset('./datasets', data_size = data_size)
# train_loader = dataset.loader(batch_size=batch_size)

# model = Autoencoder()
# model = model.to(device)

# trainer = Trainer(model, device)
# trainer.train(train_loader=train_loader, lr=lr, epochs=epochs)
# features, targets = trainer.extract_feature(train_loader=train_loader)
# features = features.reshape(features.shape[0], -1)
# print(features.shape)

# pca = PCA(n_components=10)

# features = pca.fit_transform(features)

# kmeans = KMeans(n_clusters=10, n_init=50, max_iter=3000).fit_predict(features)

# results = [[0 for i in range(10)] for j in range(10)]
# for i in range(len(kmeans)):
#     results[targets[i]][kmeans[i]] += 1
# df = pd.DataFrame(data=results)
# print(df)

# ---------------------------- question 3 ---------------------------------
num_noise = 200
data_size = -1
batch_size = 128
epochs = 50
lr = 1e-4
dataset = MNIST_Dataset('./datasets', data_size = data_size)
train_loader = dataset.loader(batch_size=batch_size)

model = Autoencoder()
model = model.to(device)

trainer = Trainer(model, device)
trainer.train_denoising(train_loader=train_loader, lr=lr, epochs=epochs, num_noise=num_noise)

test_data = dataset.get_test_data()
ori_pic = []
for data in test_data:
    ori_pic.append(data.cpu().clone().detach().data.numpy().reshape(28, 28))

noised_pic, outputs = trainer.test_denoising(test_data=test_data, num_noise=num_noise)

plt.figure()
f, axarr = plt.subplots(10, 3)
for i in range(10):
    axarr[i][0].imshow(ori_pic[i], cmap='gray', vmin=0, vmax=1)
    axarr[i][1].imshow(noised_pic[i], cmap='gray', vmin=0, vmax=1)
    axarr[i][2].imshow(outputs[i], cmap='gray', vmin=0, vmax=1)
    i += 1
plt.show()


