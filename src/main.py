import torch
from torch.utils.data import TensorDataset,DataLoader
#import torchvision
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from train import train_model
from ConvNet import ConvNetwork

#%matplotlib inline
#import time

print("torch version:::", torch.__version__)

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

train = pd.read_csv("../input/train.csv")
#train = pd.read_csv("../input/train.csv", sep=",")
print(train.shape)

validation_set = (train.iloc[0:5000,1:].values).astype('float32') # all pixel values
label_validation_set = train.iloc[0:5000,0].values.astype('int32') # only labels i.e targets digits

train_set = (train.iloc[5000:,1:].values).astype('float32') # all pixel values
label_train_set = train.iloc[5000:,0].values.astype('int32') # only labels i.e targets digits
#X_test = test.values.astype('float32')

#Convert train datset to (num_images, img_rows, img_cols) format 
train_set = train_set.reshape(train_set.shape[0], 1, 28, 28)
validation_set = validation_set.reshape(validation_set.shape[0], 1, 28, 28)

# Generating 1-hot label vectors
# trainLabel2 = np.zeros((37000,10))
# validLabel2 = np.zeros((5000,10))
# for d1 in range(label_train_set.shape[0]):
#   trainLabel2[d1,label_train_set[d1]] = 1
#for d2 in range(label_validation_set.shape[0]):
#    validLabel2[d2,label_validation_set[d2]] = 1

# plt.imshow(train_set[0][0], cmap=plt.get_cmap('gray'))
# plt.show()
# for i in range(6, 9):
#     plt.subplot(330 + (i+1))
#     plt.imshow(train_set[i], cmap=plt.get_cmap('gray'))
#     plt.title(train_set[i])

#Load Data Using Torch Dataloader
trainDataset = TensorDataset(torch.from_numpy(train_set), torch.from_numpy(label_train_set))
validDataset = TensorDataset(torch.from_numpy(validation_set), torch.from_numpy(label_validation_set))
# Creating dataloader
trainLoader = DataLoader(trainDataset, batch_size=64, shuffle=True,num_workers=4, pin_memory=True)
validLoader = DataLoader(validDataset, batch_size=64, shuffle=True,num_workers=4, pin_memory=True)

# Checking availability of GPU
use_gpu = torch.cuda.is_available()
if use_gpu:
    print("GPU is available for computin\n")
#Model Intialization
if use_gpu:
    model = ConvNetwork().float().cuda()
else:
    model = ConvNetwork()

#Loss Function
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 100
learning_rate = 1e-4

model = train_model(model, optimizer, criterion, num_epochs, learning_rate, trainLoader, validLoader, use_gpu)

print("Done")
