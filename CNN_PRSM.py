import torch.utils.data.dataloader
from torch.utils.data import Dataset,DataLoader
import numpy as np
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2 as cv
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from torch.optim import lr_scheduler
import os
import random
data_path = '/home/mohsen/Downloads/PRSM'



transform = {'Train' : transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor()
                                ,transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]),
            'Test' : transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor()
                                ,transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])}

PRSM_class = {0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',
           10:'10',11:'11',12:'12',13:'13',14:'14',15:'15',16:'16',17:'17'}
# PRSM_class = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]


image_datasets = {x: datasets.ImageFolder(os.path.join(data_path, x),
                                          transform[x])
                  for x in ['Train', 'Test']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=0)
              for x in ['Train', 'Test']}


d = iter(dataloaders['Test'])
image,label = next(d)
print(type(label))


sizes = {x: len(image_datasets[x]) for x in ['Test','Train']}
print(sizes)
# print(image_datasets['Train'].classes)





# def imshow(inp, title):
#     """Imshow for Tensor."""
#     inp = inp/2 +0.5
#     inp = inp.numpy().transpose((1, 2, 0))
#     plt.imshow(inp)
#     plt.show()


# inputs, classes = next(iter(dataloaders['Train']))
# out = torchvision.utils.make_grid(inputs)
# imshow(out, title=[PRSM_class[x] for x in classes])



class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1 = nn.Linear(8*8*32,128)
        self.fc2 = nn.Linear(128,len(PRSM_class))


    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,32*8*8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)
num_epochs = 1

for epoch in range(num_epochs):
    for i ,(images , labels) in enumerate(dataloaders['Train']):
        outputs = model(images)
        # print(type(labels))
        # print(type(outputs))
        loss = criterion(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if (i+1) % 2000 == 0:
            print (f'Epoch [{epoch+1}/{5}], Step [{i+1}/ {sizes}], Loss: {loss.item():.4f}')

# <class 'torch.Tensor'>
# {'Test': 13611, 'Train': 68056}
# Epoch [1/5], Step [2000/ {'Test': 13611, 'Train': 68056}], Loss: 1.2475
# Epoch [1/5], Step [4000/ {'Test': 13611, 'Train': 68056}], Loss: 0.4570
# Epoch [1/5], Step [6000/ {'Test': 13611, 'Train': 68056}], Loss: 0.7741
# Epoch [1/5], Step [8000/ {'Test': 13611, 'Train': 68056}], Loss: 1.2375
# Epoch [1/5], Step [10000/ {'Test': 13611, 'Train': 68056}], Loss: 0.0983
# Epoch [1/5], Step [12000/ {'Test': 13611, 'Train': 68056}], Loss: 0.7707
# Epoch [1/5], Step [14000/ {'Test': 13611, 'Train': 68056}], Loss: 2.4731
# Epoch [1/5], Step [16000/ {'Test': 13611, 'Train': 68056}], Loss: 0.5303

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in dataloaders['Test']:
        images = images
        labels = labels
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        for i in range(4):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(18):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {PRSM_class[i]}: {acc} %')
