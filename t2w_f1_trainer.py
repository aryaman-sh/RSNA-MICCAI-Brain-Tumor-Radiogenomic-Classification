# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import glob
from sklearn.metrics import roc_auc_score
from torch.nn import functional as torch_functional
import torchvision.transforms as tf
from torch.utils.data import DataLoader
from torch.optim import optimizer
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
import random

drive.mount('/content/drive')

a = [1,2,3,4]
a[-1:]

class T2wData(Dataset):
    def __init__(self, test=False, transforms=None):
        self.test = test
        self.patients = os.listdir('/zero')
        random.shuffle(self.patients)
        if test == True:
            self.patients = self.patients[-55:]
        else:
            self.patients = self.patients[:-55]
        self.transforms = transforms
                                          
    def __len__(self):
        return len(self.patients)
    
    def __getitem__(self, x):
        patient = self.patients[x]
        filename = '/zero/' + patient
        image = np.load(filename, allow_pickle=True)
        
        images = image[0][0]
        
        images = torch.from_numpy(images)
        
        images = torch.reshape(images, (1, 64, 256, 256))
        if self.transforms is not None:
            images = self.transforms(images)
                                        
        label = image[0][1].item()
        label = torch.tensor(abs(label), dtype=torch.float)
        
        return images, label

class ConvSection(nn.Module):
    """
    A convolution section
    """
    def __init__(self, in_channels, mid_channels, resize_conv=None, stride=1):
        super(ConvSection, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_channels)

        self.conv2 = nn.Conv3d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(mid_channels)

        self.conv3 = nn.Conv3d(mid_channels, mid_channels*4, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn3 = nn.BatchNorm3d(mid_channels*4)
        
        self.relu = nn.ReLU()
        self.resize_conv = resize_conv
        self.stride = stride

    def forward(self, x):
        idn = x.clone()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        
        if self.resize_conv is not None:
            idn = self.resize_conv(idn)
        
        x += idn
        x = self.relu(x)
        return x

class Resnet503D(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Resnet503D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU()
        self.mpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.apool = nn.AdaptiveAvgPool3d((1,1,1)) #TODO: set avgadaptive pool output
        
        self.layer_channel = 64
        self.conv2_x = self.convLayer(3, 64, 1)
        self.conv3_x = self.convLayer(4, 128, 2)
        self.conv4_x = self.convLayer(6, 256, 2)
        self.conv5_x = self.convLayer(3, 512, 2)
        
        self.fc = nn.Linear(512*4, num_classes)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(262144, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dp1 = nn.Dropout(0.4)
        self.dp2 = nn.Dropout(0.4)
        
    def convLayer(self, num_blocks, mid_channels, stride):
        layers = []
        resize_conv = None

        if stride != 1 or self.layer_channel != mid_channels*4:
            resize_conv = nn.Sequential(
                nn.Conv3d(self.layer_channel, mid_channels*4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(mid_channels*4),
            )
        layers.append(ConvSection(self.layer_channel, mid_channels, resize_conv, stride))
        self.layer_channel = mid_channels*4
        for i in range(num_blocks - 1):
            layers.append(ConvSection(self.layer_channel, mid_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.mpool(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.apool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        #x = self.flatten(x)
        #x = self.fc1(x)
        #x = self.relu(x)
        #x = self.dp1(x)
        #x = self.fc2(x)
        #x = self.relu(x)
        #x = self.dp2(x)
        #x = self.fc3(x)
        # with logits output
        return x

def train_loss(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    y_all = []
    outputs_all = []
    
    with torch.no_grad():    
        for X, y in dataloader:
            X = X.to(device='cuda', dtype=torch.float)
            y = y.to(device='cuda', dtype=torch.float)

            pred = model(X)
            outputs = pred.squeeze(1)
            
            outputs_all.extend(outputs.tolist())
            y_all.extend(y.tolist())
            
            #print(outputs, y)
            test_loss += loss_fn(outputs, y).item()
                
    y_all = [1 if x > 0.5 else 0 for x in y_all]
    auc = roc_auc_score(y_all, outputs_all)
    test_loss /= num_batches
    
    print(f"Test set: Avg loss: {test_loss:>8f}, AUC: {auc}")
    
    return test_loss

model = Resnet503D(in_channels=1, num_classes=1)
#writer = SummaryWriter()
DEVICE = 'cuda'
BATCH_SIZE = 11
LEARNING_RATE = 1e-3
EPOCHS = 100

train_losses = []
test_losses = []

transforms = tf.Compose([
      tf.RandomRotation(360),
])

train_ds = T2wData(transforms=transforms)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

model.to(device=DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.BCEWithLogitsLoss()
scaler = torch.cuda.amp.GradScaler()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=6, verbose=True)

test_ds = T2wData(test=True)
test_dl = DataLoader(test_ds, batch_size=2, shuffle=True)
    
def save_checkpoint2(state, auc):
    filename = 'T2w-R50-final-' + str(auc) + '.pth.tar'
    print("==> saving checkpoint ")
    torch.save(state, filename)

for epoch in range(EPOCHS):
    outputs_all = []
    y_all = []
    print(f"EPOCH = {epoch + 1} \n +++++++++++++++++++++++++++++++++")
    size = len(train_dl.dataset)
    losses = []

    for batch, (X, y) in enumerate(train_dl):
        X = X.to(device=DEVICE, dtype=torch.float)
        y = y.to(device=DEVICE, dtype=torch.float)

        with torch.cuda.amp.autocast():
            pred = model(X)
            loss = loss_fn(pred.squeeze(1), y)
        #writer.add_scalar("Loss/train", loss, epoch)
        losses.append(loss.item())
        
        outputs_all.extend(pred.squeeze(1).tolist())
        y_all.extend(y.tolist())

        scaler.scale(loss).backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        if batch % 20 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    
    #writer.flush()
    mean_loss = sum(losses) / len(losses)

    scheduler.step(mean_loss)
    
    y_all = [1 if x > 0.5 else 0 for x in y_all]
    auc = roc_auc_score(y_all, outputs_all)

    print(f'Training AUC Score ==> {auc}')
    print(f'loss at epoch {epoch + 1} was {mean_loss:.5f}')
    
    test_loss_out = train_loss(test_dl, model, loss_fn)
    
    if epoch+1 > 3:
        checkpoint = {'state_dic':model.state_dict()}
        save_checkpoint2(checkpoint, test_loss_out)

    train_losses.append(mean_loss)
    test_losses.append(test_loss_out)
    
print("Done")

import matplotlib.pyplot as plt

plt.plot(train_losses, 'b')
plt.plot(test_losses, 'r')
plt.grid()
plt.show()

