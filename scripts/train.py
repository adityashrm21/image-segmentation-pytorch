import torch
import cv2
from glob import glob
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import os

def train(n_epoch):

    vgg_model = VGGNet(model="vgg19", requires_grad=True)
    model = FCN8s(pretrained_net=vgg_model, n_class=2)
    model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    criterion = nn.BCELoss()

    for epoch in range(n_epoch):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            sample = data
            images = sample['image']
            images = images.float()
            labels = sample['label']
            labels = labels.float()
            images = Variable(images.cuda())
            labels = Variable(labels.cuda(), requires_grad=False)

            optimizer.zero_grad()
            output = model(images)
            output = torch.sigmoid(output)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 5 == 4:    # print every 5 mini-batches
                print('Epoch: %d, Loss: %.4f' %
                      (epoch + 1, running_loss / 5))
                running_loss = 0.0
    return model

def main():
    np.random.seed(1234)
    model = train(85)
