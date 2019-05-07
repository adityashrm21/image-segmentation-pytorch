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

def gen_test_output(n_class):
    model.eval();
    for i, data in enumerate(testloader):
        sample = data
        images = sample['image']
        images = images.float()
        images = Variable(images.cuda())

        output = model(images)
        output = torch.sigmoid(output)
        output = output.cpu()
        N, c, h, w = output.shape
        pred = np.squeeze(output.detach().cpu().numpy(), axis=0)

        pred = pred.transpose((1, 2, 0))
        pred = pred.argmax(axis=2)
        pred = (pred > 0.5)

        pred = pred.reshape(*pred.shape, 1)
        pred = np.concatenate((pred, np.invert(pred)), axis=2).astype('float')
        pred = np.concatenate((pred, np.zeros((*pred[:,:,0].shape, 1))), axis=2).astype('float')

        pred[pred == 1.0] = 127.0
        images = images.cpu().detach().numpy()
        images = np.squeeze(images)
        images = images.transpose((1, 2, 0))

        images = denormalize(images, mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        output = cv2.addWeighted(images, 0.6, pred, 0.4, 0, dtype = 0)
        output = output/127.0
        output = np.clip(output, 0.0, 1.0)
        yield test_paths[i], output


def main():
    np.random.seed(1234)
    model = train(85)
