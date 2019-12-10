#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 14:27:03 2019

@author: l
"""

import torch
import torchvision
import numpy as np
from torchvision.datasets import ImageFolder
from torchvision import transforms
import matplotlib.pyplot as plt
import time
from torch.autograd import Variable

# import torch.utils.data.DataLoader as dataloader


simple_transform = transforms.Compose([transforms.Scale((224, 224)), transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

train = ImageFolder('/home/l/dataset/iau/train/', simple_transform)
valid = ImageFolder('/home/l/dataset/iau/valid/', simple_transform)
print(train.class_to_idx)
print(train)


def imshow(inp):
    """imshow for tensor"""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)


train_data = torch.utils.data.DataLoader(train, batch_size=16, num_workers=4,shuffle=True)
valid_data = torch.utils.data.DataLoader(valid, batch_size=16, num_workers=4,shuffle=False)

dataloaders = {"train": train_data,
               "valid": valid_data,
               }

dataset_sizes = {"train": float(len(train)),
                 "valid": float(len(valid)),
                 }
# resnet
# model_ft = torchvision.models.resnet18(pretrained=True)
# for param in model_ft.parameters():
#     param.requires_grad = False
# num_ftrs = model_ft.fc.in_features
# model_ft.fc = torch.nn.Linear(num_ftrs, 12)

# vgg16
model_ft=torchvision.models.vgg19(pretrained=True)
model_ft.classifier[6].out_features = 21

# model_ft = torchvision.models.resnet152(pretrained=True)
# for feature in model_ft.parameters():
#     feature.requires_grad = False
# for feature in model_ft.fc.parameters():
#     feature.requires_grad = True
# class_num=12
# channel_in = model_ft.fc.in_features
# model_ft.fc =torch.nn.Linear(channel_in,class_num)


if torch.cuda.is_available():
    model_ft = model_ft.cuda()

learning_rate = 0.001
criterion = torch.nn.CrossEntropyLoss()
optimizer_ft = torch.optim.SGD( model_ft.parameters(), lr=learning_rate, momentum=0.9)
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7)


def train_model(model, criterion, optimizer, scheduler, num_epochs=20):
    since = time.time()
    train_loss=[]
    valid_loss=[]
    train_acc=[]
    valid_acc=[]

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 20)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                print("training...")
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                print("validating")
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if torch.cuda.is_available():
                    # inputs = Variable(inputs.cuda())
                    # labels = Variable(labels.cuda())
                    inputs = Variable(inputs)
                    labels = Variable(labels)
                    inputs, labels = inputs.cuda(), labels.cuda()
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            if phase=='train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
            else:
                valid_loss.append(epoch_loss)
                valid_acc.append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch' : epoch}
                torch.save(state,'/home/l/dataset/vggbestmodel.pth')


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return train_loss,valid_loss,train_acc,valid_acc
def main():
    num_epochs=50
    train_loss,valid_loss,train_acc,valid_acc=train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs)
    fig=plt.figure()
    ax1=fig.add_subplot(2,2,1)
    ax2=fig.add_subplot(2,2,2)
    ax3=fig.add_subplot(2,2,3)
    ax4=fig.add_subplot(2,2,4)

    ax1.plot(np.arange(num_epochs),train_acc,c=(205/255,92/255,92/255),linewidth = 6)
    ax2.plot(np.arange(num_epochs),train_loss,c=(30/255,144/255,255/255),linewidth = 6)
    ax3.plot(np.arange(num_epochs),valid_acc,c=(205/255,92/255,92/255),linewidth = 6)
    ax4.plot(np.arange(num_epochs),valid_loss,c=(30/255,144/255,255/255),linewidth = 6)
    plt.show()

if __name__=="__main__":
    main()
