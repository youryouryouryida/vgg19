import torch
import torchvision
import numpy as np
from torchvision.datasets import ImageFolder
from torchvision import transforms
import matplotlib.pyplot as plt
import time
from torch.autograd import Variable
import cv2

def imshow(inp):
    """imshow for tensor"""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)

simple_transform = transforms.Compose([transforms.Scale((224, 224)), transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
test = ImageFolder('/home/l/dataset/iau/test/' , simple_transform)
test_data = torch.utils.data.DataLoader(test, batch_size=1, num_workers=1,shuffle=False)
# imshow(test[0][0])
# plt.show()
dataset_sizes = float(len(test))
print('dataset_sizes',dataset_sizes)
checkpoint=torch.load('/home/l/dataset/iau/21class_vgg19epoch_50.pth')
model_ft=torchvision.models.vgg19(pretrained=True)
model_ft.classifier[6].out_features = 21
model_ft.load_state_dict(checkpoint['net'])
# print(model_ft)

if torch.cuda.is_available():
    model_ft = model_ft.cuda()

model_ft.train(False)

running_corrects=0

for data_label in test_data:

    inputs, labels = data_label
    if torch.cuda.is_available():
        # inputs = Variable(inputs.cuda())
        # labels = Variable(labels.cuda())

        inputs = Variable(inputs)
        # print(inputs)
        labels = Variable(labels)
        # print(labels)
        inputs, labels = inputs.cuda(), labels.cuda()
    outputs = model_ft(inputs)
    _, preds = torch.max(outputs.data, 1)
    running_corrects += torch.sum(preds == labels.data)
acc = running_corrects / dataset_sizes

print('test_acc:',float(acc))