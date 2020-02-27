from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms, utils, datasets
import torchvision.models as models
from tqdm import tqdm
from torch.nn.parameter import Parameter
import pdb
import torchvision
import os
import sys
import gzip
import tarfile
import gc
from PIL import Image
import pandas as pd
from skimage import io, transform
import matplotlib.pyplot as plt
import re

from IPython.core.ultratb import AutoFormattedTB

from RecursionDS import RecursionDataset

# from torchvision import transforms, utils, datasets

assert torch.cuda.is_available(), "You need to request a GPU from Runtime > Change Runtime"

test_dataset = RecursionDataset(csv_file1='../recursion_data/train-labels/train.csv', root_dir='../recursion_data/train-data', csv_file2='../recursion_data/train-labels/train_controls.csv')

model = models.resnet18(pretrained=True)
model.conv1 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.fc = nn.Linear(in_features=512, out_features=1508, bias=True)
'''model = models.vgg16(pretrained=True)
model.features[0] = nn.Conv2d(6, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
model.classifier[6] = nn.Linear(in_features=4096, out_features=test_dataset.num_classes, bias=True)'''

model.cuda()
objective = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

losses = []
full_loss = []
full_acc = []

loss_report = 0
acc_report = 0
for epoch in range(5):
  dataloader = DataLoader(test_dataset, batch_size=16, pin_memory=True, shuffle=True)
  loop = tqdm(total=len(dataloader), position=0, file=sys.stdout)

  for batch, (x, y_truth) in enumerate(dataloader):
    x = x.cuda()
    y_truth = y_truth.cuda()
    y_truth = y_truth.type(torch.cuda.LongTensor).squeeze(1) #NOTE: making y_hat a 1D tensor for crossEnropyLoss function

    optimizer.zero_grad()
    y_hat = model(x)

    loss = objective(y_hat, y_truth)

    loss.backward()

    losses.append(loss)

    loop.set_description('epoch:{}'.format(epoch))
    loop.update(1)

    optimizer.step()

  loop.close()

  print('\nGround Truth:', ' '.join('%5s' % y_truth[j].item() for j in range(4)))
  _, predicted = torch.max(y_hat, 1)
  print('Predicted: ', ' '.join('%5s' % predicted[j].item() for j in range(4)))

# Model Saving
#state = {
#    "epoch": 10,
#    "state_dict": model.state_dict(),
#    "optimizer": optimizer.state_dict()
#}
#torch.save(state, "./model")

#state_loaded = torch.load("./model")
## To Restore
#model_loaded = models.resnet18(pretrained=True)
#model_loaded.conv1 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#model_loaded.fc = nn.Linear(in_features=512, out_features=test_dataset.num_classes, bias=True)
#model_loaded.load_state_dict(state['state_dict'])
#model_loaded.cuda()

total = 0
correct = 0
loop = tqdm(total=len(dataloader), position=0)
for images, labels in dataloader:
  images, labels = images.cuda(), labels.cuda()
  labels = labels.type(torch.cuda.LongTensor).squeeze(1)
  y_hat = model_loaded(x)
  _, predicted = torch.max(y_hat, 1)
  total += labels.size(0)
  correct += (predicted == y_truth).sum().item()
  loop.update(1)
loop.close()
acc_report = correct/total
print('Accuracy:', acc_report)

gc.collect()
print('GPU Mem Used:', torch.cuda.memory_allocated(0) / 1e9)
