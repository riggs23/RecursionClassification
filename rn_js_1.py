from __future__ import print_function, division

# Set variables for testing
num_workers = 30
batch_size = 64
n_epochs = 100
lr = 4e-4
save_path = './model/resnet18_100'

print('num_workers:', num_workers)
print('batch_size:', batch_size)
print('n_epochs:', n_epochs)

import multiprocessing
print('CPUs:', multiprocessing.cpu_count(), '\n')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from torchvision import transforms, utils, datasets
import torchvision.models as models
from tqdm import tqdm
import pdb
import sys
import gc
from skimage import io, transform
from torch.nn.parameter import Parameter

from IPython.core.ultratb import AutoFormattedTB

from RecursionDS import RecursionDataset

# from torchvision import transforms, utils, datasets

assert torch.cuda.is_available() # GPU must be available

test_dataset = RecursionDataset(csv_file1='../recursion_data/train-labels/train.csv', root_dir='../recursion_data/train-data', csv_file2='../recursion_data/train-labels/train_controls.csv')

model = models.resnet18(pretrained=False)
model.load_state_dict(torch.load('resnet18-5c106cde.pth'))

model.conv1 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.fc = nn.Linear(in_features=512, out_features=test_dataset.n_classes, bias=True)

if torch.cuda.device_count() > 1:
  print('Using', torch.cuda.device_count(), 'GPUs')
  model = nn.DataParallel(model)

model.cuda()
objective = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

losses = []
full_loss = []
full_acc = []

loss_report = 0
acc_report = 0
for epoch in range(n_epochs):
  dataloader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=num_workers)
  loop = tqdm(total=len(dataloader), position=0, file=sys.stdout)

  for batch, (x, y_truth) in enumerate(dataloader):
    x = x.cuda()
    y_truth = y_truth.cuda()
    y_truth = y_truth.type(torch.cuda.LongTensor).squeeze(1) #NOTE: making y_hat a 1D tensor for crossEnropyLoss function

    optimizer.zero_grad()
    y_hat = model(x)

    loss = objective(y_hat, y_truth)

    loss.backward()

    #losses.append(loss)

    loop.set_description('epoch:{}'.format(epoch))
    loop.update(1)

    optimizer.step()

  loop.close()

  print('\nGround Truth:', ' '.join('%5s' % y_truth[j].item() for j in range(4)))
  _, predicted = torch.max(y_hat, 1)
  print('Predicted: ', ' '.join('%5s' % predicted[j].item() for j in range(4)))

# Model Saving
state = {
    "epoch": n_epochs,
    "state_dict": model.state_dict(),
    "optimizer": optimizer.state_dict()
}
torch.save(state, save_path)

#state_loaded = torch.load("../jsCode/model/resnet18")
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
  #y_hat = model_loaded(x)
  y_hat = model(x)
  _, predicted = torch.max(y_hat, 1)
  total += labels.size(0)
  correct += (predicted == y_truth).sum().item()
  loop.update(1)
loop.close()
acc_report = correct/total
print('Accuracy:', acc_report)

gc.collect()
print('GPU Mem Used:', torch.cuda.memory_allocated(0) / 1e9)
