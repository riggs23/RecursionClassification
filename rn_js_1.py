from __future__ import print_function, division

# Set variables for testing
num_workers = 30
batch_size = 32
n_epochs = 20
old_n_epochs = 5
lr = 3e-4
save_path = f'./model/resnet18_{n_epochs+old_n_epochs}'
if old_n_epochs == 0:
  load_path = ''
else:
  load_path = f'./model/resnet18_{old_n_epochs}'
freeze = True

print("model: resnet18")
print('num_workers:', num_workers)
print('batch_size:', batch_size)
print('n_epochs:', n_epochs)
print('load_path:', load_path)
print('save_path:', save_path)
print('freeze layers:', freeze)

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
from collections import OrderedDict
from skimage import io, transform
from torch.nn.parameter import Parameter
import pandas as pd

from IPython.core.ultratb import AutoFormattedTB

from RecursionDS import RecursionDataset

# from torchvision import transforms, utils, datasets

assert torch.cuda.is_available() # GPU must be available

test_dataset = RecursionDataset(csv_file1='../recursion_data/train-labels/train.csv', root_dir='../recursion_data/train-data', csv_file2='../recursion_data/train-labels/train_controls.csv')

model = models.resnet18(pretrained=False)
model.load_state_dict(torch.load('resnet18-5c106cde.pth'))

if freeze:
  for param in model.parameters():
    param.requires_grad = False

model.conv1 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.fc = nn.Linear(in_features=512, out_features=test_dataset.n_classes, bias=True)

if load_path != '':
  state_loaded = torch.load(load_path)
  new_state_dict = OrderedDict()
  for k, v in state_loaded['state_dict'].items():
    name = k[7:] # remove module.
    new_state_dict[name] = v
  model.load_state_dict(new_state_dict)

if torch.cuda.device_count() > 1:
  print('Using', torch.cuda.device_count(), 'GPUs')
  model = nn.DataParallel(model)

model.cuda()
objective = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
if load_path != '':
  optimizer.load_state_dict(state_loaded['optimizer'])

losses = []
full_loss = []
full_acc = []

loss_report = 0
acc_report = 0
for epoch in range(n_epochs):
  dataloader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=num_workers)
  loop = tqdm(total=len(dataloader), position=0, file=sys.stdout)

  for batch, (x, y_truth) in enumerate(dataloader):
    model.train()
    
    x = x.cuda()
    y_truth = y_truth.cuda()
    y_truth = y_truth.type(torch.cuda.LongTensor).squeeze(1) #NOTE: making y_hat a 1D tensor for crossEnropyLoss function

    optimizer.zero_grad()
    y_hat = model(x)

    loss = objective(y_hat, y_truth)

    loss.backward()

    loop.set_description('epoch:{}'.format(epoch))
    loop.update(1)

    optimizer.step()

  loop.close()

  print('\nGround Truth:', ' '.join('%5s' % y_truth[j].item() for j in range(4)))
  _, predicted = torch.max(y_hat, 1)
  print('Predicted: ', ' '.join('%5s' % predicted[j].item() for j in range(4)))

  with torch.no_grad():
    # Evaluate training loss and accuracy
    model.eval()
    losses.append(loss.item())
    # Evaluate accuracy
    loss_inter = []
    total = 0
    correct = 0
    loop = tqdm(total=len(dataloader), position=0)
    for images, labels in dataloader:
      images, labels = images.cuda(), labels.cuda()
      labels = labels.type(torch.cuda.LongTensor).squeeze(1)
      y_hat = model(images)
      _, predicted = torch.max(y_hat, 1)
      loss = objective(y_hat, labels)
      loss_inter.append(loss.item())
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
      loop.update(1)
    loop.close()
    full_acc.append(correct/total)
    full_loss.append(np.mean(loss_inter))
    print('Accuracy for epoch', epoch, '-', correct/total)
report = pd.DataFrame({
'train_loss': losses,
'full_acc': full_acc,
'avg_loss': full_loss
})
report.to_csv('report.csv')
  
# Model Saving
state = {
    "epoch": n_epochs,
    "state_dict": model.state_dict(),
    "optimizer": optimizer.state_dict()
}
torch.save(state, save_path)

gc.collect()
print('GPU Mem Used:', torch.cuda.memory_allocated(0) / 1e9)
