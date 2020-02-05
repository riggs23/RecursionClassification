from __future__ import print_function, division
# Correction for PIL
import sys
#sys.path.append('/lib/python3.7/site-packages')
from PIL import Image
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, utils, datasets
import re


class RecursionDataset(Dataset):
    """Recursion Dataset for Big Data Capstone."""

    def __init__(self, csv_file1, root_dir, csv_file2=None, transform=None, shuffle=True):
        """
        Args:
            csv_file1 (string): Path to the csv file with most annotations.
            root_dir (string): Directory with all the batch folders containing images.
            csv_file2 (string): Path to the csv file with control annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            shuffle (boolean): Optional shuffling, defaults to True
        """
        self.csv = pd.read_csv(csv_file1)
        if csv_file2 != None:
            csv2 = pd.read_csv(csv_file2).loc[:,'id_code':'sirna']
            self.csv = pd.concat([self.csv, csv2])\
                         .reset_index(drop=True)
        self.csv['plate'] = 'Plate'+self.csv['plate'].astype(str) # Mimic folder naming for loading pics later
        
        # Create variable for both sites 1 and 2 of each well
        self.csv['site'] = 's1'
        csv_copy = self.csv.copy()
        csv_copy['site'] = 's2'
        self.csv = pd.concat([self.csv, csv_copy])\
                     .sort_values(['id_code', 'site'])\
                     .reset_index(drop=True)
        
        if shuffle == True:
            self.csv = self.csv.sample(frac=1).reset_index(drop=True)
        self.root_dir = root_dir
    
    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, idx):
        # Generate full filename of image from csv file row info
        pathParts = self.csv.iloc[idx,:]
        pathGen = os.path.join(self.root_dir, pathParts['experiment'], pathParts['plate'])
        filenameGen = pathParts['well']+'_'+pathParts['site']+'_w'
        for i in range(1,7):
            filenameFull = filenameGen+str(i)+'.png'
            pathFull = os.path.join(pathGen, filenameFull)
            image = io.imread(pathFull)
            if i == 1:
                totalTensor = torch.from_numpy(image).unsqueeze(0)
            else:
                imageTensor = torch.from_numpy(image).unsqueeze(0)
                totalTensor = torch.cat( (totalTensor, imageTensor), 0)
        
        try:
            sirna = self.csv.iloc[idx,:].loc['sirna']
        except:
            sirna = -2
        
        if sirna=='UNTREATED': sirna = -1
        else: sirna = float(re.search('[0-9]+', sirna).group())
        sirnaTensor = torch.tensor([sirna])
        return totalTensor.float(), sirnaTensor.float()
        
test_dataset = RecursionDataset(csv_file1='../train-labels/train.csv', root_dir='../train-data', csv_file2='../train-labels/train_controls.csv')

#print(len(test_dataset))

#for i in range(1000):
#   data, label = test_dataset[i] 
#   print("data: ", data.shape)
#   print("label: ", label.item())

dataloader = DataLoader(test_dataset, batch_size=4)

for index, data in enumerate(dataloader):
    image = data[0]
    label = data[1]
    
    print("image: ", image.shape)
    print("label: ", label)
