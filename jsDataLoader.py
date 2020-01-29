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

    def __init__(self, csv_file1, root_dir, csv_file2=None, transform=None):
        """
        Args:
            csv_file1 (string): Path to the csv file with most annotations.
            root_dir (string): Directory with all the images.
            csv_file2 (string): Path to the csv file with control annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv = pd.read_csv(csv_file1)
        if csv_file2 != None:
            csv2 = pd.read_csv(csv_file2).loc[:,'id_code':'sirna']
            self.csv = pd.concat([self.csv, csv2])\
                         .sort_values(['id_code'])\
                         .reset_index()\
                         .drop(columns=['index'])
        self.root_dir = root_dir
        self.directory = sorted( os.listdir(self.root_dir) )
	
    def __len__(self):
        return len(self.directory)//6

    def __getitem__(self, idx): # I think that the best way of doing this is going to be to go plate by plate.
        # Do first  
        idx = idx * 6
        filename = self.directory[idx]
        path = os.path.join(self.root_dir, filename)
        image = io.imread(path)
        totalTensor = torch.from_numpy(image).unsqueeze(0)
        
        for _ in range(5): 
            idx = idx+1
            filename = self.directory[idx]
            path = os.path.join(self.root_dir, filename)
            image = io.imread(path)
            imageTensor = torch.from_numpy(image).unsqueeze(0)
            totalTensor = torch.cat( (totalTensor, imageTensor), 0)
        
        # Get sirna (label)
        pathParts = self.root_dir.split("/")
        experiment = pathParts[2]
        plate = pathParts[3][5:]
        pathParts = filename.split("_")
        well = pathParts[0]
        id_code = experiment + "_" + plate + "_" + well
        #print("id_code: ", id_code)       
        try: 
            result = self.csv.where(self.csv["id_code"]==id_code, inplace=False).dropna()
            #print("result: ", result)
            sirna = result.iloc[0].loc["sirna"]
        except:
            sirna = -2
        
        if sirna=='UNTREATED': sirna = -1
        else: sirna = float(re.search('[0-9]+', sirna).group())
        sirnaTensor = torch.tensor([sirna])
        return totalTensor.float(), sirnaTensor.float()
        
test_dataset = RecursionDataset(csv_file1='../train-labels/train.csv', root_dir='../train-data/HEPG2-01/Plate1', csv_file2='../train-labels/train_controls.csv')

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
