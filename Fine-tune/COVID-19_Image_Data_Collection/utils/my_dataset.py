import os
import torch
import torch.utils.data as data
import numpy as np 
import pandas as pd 

from PIL import Image
import cv2
import pdb

class XRAY(data.Dataset):

    def __init__(self, root, data_volume , split="train", transform=None):
        super(XRAY, self)
        # pdb.set_trace()
        if data_volume == '1':
            train_label_data = "train_1.txt"
        if data_volume == '10':
            train_label_data = "train_10.txt"
        if data_volume == '50':
            train_label_data = "train_50.txt"
        if data_volume == '100':
            train_label_data = "train.txt"
        test_label_data = "test.txt"
        val_label_data = "val.txt"
        self.split = split
        self.root = root
        self.transform = transform
        self.listImagePaths = []
        self.listImageLabels = []

        if self.split == "train":
            downloaded_data_label_txt = train_label_data
        elif self.split == "val":
            downloaded_data_label_txt = val_label_data
        elif self.split == "test":
            downloaded_data_label_txt = test_label_data

        # ---- Open file, get image paths and labels
        fileDescriptor = open(os.path.join(root, downloaded_data_label_txt), "r")

        # ---- get into the loop
        line = True
        root1 = os.path.join('../../data/COVID-19_Image_Data_Collection/images/')
        while line:
            line = fileDescriptor.readline()
            # --- if not empty
            if line:
                lineItems = line.split()
                imagePath = os.path.join(root1, lineItems[0])
                imageLabel = lineItems[1:]
                imageLabel = [int(i) for i in imageLabel]

                self.listImagePaths.append(imagePath)
                self.listImageLabels.append(imageLabel)

        fileDescriptor.close()

    def __getitem__(self, index):

        imagePath = self.listImagePaths[index]

        imageData = Image.open(imagePath).convert('RGB')
        imageLabel = torch.FloatTensor(self.listImageLabels[index])

        if self.transform is not None:
            imageData = self.transform(imageData)

        return imageData, imageLabel

    def __len__(self):

        return len(self.listImagePaths)
