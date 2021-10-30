import os
import random
from typing import Callable, List
import pandas as pd
import albumentations as alb
import numpy as np
from torch.utils.data import Dataset
import cv2
from refers.data.readers import LmdbReader
from refers.data.structures import ImageCaptionInstance, ImageCaptionBatch
from refers.data.tokenizers import SentencePieceBPETokenizer
from refers.data import transforms as T 

import torch
import torch.utils.data as data
from collections import defaultdict
import glob
import json
import os
import pickle
from typing import Dict, List, Tuple
import re
from loguru import logger
from pathlib import Path
import string
import pdb
from transformers import AutoTokenizer

class CaptioningDataset(Dataset):

    def __init__(
        self,
        data_root: str,
        split: str,
        tokenizer: SentencePieceBPETokenizer,
        image_transform: Callable = T.DEFAULT_IMAGE_TRANSFORM,
        max_caption_length: int = 30,
    ):
        self.filepath = data_root
        self.split = split
        self.images_name,self.captions = self.read_imgPath_texts()

        self.image_transform = image_transform
        self.caption_transform = alb.Compose(
            [
                T.NormalizeCaption(),
                T.TokenizeCaption(tokenizer),
                T.TruncateCaptionTokens(max_caption_length),
            ]
        )

        self.padding_idx = tokenizer.token_to_id("<unk>")
        self.contrastive_tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT",max_length=100)
        

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx: int) -> ImageCaptionInstance:

        def file_filter(f):
                if f[-10:] in ['-small.jpg']:
                    return True
                else:
                    return False
        # pdb.set_trace()  # pdb在这里用不了
        patient = self.images_name[idx]
        image_names = os.listdir(patient)
        image_names = list(filter(file_filter, image_names))
        # print(image_names)
        # print("patient", patient)
        filename1 = random.choice(image_names)
        filename2 = random.choice(image_names)
        filename3 = random.choice(image_names)
        

        filename1 = patient + filename1
        caption1 = self.captions[idx]
        image1 = cv2.imread(filename1)

        # Transform image-caption pair and convert image from HWC to CHW format.
        # Pass in caption to image_transform due to paired horizontal flip.
        # Caption won't be tokenized/processed here.
        image_caption1 = self.image_transform(image=image1, caption=caption1)
        image1, caption1 = image_caption1["image"], image_caption1["caption"]
        image1 = np.transpose(image1, (2, 0, 1))

        caption_tokens1 = self.caption_transform(caption=caption1)["caption"]

        filename2 = patient + filename2
        image2 = cv2.imread(filename2)
        image2 = self.image_transform(image=image2, caption=caption1)["image"]
        image2 = np.transpose(image2, (2, 0, 1))

        filename3 = patient + filename3
        image3 = cv2.imread(filename3)
        image3 = self.image_transform(image=image3, caption=caption1)["image"]
        image3 = np.transpose(image3, (2, 0, 1))

        # contrastive_caption = self.contrastive_tokenizer(caption1,padding=True,truncation=True,return_tensors="pt",max_length=100)

        return ImageCaptionInstance(image1, caption_tokens1, image2, image3, caption1)

    def collate_fn(self, instances: List[ImageCaptionInstance]) -> ImageCaptionBatch:
        return ImageCaptionBatch(instances, padding_value=self.padding_idx)

    def read_imgPath_texts(self):
        text_path = Path(self.filepath)
        csv_path = self.split + "_img_caption_patients.txt"
        text_filename = os.path.join(text_path,csv_path)
        
        df = pd.read_csv(text_filename,sep='\t',names=['img_path','caption'])
        img_path = df['img_path'].values
        caption = df['caption'].values

        return img_path,caption
