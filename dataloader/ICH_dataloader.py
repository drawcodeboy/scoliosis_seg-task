from torch.utils.data import Dataset, DataLoader

import torch
from PIL import Image
import numpy as np
import cv2
import os

class ICHDataset(Dataset):
    def __init__(self, 
                 data_dir='data/physionet.org/files/ct-ich/1.3.1/data_only', 
                 mode='train'):
        '''
            [Args]
                - model
                    - U-Net: Image와 Binary Mask를 리턴
                    - Mask R-CNN: Image, Bounding Box, Binary Mask를 리턴
        '''
        self.data_dir = data_dir
        self.mode = mode
        self.data_li = [] # {image_path, label_path}
        
        self.val = 10 if data_dir.split('/')[-1] == 'data_only' else 200
        self.test = 50 if data_dir.split('/')[-1] == 'data_only' else 500
        
        self._check()
        self._split()
    
    def __len__(self):
        return len(self.data_li)
    
    def __getitem__(self, idx):
        image, height, width = self.get_image(self.data_li[idx]['image_path'])
        mask, _, __ = self.get_image(self.data_li[idx]['label_path'])
        
        # Image To Tensor
        image /= 255.0
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) # for channel
        
        # get Target(boxes, labels, masks)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        mask = torch.where(mask > 128.0, 1.0, 0.0)
        
        return image, mask
    
    def get_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        height, width = image.shape
        return image, height, width
    
    def _check(self):
        images_path = os.path.join(self.data_dir, 'image')
        labels_path = os.path.join(self.data_dir, 'label')
        
        file_cnt = 0
        
        for i, filename in enumerate(os.listdir(images_path)):
            # if i == 1: break # for debugging
            # print(filename)
            image_path = os.path.join(images_path, filename)
            label_path = os.path.join(labels_path, f"mask_{filename}")
            
            try:
                image = Image.open(image_path)
                label = Image.open(label_path)
                self.data_li.append(dict(image_path=image_path, label_path=label_path))
            except:
                print("Can\'t Open this file")
            file_cnt += 1
            
            print(f"\rCheck {self.mode} Data {100*file_cnt/len(os.listdir(images_path)):.2f}%", end="")
        print()
    
    def _split(self):
        train_len = len(self.data_li)-(self.val+self.test)
        if self.mode == 'train':
            self.data_li = self.data_li[:train_len]
        elif self.mode == 'val':
            self.data_li = self.data_li[train_len:train_len+self.val]
        elif self.mode == 'test':
            self.data_li = self.data_li[train_len+self.val:]