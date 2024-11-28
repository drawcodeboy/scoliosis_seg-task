import torch
import torchvision
import cv2
import numpy as np
import albumentations as A

class ICHTransforms:
    '''
    How to Augmentations for Segmentation Task with Albumentations
    https://albumentations.ai/docs/getting_started/mask_augmentation/
    
    필요한 Augmentations
    Horizontal Flip, Vertical Flip, Rotation(-45, 45), Zoom(80-120%),
    Pan(-20,20%), Shear(0-10%), all randomization rate = 50%
    
    Tensor가 되기 전에 NumPy 상태에서 처리해야 한다. torchvision이랑 이 부분은 다른 듯
    '''
    def __init__(self):
        self.p = 0.5
        self.train_transforms = [
            A.HorizontalFlip(p=self.p),
            A.VerticalFlip(p=self.p),
            A.Rotate(limit=(-45, 45), p=self.p),
            A.Affine(scale=(0.8, 1.2), p=self.p), # Zoom
            A.Affine(translate_percent={'x': (-0.2, 0.2), 'y': (-0.2, 0.2)}, p=self.p), # Pan
            A.Affine(shear=(-30, 30), p=self.p) # Shear
        ]
        
        self.train_transforms = A.Compose(self.train_transforms)
        
    def __call__(self, image, mask, mode):
        if mode == 'train':
            # pass train Augmentations
            transformed = self.train_transforms(image=image, mask=mask)
            return transformed['image'], transformed['mask']
        elif mode == 'val' or mode == 'test':
            # No Augmentations for Test
            return image, mask