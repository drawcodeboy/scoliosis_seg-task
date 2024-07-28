from torch.utils.data import Dataset, DataLoader

import torch
from PIL import Image
import numpy as np
import cv2
import os

class ScoliosisDataset(Dataset):
    def __init__(self, data_dir, mode='train', model='Mask R-CNN'):
        self.data_dir = data_dir
        self.mode = mode
        self.data_li = [] # {image_path, label_path}
        self.model = model
        self._check()
    
    def __len__(self):
        return len(self.data_li)
    
    def __getitem__(self, idx):
        image, height, width = self.get_image(self.data_li[idx]['image_path'])
        with open(self.data_li[idx]['label_path']) as f:
            label = f.read().split()
        
        # Image To Tensor
        image /= 255.0
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) # for channel
        
        # get Target(boxes, labels, masks)
        label, polygon, bbox = self.get_label_polygon_bbox(label, height, width)
        mask = self.get_mask(polygon, height, width)
        
        # Target To Tensor
        labels = torch.tensor([label], dtype=torch.int64) # (N, ) -> (1, )
        masks = torch.tensor(mask, dtype=torch.uint8).unsqueeze(0) # (N, H, W) -> (1, H, W)
        boxes = torch.tensor(bbox, dtype=torch.float32).unsqueeze(0) # (N, 4) -> (1, 4)
        
        target = {}
        target["labels"] = labels
        target["masks"] = masks
        target["boxes"] = boxes
        
        if self.model == 'U-Net':
            # type casting uint8 to float32
            # target 0(background), 255(roi) => 0.(background), 1.(roi)로 변경
            masks = masks.to(dtype=torch.float32) / 255.0
            target = masks # if model is U-Net, ONLY RETURN MASK
        
        return image, target
        
    
    def get_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        height, width = image.shape
        return image, height, width

    def get_label_polygon_bbox(self, label, width, height):
        """
        Returns:
            - polygon : list, Mask를 이루는 다각형 -> tensor로 transform 필요
            - bbox : list [xmin, ymin, xmax, ymax] -> tensor로 transform 필요
        """
        polygon = [] # spine coordinates
        xmin, ymin, xmax, ymax = 1e5, 1e5, 0., 0.
        width, height = float(width), float(height)
        
        for x, y in zip(label[1::2], label[2::2]):
            x, y = float(x) * width, float(y) * height
            polygon.append([int(x), int(y)])
            xmin, ymin = min(xmin, x), min(ymin, y)
            xmax, ymax = max(xmax, x), max(ymax, y)
        bbox = [xmin, ymin, xmax, ymax]
        
        label = int(label[0])
        
        return label, polygon, bbox
    
    def get_mask(self, polygon, width, height):
        # 
        polygon = np.array(polygon)
        mask = np.zeros((height, width), dtype=np.uint8)
        mask_value = 255
        cv2.fillPoly(mask, [polygon], mask_value) # mask value는 255 or 1?
        return mask
    
    def _check(self):
        data_dir_for_mode = os.path.join(self.data_dir, self.mode) # mode에 따른 data_dir 설정
        images_path = os.path.join(data_dir_for_mode, 'images')
        labels_path = os.path.join(data_dir_for_mode, 'labels')
        
        file_cnt = 0
        
        for i, filename in enumerate(os.listdir(images_path)):
            image_path = os.path.join(images_path, filename)
            label_path = os.path.join(labels_path, f"{filename[:-4]}.txt")
            
            try:
                image = Image.open(image_path)
                with open(label_path) as f:
                    label = f.read()
                self.data_li.append(dict(image_path=image_path, label_path=label_path))
            except:
                print("Can\'t Open this file")
            
            file_cnt += 1
            
            print(f"\rCheck {self.mode} Data {100*file_cnt/len(os.listdir(images_path)):.2f}%", end="")
        print()
            
if __name__ == '__main__':
    ds = ScoliosisDataset(
        data_dir = 'data/AIS.v1i.yolov8',
        mode='train'
    )
    
    image, target = ds[0]
        
    for component in ['boxes', 'labels', 'masks']:
        print(target[component].dtype, target[component].shape)