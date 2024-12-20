import sys
import os
import cv2
import argparse
import torch
from random import shuffle
import numpy as np
from skimage.morphology import skeletonize

from models.nets import SegFormer

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    
    parser.add_argument('--device', default='cuda')
    
    parser.add_argument('--num-classes', type=int, default=1)
    parser.add_argument('--scale', default='B3')
    
    parser.add_argument("--load_weights_dir", default="saved/weights")
    parser.add_argument("--load-weights", default="SegFormer-B3-061-dice_loss.pth")
    
    # Method
    parser.add_argument("--save", action='store_true')
    parser.add_argument("--save-path")
    parser.add_argument("--image-path")
    parser.add_argument("--vis-method", type=int, default=1)
    
    return parser

def encode_image(image_path):
    # return tensor for input to model
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    image /= 255.0
    orig_image = image.copy()
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # ADD: Batch, Channel DIMENSION

    return orig_image, image

def decode_image(image: torch.tensor):
    # (C, H, W) -> (H, W, C)
    return image.permute(1, 2, 0).detach().cpu().numpy()

def get_all_images_path():
    train_path = r'data\AIS.v1i.yolov8\train\images'
    val_path = r'data\AIS.v1i.yolov8\valid\images'
    test_path = r'data\AIS.v1i.yolov8\test\images'
    
    all_paths = []
    for parent_dir in [train_path, val_path, test_path]:
        for path in os.listdir(parent_dir):
            all_paths.append(os.path.join(parent_dir, path))
    
    return all_paths

def get_only_mask(image, mask_pred):
    only_mask = image.copy()
    
    # only_mask = np.zeros_like(image)
    # only_mask[mask_pred == 1.] = 1.
    
    only_mask[mask_pred == 0] = 0
    
    only_mask = (only_mask * 255.0).astype(np.uint8)
    
    return only_mask
    
def main(args):
    if args.device == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    # Load Model    
    model = SegFormer(num_classes=args.num_classes, phi=args.scale.lower()).to(device)
    ckpt = torch.load(os.path.join(args.load_weights_dir, args.load_weights), map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    
    print(f"Load Model Complete")
    
    images_path = get_all_images_path()
    
    if args.image_path:
        images_path = [args.image_path]
    
    shuffle(images_path)
    
    for idx, image_path in enumerate(images_path):
        
        if idx == 100:
            break
        
        # Get Image Tensor for Model Input
        orig_image, image = encode_image(image_path)
        
        # Prediction
        prediction = model(image)
        mask_pred = torch.where(prediction >= 0.5, 1., 0.).squeeze().detach().cpu().numpy()
        
        image = decode_image(image[0]) # get First Batch, Batch Size is 1
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) # gray to color
        
        # 마스크를 적용하여 색칠
        colored_image = image.copy()
        colored_image[mask_pred > 0] = [0, 0, 255]
        
        alpha = 0.3  # 색칠된 이미지 가중치
        gamma = 0    # 추가 상수

        # 가중치 합성
        blended = cv2.addWeighted(colored_image, alpha, image, 1-alpha, gamma)
        
        # only mask with lines
        only_mask = get_only_mask(image, mask_pred)
        
        image_num = image_path.split('\\')[-1].split('_')[0]
        
        print(blended.shape, blended.dtype)
        print(mask_pred.shape, mask_pred.dtype)
        print(image.shape, image.dtype)
        
        if args.save_path is None:
            save_path = fr"E:\Scoliosis-Segmentation\Mask_Prediction\SegFormer\segformer_{int(image_num):04d}.jpg"
        else:
            save_path = args.save_path
            
        blended = (blended * 255.).astype(np.uint8)
        if args.save:
            if args.vis_method == 1:
                cv2.imwrite(save_path, blended)
            elif args.vis_method == 2:
                orig_save_path = fr"E:\Scoliosis-Segmentation\SegFormer\mask_test\orig_{int(image_num):04d}.jpg"
                mask_save_path = fr"E:\Scoliosis-Segmentation\SegFormer\mask_test\mask_{int(image_num):04d}.jpg"
                
                mask_pred = (mask_pred * 255.).astype(np.uint8)
                image = (image * 255.).astype(np.uint8)
                
                cv2.imwrite(orig_save_path, image)
                cv2.imwrite(mask_save_path, mask_pred)
        else:
            if args.vis_method == 1:
                # 원본 이미지에 Masking을 한 이미지
                cv2.imshow('Mask on Image', blended)
            elif args.vis_method == 2:
                # Binary Mask
                cv2.imshow('Binary Mask', mask_pred)
                # Original Image
                cv2.imshow('Original Image', image)
            
        print(f'pass {image_num}')
        
    cv2.waitKeyEx()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    
    args = parser.parse_args()
    main(args)