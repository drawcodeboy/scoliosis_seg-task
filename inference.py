import sys
import os
import cv2
import argparse
import torch
import numpy as np

from models.nets import SegFormer

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    
    parser.add_argument('--device', default='cuda')
    
    parser.add_argument('--image_path')
    
    parser.add_argument('--num-classes', type=int, default=1)
    parser.add_argument('--scale', default='B2')
    
    parser.add_argument("--load_weights_dir", default="saved/weights")
    parser.add_argument("--load-weights")
    return parser

def encode_image(image_path):
    # return tensor for input to model
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    image /= 255.0
    orig_image = image.copy()
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # ADD: Batch, Channel DIMENSION

    return orig_image, image
    
def main(args):
    if args.device == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    # Load Model    
    model = SegFormer(num_classes=args.num_classes, phi=args.scale.lower()).to(device)
    ckpt = torch.load(os.path.join(args.load_weights_dir, args.load_weights), map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    
    # Get Image Tensor for Model Input
    orig_image, image = encode_image(args.image_path)
    
    # Prediction
    prediction = model(image)
    mask_pred = prediction.squeeze().detach().cpu().numpy()
    
    # Visualize
    cv2.imshow('Origianl', orig_image)
    cv2.imshow('Prediction', mask_pred)
    cv2.waitKeyEx()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    
    args = parser.parse_args()
    main(args)