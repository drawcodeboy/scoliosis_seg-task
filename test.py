import torch
import argparse
import os, sys
import time
import cv2

from dataloader import load_dataset
from models.nets import load_model
from models.loss_fn import load_loss_fn

from torch.utils.data import DataLoader
from torch import nn

from utils import *

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    # GPU
    parser.add_argument("--use-cuda", action='store_true')
    parser.add_argument("--dist", action='store_true')
    
    # Dataset
    parser.add_argument("--dataset", default='scoliosis')
    parser.add_argument("--mode", default='test')
    
    # Model
    parser.add_argument("--model", default='SegFormer-B0')
    parser.add_argument("--num-classes", type=int, default=1, help="Num of Classes without Background")
    
    # Loss function
    parser.add_argument("--loss-fn", default='dice')
    
    # Hyperparameters
    parser.add_argument("--batch-size", type=int, default=8)
    
    # Load Model Weights
    parser.add_argument("--load_weights_dir", default="saved/weights")
    parser.add_argument("--load-weights")
    
    return parser

def print_info(device, args):
    print("######[Settings]######")
    print(f"device : {device}")
    print(f"dataset : {args.dataset}")
    print(f"mode : {args.mode}")
    print(f"batch size : {args.batch_size}")
    print("######################")
    
def get_masked_image(image, mask):
    '''
    CT Image에 Mask 영역을 색칠해서 리턴해주는 함수
    Args:
        image: Tensor(1, H, W), range[0, 1], torch.float32
        mask: Tensor(1, H, W), range[0, 1], torch.float32
    Return:
        image: np.ndarray(3, H, W), np.float32
        mask: np.ndarray(3, H, W), np.float32
        blended(masked_image): np.ndarray(3, H, W), np.float32
    '''
    # Tensor to Numpy(for cv2 processing)
    image = (image.permute(1, 2, 0) * 255.).detach().cpu().numpy().astype(np.float32)
    mask = (mask.permute(1, 2, 0) * 255.).detach().cpu().numpy().astype(np.float32)
    
    # Image (1 channel to 3 channel)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    
    # Get Color Mask
    color_mask = np.zeros_like(image, dtype=np.float32)
    color_mask[:, :] = [0, 0, 255]  # 빨간색
    
    # Color 합성
    # 마스크를 적용하여 색칠
    colored_image = image.copy()
    colored_image[mask > 0] = cv2.addWeighted(image, 1, color_mask, 0.5, 0)[mask > 0]
    
    alpha = 0.5  # 색칠된 이미지 가중치
    beta = 0.5   # 원본 이미지 가중치
    gamma = 0    # 추가 상수

    # 가중치 합성
    blended = cv2.addWeighted(colored_image.astype(np.float32), alpha, image.astype(np.float32), beta, gamma)
    
    return image, mask, blended
    
    
def main(args):
    device = 'cpu'
    if args.use_cuda and torch.cuda.is_available():
        device ='cuda'
    
    print_info(device, args)
    
    # Dataset
    test_ds = load_dataset(dataset=args.dataset, mode=args.mode)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size)
    
    image, mask = test_ds[0]
    image, mask, masked_image = get_masked_image(image, mask)
    '''
    cv2.imwrite("./test_image.jpg", image)
    cv2.imwrite("./test_mask.jpg", mask)
    cv2.imwrite("./test_masked_image.jpg", masked_image)
    sys.exit()
    '''
    
    # Model
    model_name = args.model.split('-')[0].lower()
    model_scale = args.model.split('-')[1].lower()
    model = load_model(model_name=model_name, scale=model_scale, num_classes=args.num_classes).to(device)
    
        # Distributed Learning
    if args.dist == True:
        model = nn.DataParallel(model, device_ids=[0, 1], output_device=0)
    
    ckpt = torch.load(os.path.join(args.load_weights_dir, args.load_weights), map_location=device, weights_only=False)    
    model.load_state_dict(ckpt['model'])
    
    print(f"* {args.model}")
    print(f"\t* It was trained {ckpt['epochs']} EPOCHS")
    
    # Loss function
    loss_fn = load_loss_fn(loss_fn=args.loss_fn)
    
    # Evaluate
    start_time = int(time.time())
    test_loss = evaluate(None, model, test_dl, loss_fn, None, device)
    test_time = int(time.time() - start_time)
    print(f"Test Time: {test_time//60:02d}m {test_time%60:02d}s")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    
    args = parser.parse_args()
    main(args)