import torch
import argparse
import os
import time

from dataloader import dataset
from models.nets import SegFormer
from models.loss_fn import get_loss_fn

from torch.utils.data import DataLoader

from engine import *
from utils import *

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    # GPU
    parser.add_argument("--use-cuda", action='store_true')
    
    # Dataset
    parser.add_argument("--dataset", default='scoliosis')
    parser.add_argument("--mode", default='test')
    parser.add_argument("--data_dir", default='data/AIS.v1i.yolov8')
    
    # Model
    parser.add_argument("--scale", default='B0', help="MiT Scale of SegFormer")
    parser.add_argument("--num-classes", type=int, default=1, help="Num of Classes without Background")
    
    # Loss function
    parser.add_argument("--unified-loss-fn", action='store_true')
    
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
    

def main(args):
    device = 'cpu'
    if args.use_cuda and torch.cuda.is_available():
        device ='cuda'
    
    print_info(device, args)
    
    # Dataset
    test_ds = dataset(dataset='scoliosis', data_dir=args.data_dir, mode=args.mode)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size)
    
    # Model
    model = SegFormer(num_classes=args.num_classes, phi=args.scale.lower()).to(device)
    ckpt = torch.load(os.path.join(args.load_weights_dir, args.load_weights), map_location=device)
    model.load_state_dict(ckpt['model'])
    print(f"It was trained {ckpt['epochs']} EPOCHS")
    
    # Loss function
    loss_fn = get_loss_fn(imbalance=args.unified_loss_fn)
    
    # Evaluate
    start_time = int(time.time())
    test_loss = evaluate(None, model, test_dl, loss_fn, device)
    test_time = int(time.time() - start_time)
    print(f"Test Time: {test_time//60:02d}m {test_time%60:02d}s")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    
    args = parser.parse_args()
    main(args)