import torch
import argparse
import os
import time

from dataloader import load_dataset
from models.nets import load_model
from models.loss_fn import load_loss_fn

from torch.utils.data import DataLoader

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
    

def main(args):
    device = 'cpu'
    if args.use_cuda and torch.cuda.is_available():
        device ='cuda'
    
    print_info(device, args)
    
    # Dataset
    test_ds = load_dataset(dataset='scoliosis', data_dir=args.data_dir, mode=args.mode)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size)
    
    # Model
    model_name = args.model.split('-')[0].lower()
    model_scale = args.model.split('-')[1].lower()
    model = load_model(model_name=model_name, scale=model_scale, num_classes=args.num_classes).to(device)
    ckpt = torch.load(os.path.join(args.load_weights_dir, args.load_weights), map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    print(f"It was trained {ckpt['epochs']} EPOCHS")
    
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