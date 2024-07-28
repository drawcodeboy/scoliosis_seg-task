import time
import argparse
from dataloader import dataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from engine import *
from utils import *

from models.nets import SegFormer
from models.loss_fn import get_loss_fn

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    # GPU
    parser.add_argument("--use-cuda", action='store_true')
    
    # Dataset
    parser.add_argument("--dataset", default='scoliosis')
    parser.add_argument("--mode", default='train')
    parser.add_argument("--val_mode", default='valid')
    parser.add_argument("--data_dir", default='data/AIS.v1i.yolov8')
    
    # Model
    parser.add_argument("--scale", default='B0', help="MiT Scale of SegFormer")
    parser.add_argument("--num-classes", type=int, default=1, help="Num of Classes without Background")
    
    # Loss function
    parser.add_argument("--unified-loss-fn", action='store_true')
    
    # Hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    
    # Save Model Weights & Losses Dir
    parser.add_argument("--save_weights_dir", default="saved/weights")
    parser.add_argument("--save_train_loss_dir", default="saved/losses/train")
    parser.add_argument("--save_val_loss_dir", default="saved/losses/val")
    
    return parser

def print_info(device, args):
    print("######[Settings]######")
    print(f"device : {device}")
    print(f"dataset : {args.dataset}")
    print(f"mode : {args.mode}")
    print(f"epochs : {args.epochs}")
    print(f"lr : {args.lr}")
    print(f"batch size : {args.batch_size}")
    print(f"unified loss function : {args.unified_loss_fn}")
    print("######################")

def main(args):
    device = 'cpu' # Default Device
    if args.use_cuda and torch.cuda.is_available():
        device = 'cuda'
    
    print_info(device, args)
    
    # Dataset
    train_ds = dataset(dataset='scoliosis', data_dir=args.data_dir, mode=args.mode)
    val_ds = dataset(dataset='scoliosis', data_dir=args.data_dir, mode=args.val_mode)
    
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size)
    
    # Model
    model = SegFormer(num_classes=args.num_classes, phi=args.scale.lower()).to(device)
    
    # Loss function
    loss_fn = get_loss_fn(imbalance=args.unified_loss_fn)
    
    # Optimizer
    p = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(p, lr=args.lr)
    
    # Scheduler
    scheduler = ReduceLROnPlateau(optimizer,
                                  mode='min',
                                  factor=0.5,
                                  patience=5,
                                  min_lr=1e-7)
    
    # Record Loss
    total_train_loss = []
    total_val_loss = []
    
    # Check minimum validation loss for ckpt
    min_val_loss = 1000.
    
    # Train
    for current_epoch in range(0, args.epochs):
        current_epoch += 1
        print("======================================================")
        print(f"Epoch: [{current_epoch:03d}/{args.epochs:03d}]")
        
        # Training One Epoch
        start_time = int(time.time())
        train_loss = train_one_epoch(current_epoch, model, train_dl, optimizer, scheduler, loss_fn, device)
        train_time = int(time.time() - start_time)
        print(f"Training Time: {train_time//60:02d}m {train_time%60:02d}s")
        
        # and Validation
        start_time = int(time.time())
        val_loss = evaluate(current_epoch, model, val_dl, loss_fn, device)
        val_time = int(time.time() - start_time)
        print(f"Validation Time: {val_time//60:02d}m {val_time%60:02d}s")
        
        # Save Model (Minimum Validation Loss)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            save_model_ckpt(model, args.scale.upper(), current_epoch, args.save_weights_dir, args.unified_loss_fn)
        
        total_train_loss.append(train_loss)
        total_val_loss.append(val_loss)
    
    save_loss_ckpt(total_train_loss, total_val_loss, args.save_train_loss_dir, args.save_val_loss_dir, args.unified_loss_fn)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training SegFormer', parents=[get_args_parser()])
    
    args = parser.parse_args()
    main(args)