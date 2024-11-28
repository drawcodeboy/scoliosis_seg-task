import time
import argparse
import os, sys

from dataloader import load_dataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn

from utils import *

from models.nets import load_model
from models.loss_fn import load_loss_fn

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    # GPU
    parser.add_argument("--use-cuda", action='store_true')
    parser.add_argument("--dist", action='store_true')
    
    # Dataset
    parser.add_argument("--dataset", default='scoliosis')
    parser.add_argument("--mode", default='train')
    
    # Model
    parser.add_argument("--model", default='SegFormer-B0')
    parser.add_argument("--num-classes", type=int, default=1, help="Num of Classes without Background")
    
    # Loss function
    parser.add_argument("--loss-fn", default='dice')
    
    # Hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    
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
    print(f"loss function : {args.loss_fn}")
    print("######################")

def main(args):
    device = 'cpu' # Default Device
    if args.use_cuda and torch.cuda.is_available():
        device = 'cuda'
    
    print_info(device, args)
    
    # Dataset
    train_ds = load_dataset(dataset=args.dataset, mode=args.mode)
    val_ds = load_dataset(dataset=args.dataset, mode='val')
    
    image, mask = train_ds[0]
    image = (255. * image.permute(1, 2, 0)).detach().cpu().numpy().astype(np.uint8)
    mask = (255. * mask.permute(1, 2, 0)).detach().cpu().numpy().astype(np.uint8)
    cv2.imwrite("./temp_image.jpg", image)
    cv2.imwrite("./temp_mask.jpg", mask)
    
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size)
    
    # Model
    model_name = args.model.split('-')[0].lower()
    model_scale = args.model.split('-')[1].lower()
        
    model = load_model(model_name=model_name, scale=model_scale, num_classes=args.num_classes).to(device)
    
    if args.dist == True:
        '''
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        torch.distributed.init_process_group(backend='ncll', 
                                             world_size=2, 
                                             rank=0,)
        print(torch.distributed.get_rank())
        
        model = nn.parallel.DistributedDataParallel(model)
        torch.distributed.destroy_process_group()
        '''
        # model = nn.parallel.DistributedDataParallel(model)
        model = nn.DataParallel(model, device_ids=[0, 1], output_device=0)
    
    # Loss function
    loss_fn = load_loss_fn(loss_fn=args.loss_fn)
    
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
    
    total_train_start_time = time.time()
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
        val_loss = evaluate(current_epoch, model, val_dl, loss_fn, scheduler, device)
        val_time = int(time.time() - start_time)
        print(f"Validation Time: {val_time//60:02d}m {val_time%60:02d}s")
        
        # Save Model (Minimum Validation Loss)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            save_model_ckpt(model, args.model, current_epoch, f"{args.save_weights_dir}/{args.dataset}")
        
        total_train_loss.append(train_loss)
        total_val_loss.append(val_loss)
        save_loss_ckpt(args.dataset, total_train_loss, total_val_loss, args.save_train_loss_dir, args.save_val_loss_dir, args.model)
    
    total_train_time = int(time.time() - total_train_start_time)
    print(f"Total Training Time: {total_train_time//60:02d}m {total_train_time%60:02d}s")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training Segmentation Algorithm', parents=[get_args_parser()])
    
    args = parser.parse_args()
    main(args)