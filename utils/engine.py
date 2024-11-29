import torch
from .metrics import get_metrics

import cv2
import numpy as np
import sys

def train_one_epoch(epoch, model, dataloader, optimizer, scheduler, loss_fn, device):
    model.train()
    
    total_loss = 0.
    for batch_idx, (images, targets) in enumerate(dataloader, start=1):
        optimizer.zero_grad()
        
        images, targets = images.to(device), targets.to(device)
        
        outputs = model(images)
        
        loss = loss_fn(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        # output = outputs[0].cpu().detach().permute(1, 2, 0).numpy()
        # output = (output * 255.).astype(np.uint8)
        # cv2.imwrite('./test.jpg', output)
        
        
        total_loss += loss
        print(f"\rTraining: {100*batch_idx/len(dataloader):.2f}%, Loss: {total_loss/(batch_idx+1):.6f}, LR: {scheduler.get_last_lr()[0]:.8f}", end="")
    print()
        
    return (total_loss/len(dataloader)).detach().cpu().numpy() # One Epoch Mean Loss

@torch.no_grad()
def evaluate(epoch, model, dataloader, loss_fn, scheduler, device):
    model.eval()
    
    metrics_li = ['IoU', 'Dice', 'Precision', 'Recall']
    metrics = {}
    for key in metrics_li:
        metrics[key] = []

    reduction = 'mean'
    
    total_loss = 0.
    
    for batch_idx, (images, targets) in enumerate(dataloader, start=1):
        images, targets = images.to(device), targets.to(device)
        
        outputs = model(images)
        
        loss = loss_fn(outputs, targets)
                
        '''
        idx = 3
        
        image = images[idx].cpu().detach().permute(1, 2, 0).numpy()
        image = (image * 255.).astype(np.uint8)
        cv2.imwrite('./test_image_.jpg', image) 
        
        output = outputs[idx].cpu().detach().permute(1, 2, 0).numpy()
        output = (output * 255.).astype(np.uint8)
        cv2.imwrite('./test_output.jpg', output)
        
        target = targets[idx].cpu().detach().permute(1, 2, 0).numpy()
        target = (target * 255.).astype(np.uint8)
        cv2.imwrite('./test_target.jpg', target)

        sys.exit()
        '''
        
        metrics_dict = get_metrics(outputs, targets, metrics_li)
        for key in metrics_li:
            metrics[key].extend(metrics_dict[key])
        
        total_loss += loss
        print(f"\rEvaluate: {100*batch_idx/len(dataloader):.2f}%", end='')
    print()
    
    print("\t* ", end='')
    for key in metrics_li:
        metrics[key] = sum(metrics[key])/len(metrics[key])
        print(f"{key}: {metrics[key]:.4f}", end=" | ")
        
    mean_loss = (total_loss/len(dataloader)).detach().cpu().numpy()
    
    if scheduler is not None:
        scheduler.step(mean_loss)
    
    print(f"loss: {mean_loss:.6f}")
    return mean_loss