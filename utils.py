# save_model_ckpt, save_loss_ckpt
import os
import torch
import numpy as np

def save_model_ckpt(model, scale, current_epoch, dir, train_loss_type):
    ckpt = {}
    ckpt['model'] = model.state_dict()
    ckpt['epochs'] = current_epoch
    
    loss_type = 'unified_loss' if train_loss_type == True else 'dice_loss'
    state_dict_name = f'SegFormer-{scale}-{current_epoch:03d}-{loss_type}.pth'
    
    try:
        torch.save(ckpt, os.path.join(dir, state_dict_name))
        print(f"Save Model @epoch: {current_epoch}")
    except:
        print(f"Can\'t Save Model @epoch: {current_epoch}")


def save_loss_ckpt(train_loss, val_loss, train_dir, val_dir, train_loss_type):
    loss_type = 'unified_loss' if train_loss_type == True else 'dice_loss'
    
    try:
        np.save(os.path.join(train_dir, f'total_train_{loss_type}.npy'), np.array(train_loss))
        print('Save Train Loss')
    except:
        print('Can\'t Save Train Loss')
    
    try:
        np.save(os.path.join(val_dir, f'total_val_{loss_type}.npy'), np.array(val_loss))
        print('Save Validation Loss')
    except:
        print('Can\'t Save Validation Loss')