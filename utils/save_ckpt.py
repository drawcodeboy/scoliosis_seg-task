# save_model_ckpt, save_loss_ckpt
import os
import torch
import numpy as np

def save_model_ckpt(model, model_name, current_epoch, dir):
    ckpt = {}
    ckpt['model'] = model.state_dict()
    ckpt['epochs'] = current_epoch
    
    state_dict_name = f'{model_name}-{current_epoch:03d}.pth'
    
    try:
        torch.save(ckpt, os.path.join(dir, state_dict_name))
        print(f"Save Model @epoch: {current_epoch}")
    except:
        print(f"Can\'t Save Model @epoch: {current_epoch}")


def save_loss_ckpt(train_loss, val_loss, train_dir, val_dir, model_name):
    
    try:
        np.save(os.path.join(train_dir, f'total_train_{model_name}.npy'), np.array(train_loss))
        print('Save Train Loss')
    except:
        print('Can\'t Save Train Loss')
    
    try:
        np.save(os.path.join(val_dir, f'total_val_{model_name}.npy'), np.array(val_loss))
        print('Save Validation Loss')
    except:
        print('Can\'t Save Validation Loss')