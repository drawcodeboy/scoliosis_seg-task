# save_model_ckpt, save_loss_ckpt
import os
import torch
import numpy as np

def save_model_ckpt(model, model_name, current_epoch, dir):
    ckpt = {}
    ckpt['model'] = model.state_dict()
    ckpt['epochs'] = current_epoch
    
    # 이전 Epoch를 제거하기 위함
    prev_pths = os.listdir(dir)
    
    for prev_pth in prev_pths:
        if not prev_pth.endswith(".pth"): # 확장자 파일이 아니면 건너뛰기
            continue
        
        prev_parsed = prev_pth.split('-')
        prev_model = prev_parsed[0]
        prev_scale = prev_parsed[1]
        prev_epoch = prev_parsed[2].split('.')[0]
        
        prev_epoch = int(prev_epoch)
        
        if (prev_model == model_name.split('-')[0]) and (prev_scale == model_name.split('-')[1]):
            # 같은 모델에 같은 스케일만 제거
            if prev_epoch < current_epoch:
                # 현재 epoch 수보다 적으면 제거
                os.remove(os.path.join(dir, prev_pth))
    
    state_dict_name = f'{model_name}-{current_epoch:03d}.pth'
    
    try:
        torch.save(ckpt, os.path.join(dir, state_dict_name))
        print(f"Save Model @epoch: {current_epoch}")
    except:
        print(f"Can\'t Save Model @epoch: {current_epoch}")


def save_loss_ckpt(dataset, train_loss, val_loss, train_dir, val_dir, model_name):
    
    try:
        np.save(os.path.join(train_dir, f'{dataset}/total_train_{model_name}.npy'), np.array(train_loss))
        print('Save Train Loss')
    except:
        print('Can\'t Save Train Loss')
    
    try:
        np.save(os.path.join(val_dir, f'{dataset}/total_val_{model_name}.npy'), np.array(val_loss))
        print('Save Validation Loss')
    except:
        print('Can\'t Save Validation Loss')