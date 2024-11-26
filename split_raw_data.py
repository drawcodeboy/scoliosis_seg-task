# This code loads the CT slices (grayscale images) of the brain-window for each subject in ct_scans folder then saves them to
# one folder (data\image).
# Their segmentation from the masks folder is saved to another folder (data\label).

import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import nibabel as nib
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--only", action='store_true')

args = parser.parse_args()

def window_ct (ct_scan, w_level=40, w_width=120):
    '''
    CT 스캔은 HU 값으로 표현되는데 HU 값의 정의는 다음과 같다.
    HU 값의 정의:
        HU는 물질의 밀도를 기준으로 정의되며, 물의 밀도를 0 HU로 기준화하고, 
        공기는 -1000 HU, 뼈 등 고밀도 물질은 수백에서 수천 HU에 이르는 값을 가집니다.
    일반적인 범위:
        공기: 약 -1000 HU
        지방: 약 -100 ~ -50 HU
        물: 0 HU
        연조직: 약 20 ~ 80 HU
        뼈: 약 700 HU 이상 (일부 고밀도 뼈는 3000 HU 이상 가능)
        금속 임플란트: 2000 HU 이상 (아주 높은 경우 4000 HU를 초과할 수도 있음)
    이때, 특정 HU 범위를 강조시키기 위해서 윈도우 레벨, 윈도우 폭 개념이 사용된다.
    아래의 경우 40이 중점이며, 범위는 120으로 [-20, 100] 범위의 HU 값을 강조시킨다.
    '''
    w_min = w_level - w_width / 2 # 최솟값
    w_max = w_level + w_width / 2 # 최댓값
    num_slices=ct_scan.shape[2]
    for s in range(num_slices):
        slice_s = ct_scan[:,:,s]
        
        # 0 ~ 255로 Scaling
        slice_s = (slice_s - w_min)*(255/(w_max-w_min)) #or slice_s = (slice_s - (w_level-(w_width/2)))*(255/(w_width))
        slice_s[slice_s < 0] = 0
        slice_s[slice_s > 255] = 255
        #slice_s=np.rot90(slice_s)
        
        ct_scan[:,:,s] = slice_s

    return ct_scan

numSubj = 82
new_size = (640, 640)
window_specs=[40,120] #Brain window
currentDir = Path(os.getcwd())
datasetDir = rf"data/physionet.org/files/ct-ich/1.3.1"

# Reading labels
hemorrhage_diagnosis_df = pd.read_csv(
    Path(rf"data/physionet.org/files/ct-ich/1.3.1/hemorrhage_diagnosis_raw_ct.csv"))
hemorrhage_diagnosis_array = hemorrhage_diagnosis_df.values

# reading images
extract_method = 'only' if args.only else 'all' # only = Scans Contain Hemorrhage Only, all= All Scans
train_path = f'data/physionet.org/files/ct-ich/1.3.1/data_{extract_method}'
image_path = f'{train_path}/image'
label_path = f'{train_path}/label'
if not os.path.exists(train_path):
    os.mkdir(train_path)
    os.mkdir(image_path)
    os.mkdir(label_path)

counterI = 0
interval = 49

for sNo in range(0+interval, numSubj+interval):
    if sNo>58 and sNo<66: #no raw data were available for these subjects
        next
    else:
        #Loading the CT scan
        ct_dir_subj = Path(datasetDir,'ct_scans', "{0:0=3d}.nii".format(sNo))
        ct_scan_nifti = nib.load(str(ct_dir_subj))
        ct_scan = ct_scan_nifti.get_fdata() # -1024 ~ 3000 사이의 값으로 구성
        # print(np.min(ct_scan), np.max(ct_scan))
        ct_scan = window_ct(ct_scan, window_specs[0], window_specs[1]) # 0 ~ 255로 처리 및 근육 부분 강조
        # print(np.min(ct_scan), np.max(ct_scan))
        # sys.exit()
        
        #Loading the masks
        masks_dir_subj = Path(datasetDir,'masks', "{0:0=3d}.nii".format(sNo))
        masks_nifti = nib.load(str(masks_dir_subj))
        masks = masks_nifti.get_fdata()
        
        idx = hemorrhage_diagnosis_array[:, 0] == sNo # 해당 CT Scan의 환자에 관한 rows
        sliceNos = hemorrhage_diagnosis_array[idx, 1]
        NoHemorrhage = hemorrhage_diagnosis_array[idx, 7]
        
        if sliceNos.size!=ct_scan.shape[2]:
            print('Warning: the number of annotated slices does not equal the number of slices in NIFTI file!')

        for sliceI in range(0, sliceNos.size):
            if args.only == True and NoHemorrhage[sliceI] == 1: # Image 내에 뇌출혈이 없으면?
                continue
            # Saving the a given CT slice
            x = cv2.resize(ct_scan[:,:,sliceI], new_size)
            cv2.imwrite(f"{image_path}/scan{sNo}_slice{sliceI}.png", x)
            print(f"{image_path}/image_scan{sNo}_slice{sliceI}.png")

            # Saving the segmentation for a given slice
            segment_path = Path(masks_dir_subj,str(sliceNos[sliceI]) + '_HGE_Seg.jpg')
            x = cv2.resize(masks[:,:,sliceI], new_size)
            cv2.imwrite(f"{label_path}/mask_scan{sNo}_slice{sliceI}.png", x)
            counterI = counterI+1

print(f"total data: {counterI}")