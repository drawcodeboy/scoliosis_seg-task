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
    w_min = w_level - w_width / 2
    w_max = w_level + w_width / 2
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
        ct_scan = ct_scan_nifti.get_fdata()
        ct_scan = window_ct(ct_scan, window_specs[0], window_specs[1])
        
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