import cv2
import argparse
import numpy as np
from skimage.morphology import skeletonize, thin
import sys, os

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    
    parser.add_argument('--image-path', default="test_mask_0001.jpg")
    parser.add_argument('--mask-path', default="real_mask_0001.jpg")
    
    return parser

def apply_skeletionize(binary_mask):
    gray = cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY)
    skel = thin(gray)
    print(skel.dtype, skel.shape)
    
    # skel_image = np.zeros_like(real_mask)
    binary_mask[skel==True] = [0, 0, 255]
    
    return binary_mask

def detect_line(binary_mask, line_num=10):
    # Y축 min, max 찾기
    y_coords = np.where(binary_mask.any(axis=1))[0]
    ymin, ymax = y_coords.min(), y_coords.max()
    
    factor = int((ymax-ymin)/(line_num))
    
    cut_points = [(ymin + i * factor) for i in range(1, line_num)]
    
    spine_lines = []

    for y_coord in cut_points:
        row = binary_mask[y_coord]
        
        rising_edges = np.where((row[:-1] == 0) & (row[1:] == 255))[0]
        rising_edge = rising_edges[0]
        
        falling_edges = np.where((row[:-1] == 255) & (row[1:] == 0))[0]
        falling_edge = falling_edges[0]
        
        spine_lines.append([(rising_edge, y_coord), (falling_edge, y_coord)])
        
    for pt1, pt2 in spine_lines:
        cv2.line(binary_mask, pt1, pt2, color=(0, 255, 0), thickness=2)
    
    return binary_mask
    
def main(args):
    mask = cv2.imread(args.image_path, cv2.IMREAD_COLOR)
    binary_mask = cv2.imread(args.mask_path, cv2.IMREAD_COLOR)
    
    # Filtering...
    binary_mask[binary_mask >= 128] = 255
    binary_mask[binary_mask < 128] = 0
        
    cv2.imshow('original', mask)
    
    canny_mask = apply_canny(mask, args.canny_t1, args.canny_t2)
    # cv2.imshow('Canny', canny_mask)
    
    # cv2.imshow('Binary Mask', binary_mask)
    # cv2.imshow('Erosion', erode_mask)
    
    # skel_image = apply_skeletionize(binary_mask)
    # cv2.imshow('Skeletonize', skel_image)
    
    divide_mask = detect_line(binary_mask, 8)
    cv2.imshow('Divide Mask', divide_mask)
        
    cv2.waitKeyEx(0)
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Line Detection', parents=[get_args_parser()])
    
    args = parser.parse_args()
    main(args)