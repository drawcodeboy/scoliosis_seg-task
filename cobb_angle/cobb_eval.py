from cobb_angle_detector import CobbAngleDetector

import argparse
import numpy as np
import os, sys, time
import cv2

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    
    parser.add_argument("--test-dir", default="mask_test")
    parser.add_argument("--expr", default='None') # 'Centroid', 'Curve'
    
    return parser

def main(args):
    
    image_list = os.listdir(args.test_dir)
    
    sample_cnt = 0
    errors_li = []
    
    for orig_path in image_list:
        if orig_path[:4] != 'orig':
            continue
        
        sample_cnt += 1
        
        mask_path = 'mask' + orig_path[4:]
        
        orig_path = os.path.join(args.test_dir, orig_path)
        mask_path = os.path.join(args.test_dir, mask_path)
        
        orig_image = cv2.imread(orig_path, cv2.IMREAD_COLOR)
        binary_mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        
        cobb_detector = CobbAngleDetector(orig_image=orig_image,
                                          binary_mask=binary_mask,
                                          n_segments=20,
                                          order=8,
                                          deg=7,
                                          expr=args.expr)
        
        try:
            start_time = time.time()
            
            cobb_error = cobb_detector()
            
            work_time = time.time() - start_time
            
            errors_li.append(cobb_error)
            print(f"\r{int(work_time)//60}m{int(work_time)%60}s Sample[{sample_cnt:03d}] is processing... {cobb_error}", end='')
            
        except:
            print(f"\nError @ {orig_path}")
        
    
    print()
    print(f"Test sample count: {sample_cnt:03d}")
    
    errors_li = np.array(errors_li)
    errors_mean = np.mean(errors_li, axis=0)
    
    print(f"Error Mean: {errors_mean}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    
    args = parser.parse_args()
    
    main(args)