import cv2
from PIL import Image, ImageDraw, ImageFont
import argparse
import numpy as np
from skimage.morphology import skeletonize, thin
from scipy.interpolate import CubicSpline
import sys, os
import math
from typing import Optional
import time

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    
    parser.add_argument('--img-num', type=str, default="0854")
    parser.add_argument('--save-all', action='store_true')
    
    return parser

def apply_skeletionize(binary_mask):
    gray = cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY)
    skel = thin(gray)
    print(skel.dtype, skel.shape)
    
    # skel_image = np.zeros_like(real_mask)
    binary_mask[skel==True] = [0, 0, 255]
    
    return binary_mask
    
class CobbAngleDetector():
    def __init__(self, 
                 orig_image, 
                 binary_mask,
                 name:Optional[str] = None,
                 n_segments:int = 20, 
                 order:int = 6, 
                 deg:int = 6, 
                 expr:str = 'None', 
                 format:str = 'visual'):
        '''
            Args:
                - real_mask: 원래 Image에서 Mask에 따라 척추 부분만 Segmentation한 이미지
                - binary_mask: 모델의 결과값
                - n_segments: 척추 마스크를 몇 개의 구간으로 나눌 것인가
                - order: clustering optimization 횟수, order-1번 수행한다. (중요), order=1 => init_centroids
                - deg: curve fitting (polyfit) 차수
                - expr: 
                    'None': 실험이 아님, curve를 구하는 것이 목적
                    'Centroid': Centroid Optimization Experiment
                    'Curve': Curve Fitting Optimization Experiment
                - format: 실험의 모든 과정을 시각화 할 것인가, 저장할 것인가
                    - 'visual'
                    - 'save'
                    - None: 아무 것도 하지 않음.
        '''
        _, binary_mask = cv2.threshold(binary_mask, 128, 255, cv2.THRESH_BINARY)
        self.binary_mask = self.preprocess(binary_mask)
        self.orig_image = orig_image
        self.real_mask = CobbAngleDetector.get_mask(orig_image, self.binary_mask)
        self.n_segments = n_segments
        self.order = order
        self.deg = deg
        
        self.expr = expr
        
        self.format = format
        self.name = name
    
    @staticmethod
    def get_mask(orig_image, binary_mask):
        real_mask = orig_image.copy()
        real_mask[binary_mask == 0] = 0
        
        return real_mask
        
    def __call__(self):
        '''
            Return:
                - center_opt_error: List[float], len(order-1), Center Optimization을 하면서 변경된 거리만큼의 오차
                - 
        '''
        cen_all = []
        
        if self.format is not None: self.visualize(None, None, 0, origin=True)
        
        # 1. Init Centroids
        start_time = time.time()
        centroids, edges = self.init_centroids() 
        during_time = time.time() - start_time
        print(f"1: {int(during_time)//60:02d}m {int(during_time)%60:02d}s")
        if self.format is not None: self.visualize(centroids, edges, 1)
        
        centroids = centroids[1:-1]
        edges = edges[1:-1]
        cen_all.append(centroids)
        
        start_time = time.time()
        for order in range(2, self.order+1):
            centroids, edges = self.update_centroids(centroids) # 2. Update Centroids (4times)
            cen_all.append(centroids)
            # print(f"\rCalculate centroids: {order:02d}", end="")
        during_time = time.time() - start_time
        print(f"2: {int(during_time)//60:02d}m {int(during_time)%60:02d}s")
        
        if self.format is not None: self.visualize(centroids, edges, 2)
        errors = self.get_centroids_error(cen_all)
        
        if self.expr == 'Centroid':
            return errors
        
        # 3. Find Extreme points
        start_time = time.time()
        tl_pt, tr_pt, bl_pt, br_pt = self.find_extremes(top_grad = 1/2, bottom_grad = 1/3)
        during_time = time.time() - start_time
        print(f"3: {int(during_time)//60:02d}m {int(during_time)%60:02d}s")
        
        centroids = np.concatenate((np.array([[(tl_pt[0]+tr_pt[0])//2, (tl_pt[1]+tr_pt[1])//2]]),
                                    centroids,
                                    np.array([[(bl_pt[0]+br_pt[0])//2, (bl_pt[1]+br_pt[1])//2]])), 
                                    axis=0)
        edges = np.concatenate((np.array([[tl_pt, tr_pt]]),
                                edges,
                                np.array([[bl_pt, br_pt]])),
                                axis=0)
        if self.format is not None: self.visualize(centroids, edges, 3, final=True)
        
        '''
        if self.expr == 'Curve':
            errors = []
            for deg_ in [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]:
                curve, poly_func = self.curve_fitting(centroids, deg_)
                errors.append(self.get_curve_error(curve, centroids))
            
            return errors
        '''
        
        # 4. Curve Fitting
        # curve, poly_func = self.curve_fitting(centroids, self.deg)
        start_time = time.time()
        # curve, poly_func = self.curve_fitting(centroids, 10)
        curve, spline_func = self.spline_interpolation(centroids)
        during_time = time.time() - start_time
        print(f"4: {int(during_time)//60:02d}m {int(during_time)%60:02d}s")
        
        if self.expr == 'Curve':
            curve, spline_func = self.spline_interpolation(centroids)
            errors = self.get_curve_error(curve, centroids)
            return errors
        
        if self.format is not None: self.visualize(curve, None, 4, center_pt=False, final=True)
        
        # 5. Get Cobb Angle
        start_time = time.time()
        max_line_pt, min_line_pt, cobb_angle = self.get_cobb_angle(curve, interval=10)
        during_time = time.time() - start_time
        print(f"5: {int(during_time)//60:02d}m {int(during_time)%60:02d}s")
        if self.format is not None: self.visualize(curve, 
                                                   None, 
                                                   5, 
                                                   center_pt=False, 
                                                   final=True, 
                                                   cobb_angle_line=[max_line_pt, min_line_pt], 
                                                   cobb_angle=cobb_angle)
        
        return cobb_angle

    def get_centroids_error(self, cen_all):
        cen_all = np.array(cen_all, dtype=np.float32)
        
        errors = []
        
        for idx in range(len(cen_all)-1):
            # print(f"{idx+1}-{idx+2} ERROR: {np.sum(np.abs(cen_all[idx] - cen_all[idx+1])):.6f}")
            errors.append(np.sum(np.abs(cen_all[idx] - cen_all[idx+1])))
        
        return errors
    
    def get_curve_error(self, curve, centroids):
        error = 0
        
        cen_mid_pts = []
        
        for idx in range(len(centroids)-1):
            mid_x = ((centroids[idx][0] + centroids[idx+1][0]) / 2)
            mid_y = ((centroids[idx][1] + centroids[idx+1][1]) / 2)
            
            cen_mid_pts.append([mid_x, mid_y])
            
        
        centroids = np.concatenate((centroids, cen_mid_pts), axis=0)
        
        for cen_pt in centroids:
            cen_x, cen_y = cen_pt[0], cen_pt[1]
            
            min_dist = 1e9
            temp_err = 0
            
            for cur_pt in curve:
                cur_x, cur_y = cur_pt[0], cur_pt[1]
                
                if abs(cur_y-cen_y) < min_dist:
                    min_dist = abs(cur_y-cen_y)
                    temp_err = abs(cen_x-cur_x)
                
            error += temp_err
        
        return error
    
    def preprocess(self, binary_mask):
        # Contour 추출을 위해서 Grayscale로 변환
        binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY)
        
        
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        largest_contour = max(contours, key=cv2.contourArea)

        new_mask = np.zeros_like(binary_mask)

        cv2.drawContours(new_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        
        new_mask = cv2.cvtColor(new_mask, cv2.COLOR_GRAY2BGR) # 다시 3 Channels로 변환
        
        return new_mask
        
    def init_centroids(self):
        # Y축 min, max 찾기
        y_coords = np.where(self.binary_mask.any(axis=1))[0]
        
        ymin, ymax = y_coords.min(), y_coords.max()
        
        factor = (ymax-ymin)/(self.n_segments)
        
        cut_points = [int(ymin + i * factor) for i in range(1, self.n_segments)]
        
        spine_lines = []
        centroids = []
        
        for y_coord in cut_points:
            row = self.binary_mask[y_coord]
            
            rising_edges = np.where((row[:-1] == 0) & (row[1:] == 255))[0]
            rising_edge = rising_edges[0]
            
            falling_edges = np.where((row[:-1] == 255) & (row[1:] == 0))[0]
            falling_edge = falling_edges[0]
            
            centroids.append([(rising_edge+falling_edge)//2, y_coord])
            
            spine_lines.append([(rising_edge, y_coord), (falling_edge, y_coord)])
        
        return centroids, spine_lines
    
    def get_edge_pts(self, cen_x, cen_y, degree):
        if degree == 90: # 90도 일 때, 따로 check -> tan(90) = inf라서
            pass
            
        rad = np.deg2rad(degree)
        coef = np.tan(rad)
        bias = cen_y - coef*cen_x
        
        # Left-Side
        edge_pt1 = None
        for x in range(cen_x, 1, -1):
            now_y = coef * x + bias
            next_y = coef * (x-1) + bias
            
            if now_y < 0 or next_y < 0 or now_y >= self.binary_mask.shape[0] or next_y >= self.binary_mask.shape[0]:
                break
            
            if (self.binary_mask[int(now_y), x] != self.binary_mask[int(next_y), x-1]).any():
                edge_pt1 = (x, now_y)
                
        # Right-Side
        edge_pt2 = None
        for x in range(cen_x, self.binary_mask.shape[1]-1):
            now_y = coef * x + bias
            next_y = coef * (x+1) + bias
            
            if now_y < 0 or next_y < 0 or now_y >= self.binary_mask.shape[0] or next_y >= self.binary_mask.shape[0]:
                break
            
            if (self.binary_mask[int(now_y), x] != self.binary_mask[int(next_y), x+1]).any():
                edge_pt2 = (x, now_y)
        
        return edge_pt1, edge_pt2

    def update_centroids(self, centroids):
        new_centroids = []
        edges = []
        
        for idx, (cen_x, cen_y) in enumerate(centroids, start=1):
            cen_x, cen_y = int(cen_x), int(cen_y)
            # 180도 돌면서 가장 짧은 거리 찾기
            # 기울기가 angle인 선(y = ax + b)구하기
            shortest_dist = 1e9
            min_edge_pt1, min_edge_pt2 = None, None
            
            # 0~60, 120~179
            degree_range = [i for i in range(0, 60+1)] + [i for i in range(120, 180)]
            for degree in degree_range:
                if degree == 90:
                    continue
                
                edge_pt1, edge_pt2 = self.get_edge_pts(cen_x, cen_y, degree)
                dist = np.sqrt((edge_pt2[1]-edge_pt1[1])**2 + (edge_pt2[0]-edge_pt1[0])**2)
                if dist < shortest_dist:
                    shortest_dist = dist
                    min_edge_pt1, min_edge_pt2 = edge_pt1, edge_pt2
            
            new_cen_x = (min_edge_pt1[0] + min_edge_pt2[0]) / 2
            new_cen_y = (min_edge_pt1[1] + min_edge_pt2[1]) / 2
            
            new_centroids.append([new_cen_x, new_cen_y])
            edges.append([min_edge_pt1, min_edge_pt2])
        
        return new_centroids, edges
    
    def find_extremes(self, top_grad: float = 1/2, bottom_grad: float = 1/3):
        
        tl_flag, tr_flag, bl_flag, br_flag = (False for _ in range(0, 4))
        
        for bias in range(-self.binary_mask.shape[0], self.binary_mask.shape[0]): # 각 선
            if tr_flag: break
            for x in range(0, self.binary_mask.shape[0]): # 각 선에 따른 x 좌표
                y = top_grad * x + bias
                if y < 0 or y >= self.binary_mask.shape[0]:
                    continue
                if 255 in self.binary_mask[int(y), x]:
                    tr_pt = (x, int(y))
                    # print(f"Top Right: {tr_pt}")
                    tr_flag = True
                    break
        
        for bias in range(-self.binary_mask.shape[0], self.binary_mask.shape[0]): # 각 선
            if tl_flag: break
            for x in range(0, self.binary_mask.shape[0]): # 각 선에 따른 x 좌표
                y = -top_grad * x + bias
                if y < 0 or y >= self.binary_mask.shape[0]:
                    continue
                if 255 in self.binary_mask[int(y), x]:
                    tl_pt = (x, int(y))
                    # print(f"Top Left: {tl_pt}")
                    tl_flag = True
                    break
        
        for bias in range(self.binary_mask.shape[0], -self.binary_mask.shape[0], -1): # 각 선
            if bl_flag: break
            for x in range(0, self.binary_mask.shape[0]): # 각 선에 따른 x 좌표
                y = bottom_grad * x + bias
                if y < 0 or y >= self.binary_mask.shape[0]:
                    continue
                if 255 in self.binary_mask[int(y), x]:
                    bl_pt = (x, int(y))
                    # print(f"Top Right: {bl_pt}")
                    bl_flag = True
                    break
        
        for bias in range(self.binary_mask.shape[0], -self.binary_mask.shape[0], -1): # 각 선
            if br_flag: break
            for x in range(0, self.binary_mask.shape[0]): # 각 선에 따른 x 좌표
                y = -bottom_grad * x + bias
                if y < 0 or y >= self.binary_mask.shape[0]:
                    continue
                if 255 in self.binary_mask[int(y), x]:
                    br_pt = (x, int(y))
                    # print(f"Bottom Right: {br_pt}")
                    br_flag = True
                    break
        
        return tl_pt, tr_pt, bl_pt, br_pt
    
    def curve_fitting(self, centroids, deg):
        # 여기선 y축을 x축으로 전환해서 보자.
        
        centroids = np.array(centroids)
        
        x = centroids[:, 1] # Actually, y
        y = centroids[:, 0] # Actually, x
        
        poly_func = np.polynomial.Polynomial.fit(x, y, deg) # shape: (deg+1,)
        
        x_ = np.linspace(centroids[0, 1], centroids[-1, 1], num=100)
        y_ = poly_func(x_)
        
        curve = np.concatenate((y_.reshape(-1, 1), x_.reshape(-1, 1)), axis=1)
        # print(curve.shape)
        
        return curve, poly_func

    def spline_interpolation(self, centroids):
        centroids = np.array(centroids)
        
        x = centroids[:, 1] # Actually, y
        y = centroids[:, 0] # Actually, x
        
        cs = CubicSpline(x, y)
        
        x_ = np.linspace(centroids[0, 1], centroids[-1, 1], num=100)
        y_ = cs(x_)
        
        curve = np.concatenate((y_.reshape(-1, 1), x_.reshape(-1, 1)), axis=1)
        
        return curve, cs
    
    def get_cobb_angle(self, curve, interval:int =10):
        '''
            - 가장 기울기가 크고 작은 두 직선을 구하고, Cobb Angle을 Return
        '''
        
        max_grad, min_grad = 1e9, -1e9
        max_bias, min_bias = None, None
        max_line, min_line = None, None
        
        for idx in range(len(curve)-interval):
            x1, y1 = curve[idx][0], curve[idx][1]
            x2, y2 = curve[idx+interval][0], curve[idx+interval][1]
            
            grad = (y2-y1)/(x2-x1)
            
            if grad > 0 and grad < max_grad:
                max_grad = grad
                max_bias = y1 - (grad * x1)
            
            if grad < 0 and abs(grad) < abs(min_grad):
                min_grad = grad
                min_bias = y1 - (grad * x1)
                
        max_line_pt = [(0, max_bias), (self.binary_mask.shape[1], max_grad*self.binary_mask.shape[1]+max_bias)]
        min_line_pt = [(0, min_bias), (self.binary_mask.shape[1], min_grad*self.binary_mask.shape[1]+min_bias)]
        
        # Get Gobb Angle
        theta = math.atan(abs((max_grad-min_grad) / (1+max_grad*min_grad)))
        
        cobb_angle = math.degrees(theta)
        
        cross_pt = self.line_intersection(max_line_pt[0],
                                          max_line_pt[1],
                                          min_line_pt[0],
                                          min_line_pt[1])
        
        if (max_grad * min_grad) > -1 and (max_grad * min_grad) < 0:
            # 두 기울기의 곱이 -1보다 크면 윗 각 부분이 90도를 넘는다는 의미이다.
            # 다만 이 곱이 양수가 되었다는 것은 두 선이 한 사분면에 있는 것이기 때문에
            # 무조건 윗 각이 90도보다 작아서 이 점을 유의해야 한다.
            cobb_angle = 180 - cobb_angle
        
        return max_line_pt, min_line_pt, cobb_angle

    def line_intersection(self, A, B, C, D):
        # 각 직선의 방정식 계수 계산
        a1 = B[1] - A[1]
        b1 = A[0] - B[0]
        c1 = a1 * A[0] + b1 * A[1]
        
        a2 = D[1] - C[1]
        b2 = C[0] - D[0]
        c2 = a2 * C[0] + b2 * C[1]
        
        # 교점 계산
        determinant = a1 * b2 - a2 * b1
        
        if determinant == 0:
            # 직선이 평행하거나 일치함
            return None  # 교점 없음
        else:
            x = (b2 * c1 - b1 * c2) / determinant
            y = (a1 * c2 - a2 * c1) / determinant
            return int(x), int(y)
        
    def get_extra_angle(self, pt):
        grad = (pt[1][1]-pt[0][1])/(pt[1][0]-pt[0][0]) # (y2-y1)/(x2-x1)
        theta = math.atan(abs(grad))
        
        extra_angle = math.degrees(theta)
        return extra_angle

    def visualize(self, centroids, edges, order, center_pt=True, final=False, cobb_angle_line=None, origin=False, cobb_angle=None):
        if origin == True:
            if self.format == 'visual':
                cv2.imshow(f'order: {order:02d}', self.orig_image)
            elif self.format == 'save':
                cv2.imwrite(rf'cobb_angle\cobb_test\cobb_{self.name}_{order:02d}.jpg', self.orig_image)
            return
        
        if order == 5:
            mask = self.orig_image.copy()
        else:
            mask = self.real_mask.copy()
        
        if edges is not None:
            for pt1, pt2 in edges:
                cv2.line(mask, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), color=(0, 255, 0), thickness=2)
        
        if center_pt and centroids is not None:
            for x, y in centroids:
                cv2.line(mask, (int(x), int(y)), (int(x), int(y)), color=(0, 0, 255), thickness=5)
        
        if final: # draw curve
            for idx in range(len(centroids)-1):
                cv2.line(mask, (int(centroids[idx][0]), int(centroids[idx][1])), 
                            (int(centroids[idx+1][0]), int(centroids[idx+1][1])),
                            color=(0, 0, 255), thickness=2)
                
        if cobb_angle_line is not None:
            cross_pt = self.line_intersection(cobb_angle_line[0][0],
                                              cobb_angle_line[0][1],
                                              cobb_angle_line[1][0],
                                              cobb_angle_line[1][1])
            extra_angle = self.get_extra_angle(cobb_angle_line[1])
            
            cv2.ellipse(mask, cross_pt, (30, 30), -(cobb_angle+extra_angle), 0, cobb_angle, (0, 255, 255), thickness=2)
            
            for pts in cobb_angle_line:
                cv2.line(mask, (int(pts[0][0]), int(pts[0][1])), (int(pts[1][0]), int(pts[1][1])), color=(255, 0, 0), thickness=2)
                
            cv2.line(mask, cross_pt, cross_pt, color=(0, 255, 255), thickness=5)
            
            color_coverted = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(color_coverted)
            
            draw = ImageDraw.Draw(pil_image)
            font = ImageFont.truetype("arial.ttf", 35)
            draw.text((cross_pt[0]-9, cross_pt[1]-70), "θ", font=font, fill=(255, 255, 0))
            
            font = ImageFont.truetype("arial.ttf", 20)
            draw.text((500, 10), f"θ: {cobb_angle:.4f}°", font=font, fill=(255, 255, 0))
            
            mask = np.array(pil_image)
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
            
        
        if self.format == 'visual':
            cv2.imshow(f'order: {order:02d}', mask)
        elif self.format == 'save':
            cv2.imwrite(rf'cobb_angle\cobb_test\cobb_{self.name}_{order:02d}.jpg', mask)

def main(args):
    
    if args.save_all:
        image_paths = []
        for image_path in os.listdir("mask_test"):
            if image_path[0] == 'o':
                image_path = os.path.join("mask_test", image_path)
                image_paths.append(image_path)
    else:
        image_paths = [rf"mask_test\orig_{args.img_num}.jpg"]
    
    for idx, image_path in enumerate(image_paths, start=1):
        img_num = image_path.split("\\")[-1][4:]
        mask_path = os.path.join("mask_test", f"mask{img_num}")
        
        mask = cv2.imread(image_path, cv2.IMREAD_COLOR)
        binary_mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
            
        img_num = img_num[1:5]
        print(f"{idx:03d}: {img_num}")
        print('Start Algorithm')
        
        cad = CobbAngleDetector(mask, 
                                binary_mask,
                                n_segments=20, 
                                order=6, 
                                expr='None', 
                                format='save',
                                name=img_num)
        
        cobb_angle = cad()
        print(f'Cobb Angle is {cobb_angle}')
        print(f"=" * 20)
    
    cv2.waitKeyEx(0)
    cv2.destroyAllWindows()
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Line Detection', parents=[get_args_parser()])
    
    args = parser.parse_args()
    main(args)
