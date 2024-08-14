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
    
class CobbAngleDetector():
    def __init__(self, orig_image, binary_mask, n_segments:int = 20, order:int = 5, deg:int = 7, expr:str = 'None'):
        '''
            Args:
                - real_mask: 원래 Image에서 Mask에 따라 척추 부분만 Segmentation한 이미지
                - binary_mask: 모델의 결과값
                - n_segments: 척추 마스크를 몇 개의 구간으로 나눌 것인가
                - order: clustering optimization 횟수
                - deg: curve fitting (polyfit) 차수
                - expr: 
                    'None': 실험이 아님, curve를 구하는 것이 목적
                    'Centroid': Centroid Optimization Experiment
                    'Curve': Curve Fitting Optimization Experiment
        '''
        self.orig_image = orig_image
        _, self.binary_mask = cv2.threshold(binary_mask, 128, 255, cv2.THRESH_BINARY)
        self.real_mask = CobbAngleDetector.get_mask(orig_image, self.binary_mask)
        self.n_segments = n_segments
        self.order = order
        self.deg = deg
        
        self.expr = expr
    
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
        
        centroids, edges = self.init_centroids() # 1. Init Centroids
        cen_all.append(centroids)
        # print(f"\rCalculate centroids: {1:02d}", end="")
        # self.visualize(centroids, edges, 1)
        
        for order in range(2, self.order+1):
            centroids, edges = self.update_centroids(centroids) # 2. Update Centroids (4times)
            cen_all.append(centroids)
            # print(f"\rCalculate centroids: {order:02d}", end="")

        # self.visualize(centroids, edges, 2)
        errors = self.get_centroids_error(cen_all)
        
        if self.expr == 'Centroid':
            return errors
        
        # 3. Find Extreme points
        tl_pt, tr_pt, bl_pt, br_pt = self.find_extremes(top_grad = 1/2, bottom_grad = 1/3)
        centroids = np.concatenate((np.array([[(tl_pt[0]+tr_pt[0])//2, (tl_pt[1]+tr_pt[1])//2]]),
                                    centroids,
                                    np.array([[(bl_pt[0]+br_pt[0])//2, (bl_pt[1]+br_pt[1])//2]])), 
                                    axis=0)
        edges = np.concatenate((np.array([[tl_pt, tr_pt]]),
                                edges,
                                np.array([[bl_pt, br_pt]])),
                                axis=0)
        # self.visualize(centroids, edges, 3, final=True)
        
        # 4. Curve Fitting
        curve, poly_func = self.curve_fitting(centroids, self.deg)
        self.visualize(curve, None, 4, center_pt=False, final=True)
        
        return centroids, edges

    def get_centroids_error(self, cen_all):
        cen_all = np.array(cen_all, dtype=np.float32)
        
        errors = []
        
        for idx in range(len(cen_all)-1):
            # print(f"{idx+1}-{idx+2} ERROR: {np.sum(np.abs(cen_all[idx] - cen_all[idx+1])):.6f}")
            errors.append(np.sum(np.abs(cen_all[idx] - cen_all[idx+1])))
        
        return errors
        
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
        
        # 첫 점과 끝 점 무시
        return centroids[1:-1], spine_lines[1:-1]
    
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
        
        poly_func = np.polynomial.Polynomial.fit(x, y, self.deg) # shape: (deg+1,)
        
        x_ = np.linspace(centroids[0, 1], centroids[-1, 1], num=100)
        y_ = poly_func(x_)
        
        curve = np.concatenate((y_.reshape(-1, 1), x_.reshape(-1, 1)), axis=1)
        # print(curve.shape)
        
        return curve, poly_func

    def visualize(self, centroids, edges, order, center_pt=True, final=False):
        mask = self.real_mask.copy()
        
        if edges is not None:
            for pt1, pt2 in edges:
                cv2.line(mask, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), color=(0, 255, 0), thickness=2)
        
        if center_pt:
            for x, y in centroids:
                cv2.line(mask, (int(x), int(y)), (int(x), int(y)), color=(0, 0, 255), thickness=5)
        
        if final:
            for idx in range(len(centroids)-1):
                cv2.line(mask, (int(centroids[idx][0]), int(centroids[idx][1])), 
                            (int(centroids[idx+1][0]), int(centroids[idx+1][1])),
                            color=(0, 0, 255), thickness=2)
        
        cv2.imshow(f'order: {order:02d}', mask)

def main(args):
    mask = cv2.imread(args.image_path, cv2.IMREAD_COLOR)
    binary_mask = cv2.imread(args.mask_path, cv2.IMREAD_COLOR)
    
    print(mask.dtype, binary_mask.dtype)
    
    cad = CobbAngleDetector(mask, binary_mask, 20, 2)
    
    centroids, edges = cad()
    
    cv2.waitKeyEx(0)
    cv2.destroyAllWindows()
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Line Detection', parents=[get_args_parser()])
    
    args = parser.parse_args()
    main(args)
