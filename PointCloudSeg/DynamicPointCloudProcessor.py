import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree as KDTree
from numba import njit, prange

import cv2
from sklearn.cluster import DBSCAN

@njit(parallel=True)
def fast_process_logic(local_pts, tx, ty, theta, map_res, h_max, w_max, sdf_map, dist_threshold):
    """
    Numba 加速核心：坐标变换 + SDF 查询
    """
    num_points = local_pts.shape[0]
    global_pts = np.empty((num_points, 2), dtype=np.float64)
    alpha = np.empty(num_points, dtype=np.float64)
    
    c, s = np.cos(theta), np.sin(theta)
    
    for i in prange(num_points):
        # 1. 坐标变换
        px, py = local_pts[i, 0], local_pts[i, 1]
        gx = px * c - py * s + tx
        gy = px * s + py * c + ty
        global_pts[i, 0], global_pts[i, 1] = gx, gy
        
        # 2. 栅格索引计算 (对应标准化后的 grid)
        ix = int(gx / map_res)
        iy = int(gy / map_res)
        
        # 3. 边界检查与 SDF 查询
        if 0 <= ix < w_max and 0 <= iy < h_max:
            dist_to_static = sdf_map[ix, iy] 
            alpha[i] = 0.7 if dist_to_static <= dist_threshold else 0.3
        else:
            alpha[i] = 0.3
            
    return global_pts, alpha

class DynamicPointCloudProcessor:
    def __init__(self, v_threshold=0.1, dt=0.1, alpha_weight=0.6, beta_weight=0.4):
        # 速度阈值：超出会增加动态概率
        self.v_threshold = v_threshold
        # 时间间隔
        self.dt = dt
        # SDF判据权重
        self.alpha_weight = alpha_weight
        # 速度判据权重
        self.beta_weight = beta_weight
        # 预计算 SDF 地图
        self.sdf_map = None
        # 地图参数
        self.map_res = 0.1
        self.map_width_m = 10.0
        self.map_height_m = 10.0
        # 上一帧点云的 KDTree
        self.prev_kdtree = None
        # 障碍物ID计数器（用于唯一标识每个障碍物）
        self.obstacle_id_counter = 0

    def set_map(self, map_obj):
        """
        1. 保持自适应尺寸调整
        2. 预计算 SDF
        """
        # 获取地图预设参数
        self.map_res = map_obj.resolution
        self.map_width_m = map_obj.width
        self.map_height_m = map_obj.height
        
        # --- 尺寸标准化 (Resampling) ---
        target_h = int(self.map_height_m / self.map_res)
        target_w = int(self.map_width_m / self.map_res)
        current_h, current_w = map_obj.grid.shape
        
        if (current_h, current_w) != (target_h, target_w):
            zoom_factors = (target_h / current_h, target_w / current_w)
            # 使用最近邻插值保持 0/100 离散性
            standardized_grid = ndimage.zoom(map_obj.grid, zoom_factors, order=0)
        else:
            standardized_grid = map_obj.grid.copy()

        # --- 生成 SDF 距离场 ---
        # 提取静态障碍物 (100)
        static_binary = (standardized_grid == 100).astype(np.uint8)
        
        # 计算每个点到最近障碍物的像素距离
        # 取反：计算“非障碍物”点到“障碍物”点的距离
        inverse_binary = (static_binary == 0).astype(np.uint8)
        pixel_dist_map = ndimage.distance_transform_edt(inverse_binary)
        
        # 转换为物理距离场 (SDF)
        self.sdf_map = pixel_dist_map * self.map_res
        self.h_max, self.w_max = self.sdf_map.shape

    def process(self, scan_data, robot_state):
        """
        雷达数据预处理
        输出接口修改：返回结构化的障碍物列表，每个障碍物包含独立点云、矩形参数、ID等信息
        """
        # 点云极坐标长度
        ranges = np.array(scan_data['ranges'])
        # 点云极坐标角度
        angles = scan_data['angle_min'] + np.arange(len(ranges)) * scan_data['angle_increment']
        # 筛选指定范围点云
        mask = (ranges > 0.1) & (ranges < 5.8)
        # 雷达坐标系下点云集合
        local_pts = np.column_stack((
            ranges[mask] * np.cos(angles[mask]),
            ranges[mask] * np.sin(angles[mask])
        ))

        # 获取有效点云数量
        num_points = len(local_pts)
        if num_points == 0: return self._empty_res()

        # 调用 Numba 逻辑 (传入预计算好的 SDF)
        tx, ty, theta = robot_state[0,0], robot_state[1,0], robot_state[2,0]
        global_pts, alpha = fast_process_logic(
            local_pts, tx, ty, theta, 
            self.map_res, self.h_max, self.w_max, 
            self.sdf_map, dist_threshold=0.2
        )

        # 速度判据 (Beta)
        beta = np.full(num_points, 0.05)
        if self.prev_kdtree is not None:
            dists, _ = self.prev_kdtree.query(global_pts, k=1)
            avg_vel = dists / self.dt
            beta = np.where( avg_vel < self.v_threshold, 0.95, 0.05)

        # 融合与分类 四种情况
        confidence_C = alpha * self.alpha_weight + beta * self.beta_weight

        # 静态/动态点云划分
        is_static = confidence_C >= 0.43
        static_points = global_pts[is_static]
        foreground_pts = global_pts[~is_static]
        # 保存每个前景点的置信度（用于后续筛选）
        foreground_confidence = confidence_C[~is_static]

        # 保存上一帧KDtree
        self.prev_kdtree = KDTree(global_pts)
        
        # 调用矩形拟合：返回每个障碍物的独立信息
        obstacles = self.scan_rectangle(foreground_pts, foreground_confidence)
        
        # 构造输出结果（结构化接口）
        output = {
            # 全局点云数据（兼容原有接口）
            'static_points': static_points,       # 全局静态点云 (N, 2)
            'dynamic_points': foreground_pts,    # 全局前景点云 (M, 2)
            
            # 结构化障碍物列表（新增核心接口）
            'obstacles': obstacles,              # 列表：每个元素是一个障碍物的完整信息
            'obstacle_count': len(obstacles),    # 障碍物总数
            
            # 原有接口字段（保留，确保向后兼容）
            'centers': [obs['rect_params'][0:2] for obs in obstacles],  # 所有障碍物中心 (cx, cy)
            'boxes': [obs['box'] for obs in obstacles]                  # 所有障碍物矩形顶点
        }
        
        return output
        
    def scan_rectangle(self, foreground_pts, foreground_confidence):
        """
        输入: 
            foreground_pts: 全局坐标系下的前景点云 (M, 2)
            foreground_confidence: 前景点云的动态置信度 (M,)
        输出: 
            obstacles: 结构化障碍物列表，每个元素包含：
                {
                    'obstacle_id': int,          # 唯一障碍物ID
                    'cluster_pts': np.ndarray,   # 该障碍物的点云簇 (K, 2)
                    'rect_params': list,         # 矩形参数 [cx, cy, w, h, angle_rad]
                    'box': np.ndarray,           # 矩形4个顶点 (2, 4)
                    'confidence': float,         # 障碍物动态置信度（簇内点均值）
                    'point_count': int           # 该障碍物的点云数量
                }
        """
        obstacles = []

        # 1. 基础过滤：前景点云数量过少
        if len(foreground_pts) < 3:
            return obstacles

        # 2. 使用 DBSCAN 聚类
        labels = DBSCAN(eps=0.2, min_samples=3).fit_predict(foreground_pts)

        # 3. 遍历每个有效簇（排除噪声）
        for label in np.unique(labels):
            if label == -1:  # 噪声点，跳过
                continue
            
            # 提取当前簇的点云和置信度
            cluster_mask = (labels == label)
            cluster_pts = foreground_pts[cluster_mask].astype(np.float32)
            cluster_confidence = foreground_confidence[cluster_mask]
            
            # 过滤点数过少的小簇
            if len(cluster_pts) < 3:
                continue
            
            try:
                # 4. 计算最小外接矩形
                rect = cv2.minAreaRect(cluster_pts)
                (cx, cy), (w, h), angle_deg = rect
                angle_rad = np.deg2rad(angle_deg)
                
                # 5. 生成矩形4个顶点（形状：2x4，便于可视化）
                box = cv2.boxPoints(rect).T  # (2, 4)：每行x/y坐标，每列一个顶点
                
                # 6. 计算该障碍物的动态置信度（簇内点置信度均值）
                avg_confidence = np.mean(cluster_confidence)
                
                # 7. 构造单个障碍物信息
                obstacle = {
                    'obstacle_id': self._get_next_obstacle_id(),
                    'cluster_pts': cluster_pts,
                    'rect_params': [cx, cy, w, h, angle_rad],
                    'box': box,
                    'confidence': avg_confidence,
                    'point_count': len(cluster_pts)
                }
                
                obstacles.append(obstacle)
                
            except Exception as e:
                print(f"Warning: 处理障碍物簇时出错 - {str(e)}")
                continue

        return obstacles
    
    def _get_next_obstacle_id(self):
        """生成唯一的障碍物ID（自增）"""
        current_id = self.obstacle_id_counter
        self.obstacle_id_counter += 1
        # 防止ID溢出（可选：达到最大值后重置）
        if self.obstacle_id_counter >= 1000000:
            self.obstacle_id_counter = 0
        return current_id

    def _empty_res(self):
        """空结果返回（保持接口一致性）"""
        return {
            'static_points': np.zeros((0, 2)),
            'dynamic_points': np.zeros((0, 2)),
            'obstacles': [],
            'obstacle_count': 0,
            'centers': [],
            'boxes': []
        }
    
    