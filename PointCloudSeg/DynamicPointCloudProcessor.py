import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree as KDTree
from numba import njit, prange

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
            # 注意：此处索引需与标准化 grid 的存储顺序一致 (通常为 grid[row, col])
            # 在 set_map 中我们确保了坐标与索引的映射关系
            dist_to_static = sdf_map[ix, iy] 
            alpha[i] = 0.8 if dist_to_static <= dist_threshold else 0.2
        else:
            alpha[i] = 0.2
            
    return global_pts, alpha

class DynamicPointCloudProcessor:
    def __init__(self, v_threshold=0.1, dt=0.1, alpha_weight=0.8, beta_weight=0.2):
        self.v_threshold = v_threshold
        self.dt = dt
        self.alpha_weight = alpha_weight
        self.beta_weight = beta_weight
        
        self.sdf_map = None
        self.map_res = 0.1
        self.map_width_m = 10.0
        self.map_height_m = 10.0
        self.prev_kdtree = None

    def set_map(self, map_obj):
        """
        1. 保持自适应尺寸调整
        2. 预计算 SDF
        """
        self.map_res = map_obj.resolution
        self.map_width_m = map_obj.width
        self.map_height_m = map_obj.height
        
        # --- 步骤 1: 尺寸标准化 (Resampling) ---
        target_h = int(self.map_height_m / self.map_res)
        target_w = int(self.map_width_m / self.map_res)
        current_h, current_w = map_obj.grid.shape
        
        if (current_h, current_w) != (target_h, target_w):
            zoom_factors = (target_h / current_h, target_w / current_w)
            # 使用最近邻插值保持 0/100 离散性
            standardized_grid = ndimage.zoom(map_obj.grid, zoom_factors, order=0)
        else:
            standardized_grid = map_obj.grid.copy()

        # --- 步骤 2: 生成 SDF 距离场 ---
        # 提取静态障碍物 (100)
        static_binary = (standardized_grid == 100).astype(np.uint8)
        
        # 计算每个点到最近障碍物的像素距离
        # 取反：计算“非障碍物”点到“障碍物”点的距离
        inverse_binary = (static_binary == 0).astype(np.uint8)
        pixel_dist_map = ndimage.distance_transform_edt(inverse_binary)
        
        # 转换为物理距离场 (SDF)
        self.sdf_map = pixel_dist_map * self.map_res
        self.h_max, self.w_max = self.sdf_map.shape

        # --- 新增：对比可视化调试 ---
        # self._visualize_sdf_comparison(standardized_grid, self.sdf_map)

    def _visualize_sdf_comparison(self, original_grid, sdf_map):
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 5))
        
        # 左图：原始标准化栅格地图
        plt.subplot(1, 2, 1)
        plt.title("Standardized Grid (Original)")
        plt.imshow(original_grid, origin='lower', cmap='gray_r')
        plt.colorbar(label='Occupancy Value')
        
        # 右图：计算出的 SDF 距离场
        plt.subplot(1, 2, 2)
        plt.title("SDF (Distance to Obstacles)")
        # 使用 'viridis' 或 'jet' 色带可以更清晰地看出距离梯度
        plt.imshow(sdf_map, origin='lower', cmap='viridis')
        plt.colorbar(label='Distance (meters)')
        
        plt.tight_layout()
        plt.show()

    def _radius_outlier_removal(self, points, r=0.2, min_pts=5):
        if len(points) < min_pts: return np.zeros((0, 2))
        tree = KDTree(points)
        counts = tree.query_ball_point(points, r, return_length=True)
        return points[counts >= min_pts]

    def process(self, scan_data, robot_state):
        # 雷达数据预处理
        ranges = np.array(scan_data['ranges'])
        angles = scan_data['angle_min'] + np.arange(len(ranges)) * scan_data['angle_increment']
        mask = (ranges > 0.1) & (ranges < 5.8)
        local_pts = np.column_stack((
            ranges[mask] * np.cos(angles[mask]),
            ranges[mask] * np.sin(angles[mask])
        ))

        num_points = len(local_pts)
        if num_points == 0: return self._empty_res()

        # 调用 Numba 逻辑 (传入预计算好的 SDF)
        tx, ty, theta = robot_state[0,0], robot_state[1,0], robot_state[2,0]
        global_pts, alpha = fast_process_logic(
            local_pts, tx, ty, theta, 
            self.map_res, self.h_max, self.w_max, 
            self.sdf_map, dist_threshold=0.3
        )

        # 速度判据 (Beta)
        beta = np.full(num_points, 0.3)
        if self.prev_kdtree is not None:
            dists, _ = self.prev_kdtree.query(global_pts, k=1)
            avg_vel = dists / self.dt
            beta = np.where( avg_vel < self.v_threshold, 0.7, 0.3)

        # 融合与分类 0.2*0.8 + 0.3*0.2=0.22
        confidence_C = alpha * self.alpha_weight + beta * self.beta_weight

        is_static = confidence_C >= 0.3
        static_points = global_pts[is_static]
        foreground_pts = global_pts[~is_static]

        # 滤波
        # dynamic_points = self._radius_outlier_removal(foreground_pts)
        self.prev_kdtree = KDTree(global_pts)
        
        return {'static_points': static_points, 'dynamic_points': foreground_pts}

    def cluster_dynamic_points(self, dynamic_pts, d_th=0.3, min_pts=2):
        """
        按照论文思路对动态点进行聚类
        :param dynamic_pts: N x 2 的动态点坐标
        :param d_th: 聚类距离阈值 (论文中可根据距离 r 动态计算)
        :param min_pts: 形成簇的最小点数
        """
        if len(dynamic_pts) < min_pts:
            return []

        # 1. 构建 KDTree 加速搜索
        tree = KDTree(dynamic_pts)
        
        # 2. 找到每个点的邻居索引
        # query_ball_point 会返回每个点在 d_th 范围内的点索引列表
        adj_list = tree.query_ball_point(dynamic_pts, d_th)
        
        # 3. 使用深度优先搜索 (DFS) 或 Breadth-First 标记连通分量
        clusters = []
        visited = np.zeros(len(dynamic_pts), dtype=bool)
        
        for i in range(len(dynamic_pts)):
            if not visited[i]:
                # 开始一个新的簇
                new_cluster_indices = []
                stack = [i]
                visited[i] = True
                
                while stack:
                    curr = stack.pop()
                    new_cluster_indices.append(curr)
                    for neighbor in adj_list[curr]:
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            stack.append(neighbor)
                
                # 4. 过滤小噪声点簇
                if len(new_cluster_indices) >= min_pts:
                    clusters.append(dynamic_pts[new_cluster_indices])
                    
        return clusters

    def get_cluster_boxes(self, clusters):
        """
        计算每个点簇的 AABB 包围盒顶点
        Args:
            clusters: 包含多个 (N, 2) 点阵的列表
        Returns:
            box_vertices: 包含多个 (2, 5) 顶点矩阵的列表
        """
        box_list = []
        for cluster in clusters:
            # 1. 寻找点簇在 X 和 Y 轴上的极值
            x_min, y_min = np.min(cluster, axis=0)
            x_max, y_max = np.max(cluster, axis=0)
            
            # 2. 构建 5 个顶点 (闭合矩形：左下-右下-右上-左上-左下)
            # 形状为 (2, 5) 以符合 irsim 的 draw_box 要求
            vertex = np.array([
                [x_min, x_max, x_max, x_min, x_min],
                [y_min, y_min, y_max, y_max, y_min]
            ])
            box_list.append(vertex)
            
        return box_list

    def _empty_res(self):
        return {'static_points': np.zeros((0,2)), 'dynamic_points': np.zeros((0,2))}