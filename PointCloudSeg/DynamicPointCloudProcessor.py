import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN  # 引入 DBSCAN
from typing import Tuple, List, Optional, Dict, Any

class DynamicPointCloudProcessor:
    """
    全局坐标系下的动态点云处理器 (引入聚类预处理)
    功能：将雷达坐标系的点云转换到全局坐标系，并分离动静态点云
    
    采用 DBSCAN 聚类，然后基于簇的平均速度进行动静态区分，提高鲁棒性。
    """
    
    def __init__(self, range_min: float = 0, range_max: float = 5.95, 
                 static_velocity_threshold: float = 0.1,
                 dbscan_eps: float = 0.25,  # DBSCAN 邻域半径 (例如 0.5米)
                 dbscan_min_samples: int = 3):  # 构成核心点的最小样本数
        """
        初始化处理器参数
        
        Args:
            range_min: 最小探测距离
            range_max: 最大探测距离
            static_velocity_threshold: 静态点速度阈值
            dbscan_eps: DBSCAN 邻域半径 (用于聚类)
            dbscan_min_samples: 构成簇的最小点数
        """
        self.range_min = range_min
        self.range_max = range_max
        self.static_velocity_threshold = static_velocity_threshold
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        
        # 存储历史数据用于时序分析
        self.previous_global_points = None
        self.previous_kdtree = None
        self.previous_robot_state = None
        self.frame_count = 0
    
    def polar_to_cartesian(self, ranges: np.ndarray, 
                           angle_min: float, angle_increment: float) -> np.ndarray:
        """
        将雷达极坐标数据转换为笛卡尔坐标点云（雷达坐标系）
        """
        num_total_points = len(ranges)
        # 生成与原始ranges数组完全对应的所有角度
        all_angles = angle_min + np.arange(num_total_points) * angle_increment
        
        # 过滤无效距离值
        valid_mask = (ranges >= self.range_min) & (ranges <= self.range_max)
        valid_ranges = ranges[valid_mask]
        valid_angles = all_angles[valid_mask]
        
        # 极坐标转笛卡尔坐标
        x = valid_ranges * np.cos(valid_angles)
        y = valid_ranges * np.sin(valid_angles)
        
        cartesian_points = np.column_stack((x, y))
        return cartesian_points
    
    def lidar_to_global_transform(self, points: np.ndarray, 
                                  robot_state: np.ndarray) -> np.ndarray:
        """
        将点云从雷达坐标系转换到全局坐标系
        
        Args:
            points: 雷达坐标系下的点云 (N, 2)
            robot_state: 机器人状态 [x, y, theta]^T (3, 1)
        """
        if len(points) == 0:
            return np.zeros((0, 2))
        
        # 提取机器人状态
        robot_x = robot_state[0, 0]
        robot_y = robot_state[1, 0] 
        robot_theta = robot_state[2, 0]
        
        # 构建旋转矩阵
        cos_theta = np.cos(robot_theta)
        sin_theta = np.sin(robot_theta)
        rotation_matrix = np.array([[cos_theta, -sin_theta],
                                    [sin_theta, cos_theta]])
        
        # 应用坐标系变换： P_global = R * P_lidar + T
        rotated_points = (rotation_matrix @ points.T).T
        global_points = rotated_points + np.array([robot_x, robot_y])
        return global_points
    
    def build_kd_tree(self, points: np.ndarray) -> KDTree:
        """
        为点云构建KD-Tree结构
        """
        if len(points) == 0:
            # 允许构建空的 KDTree，但为了兼容 scipy，抛出异常或返回 None
            raise ValueError("点云数据不能为空")
        
        kdtree = KDTree(points)
        return kdtree
    
    def _update_previous_data(self, current_global_points: np.ndarray):
        """更新历史数据"""
        self.previous_global_points = current_global_points.copy()
        if len(current_global_points) > 0:
            try:
                self.previous_kdtree = self.build_kd_tree(current_global_points)
            except ValueError:
                self.previous_kdtree = None
        else:
            self.previous_kdtree = None
            
        self.frame_count += 1

    def cluster_points_dbscan(self, global_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用DBSCAN对全局点云进行聚类
        
        Returns:
            labels: 每个点的簇标签 (-1表示噪声点)
            unique_cluster_ids: 唯一有效的簇ID列表
        """
        if len(global_points) < self.dbscan_min_samples:
            return np.full(len(global_points), -1), np.array([], dtype=int)
            
        db = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
        db.fit(global_points)
        labels = db.labels_
        
        # 排除噪声点(-1)的簇ID
        unique_cluster_ids = np.unique(labels[labels != -1])
        
        return labels, unique_cluster_ids

    def compute_cluster_velocities_global(self, 
                                          current_global_points: np.ndarray,
                                          current_labels: np.ndarray,
                                          unique_cluster_ids: np.ndarray,
                                          time_interval: float = 0.1) -> Dict[int, np.ndarray]:
        """
        基于聚类，计算每个有效簇的平均速度矢量。
        
        Returns:
            cluster_velocities: 字典 {cluster_id: average_velocity (1x2)}
        """
        # 检查前一帧数据和 KD-Tree 是否可用
        if self.previous_global_points is None or self.previous_kdtree is None or len(unique_cluster_ids) == 0:
            return {}
        
        cluster_velocities = {}
        # 匹配距离可以略大于聚类半径，但也不宜过大，防止跨物体匹配
        max_match_distance = self.dbscan_eps * 2.0

        for cluster_id in unique_cluster_ids:
            # 1. 提取当前簇的点
            current_cluster_mask = (current_labels == cluster_id)
            current_cluster_points = current_global_points[current_cluster_mask]
            
            # 2. 对簇内所有点进行最近邻匹配
            distances, indices = self.previous_kdtree.query(current_cluster_points, k=1)
            
            # 3. 过滤有效匹配点
            valid_match_mask = distances < max_match_distance
            
            if np.sum(valid_match_mask) < self.dbscan_min_samples:
                # 如果有效匹配点过少，则认为该簇无法追踪，跳过
                continue
            
            # 4. 计算平均位移矢量
            valid_current_points = current_cluster_points[valid_match_mask]
            valid_previous_points = self.previous_global_points[indices[valid_match_mask]]
            
            # 位移 = 当前点 - 匹配的前一帧点
            displacements = valid_current_points - valid_previous_points
            average_displacement = np.mean(displacements, axis=0)  # 簇的平均位移
            
            # 5. 计算簇的平均速度
            average_velocity = average_displacement / time_interval
            cluster_velocities[cluster_id] = average_velocity
            
        return cluster_velocities
    
    def classify_static_dynamic_global_clustered(self, 
                                                 global_points: np.ndarray, 
                                                 labels: np.ndarray,
                                                 cluster_velocities: Dict[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        基于簇平均速度区分动静态点云
        """
        if len(global_points) == 0:
            return np.zeros((0, 2)), np.zeros((0, 2))
            
        # 默认为静态点（包括所有噪声点、未被追踪的簇）
        is_dynamic = np.zeros(len(global_points), dtype=bool)
        
        for cluster_id, velocity in cluster_velocities.items():
            velocity_magnitude = np.linalg.norm(velocity)
            
            if velocity_magnitude > self.static_velocity_threshold:
                # 如果簇的平均速度大于阈值，则将簇内所有点标记为动态点
                cluster_mask = (labels == cluster_id)
                is_dynamic[cluster_mask] = True
        
        # 区分点云
        static_points = global_points[~is_dynamic]
        dynamic_points = global_points[is_dynamic]
        
        return static_points, dynamic_points

    def process_frame_global(self, scan_data: dict, robot_state: np.ndarray,
                             time_interval: float = 0.1) -> dict:
        """
        处理单帧雷达数据的完整流程（全局坐标系, 包含聚类）
        """
        # 1. 坐标转换：极坐标 -> 雷达坐标系笛卡尔坐标
        lidar_points = self.polar_to_cartesian(
            scan_data['ranges'], 
            scan_data['angle_min'],
            scan_data['angle_increment']
        )
        
        # 2. 坐标系转换：雷达坐标系 -> 全局坐标系
        global_points = self.lidar_to_global_transform(lidar_points, robot_state)
        
        if len(global_points) == 0:
            # 更新历史数据并返回空结果
            self._update_previous_data(global_points)
            return {
                'global_points': global_points,
                'static_points': np.zeros((0, 2)), 
                'dynamic_points': np.zeros((0, 2)), 
                'static_count': 0, 
                'dynamic_count': 0,
                'robot_state': robot_state
            }
        
        # 3. 聚类预处理
        labels, unique_cluster_ids = self.cluster_points_dbscan(global_points)
        
        # 4. 计算簇速度矢量（基于帧间匹配）
        # 注意：这里我们使用上一帧的点云和 KD-Tree，所以计算完成后才更新历史数据
        cluster_velocities = self.compute_cluster_velocities_global(
            global_points, labels, unique_cluster_ids, time_interval
        )
        
        # 5. 动静态点云分离（基于簇速度）
        static_points, dynamic_points = self.classify_static_dynamic_global_clustered(
            global_points, labels, cluster_velocities
        )
        
        # 6. 更新历史数据
        self._update_previous_data(global_points)
        self.previous_robot_state = robot_state
        
        result = {
            'lidar_points': lidar_points,
            'global_points': global_points,
            'labels': labels,                     # DBSCAN 簇标签
            'cluster_velocities': cluster_velocities, # 簇的平均速度
            'static_points': static_points,
            'dynamic_points': dynamic_points,
            'static_count': len(static_points),
            'dynamic_count': len(dynamic_points),
            'robot_state': robot_state
        }
        
        return result

    def get_processing_stats(self) -> dict:
        """获取处理统计信息"""
        return {
            'processed_frames': self.frame_count,
            'has_previous_data': self.previous_global_points is not None,
            'has_robot_state': self.previous_robot_state is not None
        }