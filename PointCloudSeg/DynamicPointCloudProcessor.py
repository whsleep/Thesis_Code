import numpy as np
from scipy.spatial import KDTree
from typing import Tuple, List, Optional, Dict, Any
import matplotlib.pyplot as plt


class DynamicPointCloudProcessor:
    """
    全局坐标系下的动态点云处理器
    功能：将雷达坐标系的点云转换到全局坐标系，并分离动静态点云
    """
    
    def __init__(self, range_min: float = 0, range_max: float = 5.95, 
                 static_velocity_threshold: float = 0.3):
        """
        初始化处理器参数
        
        Args:
            range_min: 最小探测距离
            range_max: 最大探测距离
            static_velocity_threshold: 静态点速度阈值
        """
        self.range_min = range_min
        self.range_max = range_max
        self.static_velocity_threshold = static_velocity_threshold
        
        # 存储历史数据用于时序分析
        self.previous_global_points = None
        self.previous_kdtree = None
        self.previous_robot_state = None
        self.frame_count = 0
    
    def polar_to_cartesian(self, ranges: np.ndarray, 
                        angle_min: float, angle_increment: float) -> np.ndarray:
        """
        将雷达极坐标数据转换为笛卡尔坐标点云（雷达坐标系）
        
        Args:
            ranges: 距离测量数组，包含所有原始测量点（包括无效值）
            angle_min: 起始角度（雷达扫描范围的开始，通常是-π）
            angle_increment: 角度增量（相邻测量点间的角度差）
            
        Returns:
            cartesian_points: 笛卡尔坐标点云数组 (N, 2)
        """
        # 1. 生成与原始ranges数组完全对应的所有角度
        num_total_points = len(ranges)
        all_angles = angle_min + np.arange(num_total_points) * angle_increment
        
        # 2. 过滤无效距离值（同时适用于ranges和all_angles）
        valid_mask = (ranges >= self.range_min) & (ranges <= self.range_max)
        valid_ranges = ranges[valid_mask]
        valid_angles = all_angles[valid_mask]  # 关键修正：使用相同的掩码过滤角度
        
        # 3. 极坐标转笛卡尔坐标（仅对有效点）
        x = valid_ranges * np.cos(valid_angles)
        y = valid_ranges * np.sin(valid_angles)
        
        cartesian_points = np.column_stack((x, y))
        return cartesian_points
    
    def lidar_to_global_transform(self, points: np.ndarray, 
                                 robot_state: np.ndarray) -> np.ndarray:
        """
        将点云从雷达坐标系转换到全局坐标系[1,6,7](@ref)
        
        Args:
            points: 雷达坐标系下的点云 (N, 2)
            robot_state: 机器人状态 [x, y, theta]^T
            
        Returns:
            global_points: 全局坐标系下的点云 (N, 2)
        """
        if len(points) == 0:
            return np.zeros((0, 2))
        
        # 提取机器人状态
        robot_x = robot_state[0, 0]
        robot_y = robot_state[1, 0] 
        robot_theta = robot_state[2, 0]
        
        # 构建旋转矩阵[6,8](@ref)
        cos_theta = np.cos(robot_theta)
        sin_theta = np.sin(robot_theta)
        rotation_matrix = np.array([[cos_theta, -sin_theta],
                                   [sin_theta, cos_theta]])
        
        # 应用坐标系变换：先旋转，后平移[1](@ref)
        # P_global = R * P_lidar + T
        rotated_points = (rotation_matrix @ points.T).T
        global_points = rotated_points + np.array([robot_x, robot_y])
        return global_points
    
    def build_kd_tree(self, points: np.ndarray) -> KDTree:
        """
        为点云构建KD-Tree结构
        """
        if len(points) == 0:
            raise ValueError("点云数据不能为空")
        
        kdtree = KDTree(points)
        return kdtree
    
    def compute_velocity_vectors_global(self, current_global_points: np.ndarray,
                                      time_interval: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        在全局坐标系下基于时序差分计算每个点的速度矢量
        """
        if self.previous_global_points is None or self.previous_kdtree is None:
            # 第一帧数据，无法计算速度
            self._update_previous_data(current_global_points)
            return np.zeros((0, 2)), np.array([], dtype=int)
        
        if len(current_global_points) == 0:
            return np.zeros((0, 2)), np.array([], dtype=int)
        
        # 使用KD-Tree寻找最近邻匹配点
        distances, indices = self.previous_kdtree.query(current_global_points, k=1)
        
        # 过滤匹配距离过大的点（可能是噪声或新出现的点）
        max_match_distance = 1.0  # 全局坐标系下增大匹配距离阈值
        valid_mask = distances < max_match_distance
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            velocities = np.zeros((0, 2))
        else:
            # 计算位移矢量（全局坐标系）
            displacements = current_global_points[valid_indices] - self.previous_global_points[indices[valid_mask]]
            
            # 计算速度矢量 (位移/时间)
            velocities = displacements / time_interval
        
        self._update_previous_data(current_global_points)
        return velocities, valid_indices
    
    def _update_previous_data(self, current_global_points: np.ndarray):
        """更新历史数据"""
        self.previous_global_points = current_global_points.copy()
        if len(current_global_points) > 0:
            self.previous_kdtree = self.build_kd_tree(current_global_points)
        self.frame_count += 1
    
    def classify_static_dynamic_global(self, global_points: np.ndarray, 
                                      velocities: np.ndarray,
                                      valid_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        在全局坐标系下根据速度矢量区分动静态点云
        """
        if len(valid_indices) == 0:
            # 没有速度数据，将所有点视为静态
            return global_points, np.zeros((0, 2))
        
        # 计算速度大小
        velocity_magnitudes = np.linalg.norm(velocities, axis=1)
        
        # 根据速度阈值分类
        dynamic_mask = velocity_magnitudes > self.static_velocity_threshold
        dynamic_indices = valid_indices[dynamic_mask]
        
        # 获取所有点的索引
        all_indices = np.arange(len(global_points))
        static_indices = np.setdiff1d(all_indices, dynamic_indices)
        
        static_points = global_points[static_indices] if len(static_indices) > 0 else np.zeros((0, 2))
        dynamic_points = global_points[dynamic_indices] if len(dynamic_indices) > 0 else np.zeros((0, 2))
        
        return static_points, dynamic_points
    
    def process_frame_global(self, scan_data: dict, robot_state: np.ndarray,
                            time_interval: float = 0.1) -> dict:
        """
        处理单帧雷达数据的完整流程（全局坐标系）
        
        Args:
            scan_data: 雷达数据字典，包含ranges, angle_min, angle_increment等
            robot_state: 机器人状态 [x, y, theta]^T
            time_interval: 时间间隔
            
        Returns:
            result: 处理结果字典
        """
        # 1. 坐标转换：极坐标 -> 雷达坐标系笛卡尔坐标
        lidar_points = self.polar_to_cartesian(
            scan_data['ranges'], 
            scan_data['angle_min'],
            scan_data['angle_increment']
        )
        
        # 2. 坐标系转换：雷达坐标系 -> 全局坐标系[1](@ref)
        global_points = self.lidar_to_global_transform(lidar_points, robot_state)
        
        # 3. 构建KD-Tree（全局坐标系）
        kdtree = self.build_kd_tree(global_points) if len(global_points) > 0 else None
        
        # 4. 计算速度矢量（全局坐标系）
        velocities, valid_indices = self.compute_velocity_vectors_global(
            global_points, time_interval
        )
        
        # 5. 动静态点云分离（全局坐标系）
        static_points, dynamic_points = self.classify_static_dynamic_global(
            global_points, velocities, valid_indices
        )
        
        result = {
            'lidar_points': lidar_points,  # 雷达坐标系下的点云
            'global_points': global_points,  # 全局坐标系下的点云
            'kdtree': kdtree,
            'velocities': velocities,  # 全局坐标系下的速度
            'valid_indices': valid_indices,
            'static_points': static_points,  # 全局坐标系下的静态点
            'dynamic_points': dynamic_points,  # 全局坐标系下的动态点
            'static_count': len(static_points),
            'dynamic_count': len(dynamic_points),
            'robot_state': robot_state  # 当前机器人状态
        }
        
        # 更新机器人状态历史
        self.previous_robot_state = robot_state
        
        return result
    
    def get_processing_stats(self) -> dict:
        """获取处理统计信息"""
        return {
            'processed_frames': self.frame_count,
            'has_previous_data': self.previous_global_points is not None,
            'has_robot_state': self.previous_robot_state is not None
        }
