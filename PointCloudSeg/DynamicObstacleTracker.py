import numpy as np
from scipy.spatial import cKDTree
from filterpy.kalman import KalmanFilter
from numba import njit


class DynamicObstacleTracker:
    def __init__(self, 
                 sim_threshold=0.7,  # 特征相似度阈值（论文式定义）
                 vel_threshold=0.1,  # 动态判定速度阈值 (m/s)
                 vote_ratio=0.5,     # 点云投票动态判定比例（论文式双判据）
                 dt=0.1):            # 时间步长（与点云处理器一致）
        """
        初始化动态障碍物跟踪器
        适配修改后的 DynamicPointCloudProcessor 输出接口
        """
        self.sim_threshold = sim_threshold
        self.vel_threshold = vel_threshold
        self.vote_ratio = vote_ratio
        self.dt = dt
        
        # 跟踪器状态：存储所有活跃障碍物的跟踪信息
        # 格式：{track_id: {'state': 卡尔曼滤波实例, 'history': 特征历史, 'point_cloud_history': 点云历史}}
        self.active_tracks = {}
        self.next_track_id = 0  # 下一个未使用的跟踪ID
    
    # ========== 1. 特征提取函数（无装饰器，规范缩进） ==========
    def _extract_obstacle_feature(self, obstacle):
        """
        提取障碍物特征向量（论文公式3，适配结构化障碍物输入）
        输入：
            obstacle: 结构化障碍物字典（来自 DynamicPointCloudProcessor 的 output['obstacles']）
        输出：
            feat: 归一化特征向量 [pos_x, pos_y, dim_w, dim_h, point_len, point_std]
        """
        # 从障碍物字典中提取核心信息
        cx, cy, w, h, _ = obstacle['rect_params']
        cluster_pts = obstacle['cluster_pts']
        point_len = obstacle['point_count']
        # 点云位置标准差（统计特征，论文核心）
        point_std = np.std(cluster_pts, axis=0).mean()  # 二维标准差的均值
        
        # 特征向量（对齐论文公式3：pos, dim, len, std）
        feat = np.array([
            cx, cy,          # 中心位置 pos(i)
            w, h,            # 尺寸 dim(i)（2D简化，论文为3D）
            point_len,       # 点云数量 len(i)
            point_std        # 点云标准差 std(i)
        ], dtype=np.float64)
        
        # 特征归一化（减少量纲影响，论文关键步骤）
        # feat = (feat - feat.mean()) / (feat.std() + 1e-8)
        return feat
    
    # ========== 2. 相似度计算函数（仅此处加 njit，规范缩进） ==========
    def _calculate_similarity(self, feat1, feat2):
        """
        纯数值计算，适合 Numba 加速
        """
        l2_dist_sq = np.sum((feat1 - feat2)**2)/1000
        return np.exp(-l2_dist_sq)
    
    def _data_association(self, current_obstacles):
        """
        特征基数据关联（论文3.4节）：匹配当前帧与历史跟踪障碍物
        输入：
            current_obstacles: 结构化障碍物列表（来自处理器输出的 obstacles 字段）
        输出：
            matches: 匹配结果 (track_id, current_obstacle)
            unmatched_tracks: 未匹配的历史跟踪ID
            unmatched_current: 未匹配的当前帧障碍物
        """
        M = len(current_obstacles)
        T = len(self.active_tracks)
        if M == 0 or T == 0:
            return [], list(self.active_tracks.keys()), current_obstacles
        
        # 提取当前帧所有障碍物的特征
        current_features = [self._extract_obstacle_feature(obs) for obs in current_obstacles]
        
        # 构建相似度矩阵：shape (T, M)
        track_ids = list(self.active_tracks.keys())
        sim_matrix = np.zeros((T, M), dtype=np.float64)
        
        for i, track_id in enumerate(track_ids):
            # 历史跟踪障碍物的预测特征（使用卡尔曼滤波预测位置更新特征）
            track = self.active_tracks[track_id]
            kf = track['state']
            # 卡尔曼滤波预测的位置 (cx, cy)
            pred_cx, pred_cy = kf.x[0], kf.x[1]
            # 用预测位置更新历史特征（论文线性传播策略）
            pred_feat = track['history'][-1].copy()
            pred_feat[0], pred_feat[1] = pred_cx, pred_cy  # 更新位置特征
            
            # 计算与当前所有障碍物的相似度
            for j in range(M):
                sim_matrix[i, j] = self._calculate_similarity(pred_feat, current_features[j])
        
        # 匈牙利算法匹配（最大权重匹配）
        matches = []
        used_tracks = set()
        used_current = set()
        
        # 按相似度降序匹配
        while True:
            # 找到最大相似度
            max_sim = np.max(sim_matrix)
            if max_sim < self.sim_threshold:
                break
            
            # 找到对应索引
            t_idx, c_idx = np.unravel_index(np.argmax(sim_matrix), sim_matrix.shape)
            track_id = track_ids[t_idx]
            current_obstacle = current_obstacles[c_idx]
            
            # 标记已使用
            used_tracks.add(t_idx)
            used_current.add(c_idx)
            matches.append((track_id, current_obstacle))
            
            # 屏蔽该行列
            sim_matrix[t_idx, :] = 0
            sim_matrix[:, c_idx] = 0
        
        # 未匹配的跟踪和当前帧障碍物
        unmatched_tracks = [track_ids[i] for i in range(T) if i not in used_tracks]
        unmatched_current = [current_obstacles[j] for j in range(M) if j not in used_current]
        
        return matches, unmatched_tracks, unmatched_current
    
    def _init_kalman_filter(self, obstacle):
        """
        初始化卡尔曼滤波器（论文3.4节：恒加速模型）
        输入：
            obstacle: 结构化障碍物字典
        输出：
            kf: 卡尔曼滤波实例
        状态向量 X = [cx, cy, vx, vy, ax, ay]^T（位置、速度、加速度）
        """
        kf = KalmanFilter(dim_x=6, dim_z=2)
        dt = self.dt
        
        # 状态转移矩阵 A（论文公式7，2D简化）
        kf.F = np.array([
            [1, 0, dt, 0, 0.5*dt**2, 0],
            [0, 1, 0, dt, 0, 0.5*dt**2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # 测量矩阵 H（仅测量位置 cx, cy）
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ])
        
        # 过程噪声 Q（加速度噪声为主）
        kf.Q = np.diag([1e-1, 1e-1, 1e-2, 1e-2, 1e-2, 1e-2])
        # 测量噪声 R（位置测量噪声）
        kf.R = np.diag([1e-2, 1e-2])
        
        # 初始状态（位置=矩形中心，速度/加速度初始为0）
        cx, cy = obstacle['rect_params'][0], obstacle['rect_params'][1]
        kf.x = np.array([cx, cy, 0.0, 0.0, 0.0, 0.0])
        
        # 初始协方差矩阵 P
        kf.P = np.eye(6) * 0.1
        
        return kf
    
    def _dynamic_identification(self, track_id, current_obstacle):
        """
        动态障碍物识别（论文3.5节：双判据）
        1. 中心速度判据：|V_center| > vel_threshold
        2. 点云投票判据：动态点比例 > vote_ratio
        输入：
            track_id: 跟踪ID
            current_obstacle: 当前帧该障碍物的结构化信息
        输出：
            is_dynamic: 是否为动态障碍物（bool）
        """
        track = self.active_tracks[track_id]
        kf = track['state']
        current_cluster_pts = current_obstacle['cluster_pts']
        
        # 1. 中心速度判据（论文公式9前置条件）
        vx, vy = kf.x[2], kf.x[3]
        v_center = np.sqrt(vx**2 + vy**2)
        if v_center < self.vel_threshold:
            return False
        
        # 2. 点云投票判据（论文公式9）
        if len(track['point_cloud_history']) < 2:
            return False  # 缺乏历史点云，无法投票
        
        # 获取上一帧点云（用于计算点速度）
        prev_cluster_pts = track['point_cloud_history'][-2]
        if len(prev_cluster_pts) < 3 or len(current_cluster_pts) < 3:
            return False
        
        # 最近邻匹配点对
        prev_tree = cKDTree(prev_cluster_pts)
        dists, indices = prev_tree.query(current_cluster_pts, k=1)
        
        # 计算每个点的速度
        v_vote = dists / self.dt
        # 过滤无效匹配（距离过大）
        valid_mask = dists < 0.5  # 最大允许移动距离（可调整）
        if not np.any(valid_mask):
            return False
        
        # 动态点计数（速度超过阈值 + 方向与中心速度一致）
        v_center_dir = np.array([vx, vy]) / (v_center + 1e-8)
        current_valid_pts = current_cluster_pts[valid_mask]
        prev_valid_pts = prev_cluster_pts[indices[valid_mask]]
        
        # 计算每个有效点的速度方向
        point_vel_dir = (current_valid_pts - prev_valid_pts) / (dists[valid_mask].reshape(-1,1) + 1e-8)
        # 方向夹角 < 90度（论文公式10）
        dot_products = np.sum(point_vel_dir * v_center_dir, axis=1)
        dynamic_point_mask = (v_vote[valid_mask] > self.vel_threshold) & (dot_products > 0)
        
        # 动态点比例
        dynamic_ratio = np.sum(dynamic_point_mask) / len(valid_mask)
        return dynamic_ratio > self.vote_ratio
    
    def update(self, obstacles):
        """
        核心更新接口：直接输入结构化障碍物列表（与处理器输出完全匹配）
        输入：
            obstacles: 结构化障碍物列表（来自 DynamicPointCloudProcessor 输出的 'obstacles' 字段）
        输出：
            tracked_dynamic_obstacles: 跟踪后的动态障碍物
                格式：{track_id: {'rect': 矩形参数, 'box': 顶点列表, 'velocity': (vx, vy), 'confidence': 动态置信度}}
        """
        # 步骤1：过滤低置信度障碍物（可选，提升鲁棒性）
        valid_obstacles = [obs for obs in obstacles if obs['confidence'] < 0.43]  # 仅保留前景点障碍物
        
        # 步骤2：数据关联（匹配当前帧与历史跟踪）
        matches, unmatched_tracks, unmatched_current = self._data_association(valid_obstacles)
        
        # 步骤3：更新已匹配的跟踪
        for track_id, current_obstacle in matches:
            track = self.active_tracks[track_id]
            kf = track['state']
            # 卡尔曼滤波更新（测量值为当前矩形中心）
            cx, cy = current_obstacle['rect_params'][0], current_obstacle['rect_params'][1]
            z = np.array([cx, cy])
            kf.predict()
            kf.update(z)
            
            # 更新历史特征和点云
            current_feat = self._extract_obstacle_feature(current_obstacle)
            track['history'].append(current_feat)
            track['point_cloud_history'].append(current_obstacle['cluster_pts'])
            # 限制历史长度（节省内存）
            if len(track['history']) > 20:
                track['history'] = track['history'][-20:]
                track['point_cloud_history'] = track['point_cloud_history'][-20:]
        
        # 步骤4：删除未匹配的跟踪（跟踪丢失）
        for track_id in unmatched_tracks:
            del self.active_tracks[track_id]
        
        # 步骤5：初始化未匹配的当前帧为新跟踪
        for obstacle in unmatched_current:
            # 初始化卡尔曼滤波
            kf = self._init_kalman_filter(obstacle)
            # 提取初始特征
            init_feat = self._extract_obstacle_feature(obstacle)
            # 添加新跟踪
            self.active_tracks[self.next_track_id] = {
                'state': kf,
                'history': [init_feat],  # 特征历史
                'point_cloud_history': [obstacle['cluster_pts']]  # 点云历史
            }
            self.next_track_id += 1
        
        # 步骤6：动态障碍物识别与输出
        tracked_dynamic_obstacles = {}
        for track_id, track in self.active_tracks.items():
            # 找到当前帧对应的障碍物（通过特征匹配确认，简化版：取最新点云对应的障碍物）
            # 更严谨的方式：存储当前障碍物的引用，此处简化为从点云历史取最新
            current_cluster_pts = track['point_cloud_history'][-1]
            # 查找当前帧中对应的障碍物（简化：通过点云中心匹配）
            current_obstacle = None
            for obs in valid_obstacles:
                obs_cx, obs_cy = obs['rect_params'][0], obs['rect_params'][1]
                track_cx, track_cy = track['state'].x[0], track['state'].x[1]
                if np.sqrt((obs_cx - track_cx)**2 + (obs_cy - track_cy)**2) < 0.1:
                    current_obstacle = obs
                    break
            if current_obstacle is None:
                continue
            
            # 双判据动态识别（如需启用，取消注释）
            is_dynamic = self._dynamic_identification(track_id, current_obstacle)
            if not is_dynamic:
                continue
            
            # 提取输出信息
            kf = track['state']
            vx, vy = kf.x[2], kf.x[3]
            tracked_dynamic_obstacles[track_id] = {
                'rect': current_obstacle['rect_params'],
                'box': current_obstacle['box'],
                'velocity': (vx, vy),
                'confidence': np.sqrt(vx**2 + vy**2) / (2*self.vel_threshold)  # 动态置信度（0-1）
            }
        
        return tracked_dynamic_obstacles