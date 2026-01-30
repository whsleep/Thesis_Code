import numpy as np
import math

class AccDwaSolver:
    """
    基于加速度采样的全向 DWA 规划器
    状态输入仅为 [x, y, theta]，其余状态通过差分估算。
    """
    def __init__(self, 
                 ref_path: np.ndarray,
                 delta_t: float = 0.1,
                 v_max: float = 5.0, 
                 omega_max: float = 3.0, 
                 a_limit: float = 2.0, 
                 sample_count: int = 10,
                 predict_time: float = 2.0,
                 w_heading: float = 0.2,
                 w_dist: float = 0.4,
                 w_vel: float = 0.4,
                 w_obs: float = 20.0,
                 w_track: float = 0.3,
                 safe_dist: float = 1.5,
                 lookahead_dist: float = 5.0,
                 **kwargs):
        # 路径与车辆参数
        self.ref_path = ref_path
        self.dt = delta_t
        self.v_max = v_max
        self.omega_max = omega_max
        self.a_limit = a_limit

        # 加速度采样参数
        self.predict_time = predict_time
        self.sample_count = sample_count
        self.N = int(self.predict_time / self.dt)

        # 评价参数
        self.w_heading = w_heading
        self.w_dist = w_dist
        self.w_vel = w_vel
        self.w_obs = w_obs
        self.w_track = w_track

        self.safe_dist = safe_dist
        
        # 状态记录 (用于差分求解)
        self.last_state = None  # [x, y, theta]
        self.curr_v = 0.0
        self.curr_w = 0.0

        # 路径跟踪参数
        self.prev_idx = 0
        self.lookahead_dist = lookahead_dist

    def calc_control_input(self, observed_x: np.ndarray, obstacles: list = None):
        """
        observed_x: [x, y, theta]
        obstacles: 动态障碍物预测信息 [[x, y, vx, vy, theta], ...]
        """
        # 1. 状态差分求解: 获取当前 v 和 w
        if self.last_state is not None:
            dx = observed_x[0] - self.last_state[0]
            dy = observed_x[1] - self.last_state[1]
            dtheta = (observed_x[2] - self.last_state[2] + math.pi) % (2 * math.pi) - math.pi
            
            # 线速度 (沿当前航向的分量)
            dist = math.hypot(dx, dy)
            self.curr_v = dist / self.dt
            # 角速度
            self.curr_w = dtheta / self.dt
        
        self.last_state = observed_x.copy()
        
        # 2. 准备预测初始态 (全向模型 [x, y, vx, vy])
        x, y, theta = observed_x.flatten()
        vx0 = self.curr_v * math.cos(theta)
        vy0 = self.curr_v * math.sin(theta)
        h_state0 = np.array([x, y, vx0, vy0])
        
        # 3. 确定局部目标
        goal = self._get_local_goal_from_path(x, y)
        
        # 4. 加速度空间采样与评价
        acc_range = np.linspace(-self.a_limit, self.a_limit, self.sample_count)
        best_u_acc = np.array([0.0, 0.0])
        min_cost = float('inf')
        all_trajectories = []
        best_traj = None

        for ax in acc_range:
            for ay in acc_range:
                # 预测全向轨迹
                traj_h = self._predict_holonomic_traj(h_state0, ax, ay)
                all_trajectories.append(traj_h[:, :2])
                
                # 代价计算 
                cost = self._evaluate_traj(traj_h, goal, obstacles)
                
                if cost < min_cost:
                    min_cost = cost
                    best_u_acc = np.array([ax, ay])
                    best_traj = traj_h[:, :2]

        # 5. 转换加速度到控制量 [av, aw]
        control_u = self._convert_acc_to_control(best_u_acc, h_state0, theta)

        if best_traj is None:
            best_traj = np.zeros((self.N, 2))
        
        return best_traj, np.array(all_trajectories), control_u

    def _predict_holonomic_traj(self, h0, ax, ay):
        steps = self.N
        t_seq = np.arange(1, steps + 1) * self.dt
        traj = np.zeros((steps, 4))
        
        # x = x0 + v0*t + 0.5*a*t^2
        traj[:, 0] = h0[0] + h0[2] * t_seq + 0.5 * ax * t_seq**2
        traj[:, 1] = h0[1] + h0[3] * t_seq + 0.5 * ay * t_seq**2
        traj[:, 2] = h0[2] + ax * t_seq
        traj[:, 3] = h0[3] + ay * t_seq
        return traj

    def _evaluate_traj(self, traj_h, goal, obstacles):
        """
        完全参考 DWA 评价体系，并集成椭圆障碍物碰撞检测
        """
        # --- 1. 基础状态提取 ---
        end_x, end_y = traj_h[-1, 0], traj_h[-1, 1]
        vx_end, vy_end = traj_h[-1, 2], traj_h[-1, 3]
        
        # --- 2. 航向代价 (Heading) ---
        # 目标方位角与当前预测末端速度方向的偏差
        goal_theta = math.atan2(goal[1] - end_y, goal[0] - end_x)
        traj_theta = math.atan2(vy_end, vx_end)
        angle_diff = abs((goal_theta - traj_theta + math.pi) % (2 * math.pi) - math.pi)
        cost_heading = angle_diff 

        # --- 3. 距离代价 (Distance) ---
        cost_dist = math.hypot(goal[0] - end_x, goal[1] - end_y)

        # --- 4. 速度代价 (Velocity) ---
        v_final = math.hypot(vx_end, vy_end)
        cost_vel = self.v_max - v_final

        # --- 5. 路径跟踪代价 (Tracking) ---
        cost_track = 0.0
        # traj_h[:-1] 排除最后一个点
        track_points = traj_h[:-1]
        if len(track_points) > 0:
            tx = track_points[:, 0]
            ty = track_points[:, 1]
            
            # 批量获取距离最近的参考点距离平方
            dist_sq_vals = self._batch_get_nearest_dist_sq(tx, ty)
            
            # 计算均方误差 (MSE) 作为代价
            cost_track = np.mean(dist_sq_vals)

        # --- 6. 障碍物碰撞评价 (基于椭圆方程) ---
        cost_obs = 0.0
        min_ellipse_dist_sq = float('inf') # 记录全轨迹到障碍物的最小代数距离

        if obstacles is not None and len(obstacles) > 0:
            # 轨迹点坐标
            traj_x = traj_h[:, 0]
            traj_y = traj_h[:, 1]

            for obs in obstacles:
                # 提取椭圆参数: [cx, cy, a, b, theta, cos, sin]
                # 注意：如果输入格式不同，请确保索引匹配
                cx, cy, a, b, _, cos_ot, sin_ot = obs
                
                # 膨胀椭圆长短轴
                safe_a = a + self.safe_dist
                safe_b = b + self.safe_dist
                
                # 计算轨迹点相对于椭圆中心的坐标
                dx_vec = traj_x - cx
                dy_vec = traj_y - cy
                
                # 旋转到椭圆局部坐标系
                lx = dx_vec * cos_ot + dy_vec * sin_ot
                ly = -dx_vec * sin_ot + dy_vec * cos_ot
                
                # 计算代数距离 (椭圆方程: (x/a)^2 + (y/b)^2)
                dist_sq = (lx / safe_a)**2 + (ly / safe_b)**2
                
                # 获取该轨迹上的最小值
                current_min_dist = np.min(dist_sq)

                # 记录全局最小代数距离用于软约束
                if current_min_dist < min_ellipse_dist_sq:
                    min_ellipse_dist_sq = current_min_dist

            # --- 软约束：避障评分 ---
            # 距离障碍物越近，代价越高
            cost_obs = 1.0 / (min_ellipse_dist_sq + 1e-6)

        # --- 7. 总代价加权 ---
        total_cost = (self.w_heading * cost_heading +
                      self.w_dist * cost_dist +
                      self.w_vel * cost_vel +
                      self.w_track * cost_track +
                      self.w_obs * cost_obs)
        
        return total_cost

    def _convert_acc_to_control(self, best_acc, h0, curr_theta):
        ax, ay = best_acc
        vx, vy = h0[2], h0[3]
        
        # 计算期望角速度 w
        v_sq = vx**2 + vy**2 + 1e-6
        target_w = (vx * ay - vy * ax) / v_sq
        
        # 计算线加速度 av (在当前航向上的投影)
        av = math.sqrt(ax**2 + ay**2) * math.cos(math.atan2(ay, ax) - curr_theta)
        target_v = math.sqrt(v_sq) + av * self.dt
        # 限制范围
        return np.array([np.clip(target_v, -self.v_max, self.v_max), 
                         np.clip(target_w, -self.omega_max, self.omega_max)])

    def _get_local_goal_from_path(self, x, y):
        """寻找前瞻点"""
        search_range = 10 
        start_search = max(0, self.prev_idx - 10)
        end_search = min(len(self.ref_path), self.prev_idx + search_range)
        
        path_segment = self.ref_path[start_search:end_search]
        if len(path_segment) == 0:
            return self.ref_path[-1] 
            
        dx = x - path_segment[:, 0]
        dy = y - path_segment[:, 1]
        dist_sq = dx**2 + dy**2
        min_idx_local = np.argmin(dist_sq)
        min_idx_global = start_search + min_idx_local
        
        self.prev_idx = min_idx_global 
        
        lookahead_idx = min_idx_global
        current_dist = 0.0
        while lookahead_idx < len(self.ref_path) - 1:
            p1 = self.ref_path[lookahead_idx]
            p2 = self.ref_path[lookahead_idx + 1]
            d = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
            current_dist += d
            if current_dist >= self.lookahead_dist:
                break
            lookahead_idx += 1
            
        return self.ref_path[lookahead_idx]
    
    def _batch_get_nearest_dist_sq(self, x_arr, y_arr):
        """
        仿照 MPPI 的批量最近点查找，计算点集到参考路径的最近距离平方
        input:
            x_arr: [N] 轨迹点的 x 坐标
            y_arr: [N] 轨迹点的 y 坐标
        output:
            min_dist_sq: [N] 每个点对应的最近距离平方
        """
        # 使用 prev_idx 限制搜索范围，提高效率
        search_range = 50
        start_idx = max(0, self.prev_idx - 5)
        end_idx = min(len(self.ref_path), self.prev_idx + search_range)
        
        ref_segment = self.ref_path[start_idx:end_idx] # [M, 2+]
        
        if len(ref_segment) == 0:
            return np.zeros_like(x_arr)
        
        # 利用广播机制计算距离矩阵
        # x_arr: [N] -> [N, 1]
        # ref_x: [M] -> [1, M]
        dx = x_arr[:, None] - ref_segment[:, 0][None, :]
        dy = y_arr[:, None] - ref_segment[:, 1][None, :]
        
        # 距离平方矩阵 [N, M]
        dist_sq_matrix = dx**2 + dy**2
        
        # 沿轴 1 (ref_path方向) 取最小值，得到每个轨迹点到路径的最短距离平方
        min_dist_sq = np.min(dist_sq_matrix, axis=1) # [N]
        
        return min_dist_sq