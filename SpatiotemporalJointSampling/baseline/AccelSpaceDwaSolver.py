import numpy as np
import math

class AccelSpaceDwaSolver:
    """
    重构版加速度空间 DWA 求解器
    特点：
    1. 保持原有的 (ax, ay) 笛卡尔加速度采样逻辑。
    2. 采用 MPPI 风格的 NumPy 矩阵运算进行批量轨迹推演。
    3. 过滤掉碰撞轨迹，使其不出现在返回的采样列表中。
    """
    def __init__(self, 
                 ref_path: np.ndarray,
                 delta_t: float = 0.1,
                 v_max: float = 5.0,
                 omega_max: float = 3.0,
                 a_limit: float = 2.0,    # 加速度采样范围 [-A, A]
                 sample_count: int = 7,  # 采样分辨率 (总样本数 = count^2)
                 predict_time: float = 2.0,
                 w_heading: float = 0.2,
                 w_dist: float = 0.5,
                 w_vel: float = 0.2,
                 w_obs: float = 20.0,
                 w_track: float = 1.0,    # 路径贴合代价
                 safe_distance: float = 0.4,
                 lookahead_dist: float = 5.0,
                 **kwargs):
        
        # 1. 基础参数
        self.ref_path = ref_path
        self.dt = delta_t
        self.predict_time = predict_time
        self.predict_steps = int(predict_time / delta_t)
        
        # 2. 约束参数
        self.v_max = v_max
        self.omega_max = omega_max
        self.a_limit = a_limit
        self.sample_count = sample_count
        
        # 3. 权重参数
        self.w_heading = w_heading
        self.w_dist = w_dist
        self.w_vel = w_vel
        self.w_obs = w_obs
        self.w_track = w_track
        self.safe_distance = safe_distance
        
        # 4. 路径跟踪状态
        self.lookahead_dist = lookahead_dist
        self.prev_idx = 0
        
        # 5. 内部动力学状态
        self.curr_v = 0.0
        self.curr_vx = 0.0
        self.curr_vy = 0.0

    def calc_control_input(self, observed_x: np.ndarray, obstacles: list = None):
        """
        标准接口：计算控制量
        input: 
            observed_x: [x, y, theta]
            obstacles: [cx, cy, a, b, theta, cos, sin] (椭圆列表)
        """
        if observed_x.ndim > 1:
            observed_x = observed_x.flatten()
        
        x, y, theta = observed_x
        
        # 1. 获取局部目标点
        local_goal = self._get_local_goal_from_path(x, y)
        
        # 2. 生成加速度采样网格 (Vectorized)
        ax_samples = np.linspace(-self.a_limit, self.a_limit, self.sample_count)
        ay_samples = np.linspace(-self.a_limit, self.a_limit, self.sample_count)
        AX, AY = np.meshgrid(ax_samples, ay_samples)
        ax_flat = AX.flatten()
        ay_flat = AY.flatten()
        
        # 3. 批量计算动力学更新
        accel_mag = np.sqrt(ax_flat**2 + ay_flat**2)
        accel_dir = np.arctan2(ay_flat, ax_flat)
        angle_diff = accel_dir - theta
        av = accel_mag * np.cos(angle_diff)
        
        # 更新线速度 v
        next_v = self.curr_v + av * self.dt
        next_v = np.clip(next_v, -self.v_max, self.v_max)
        
        # 更新角速度 w
        vx_curr = self.curr_v * np.cos(theta)
        vy_curr = self.curr_v * np.sin(theta)
        cross_prod = vx_curr * ay_flat - vy_curr * ax_flat
        v_sq = self.curr_v ** 2
        
        next_w = np.where(
            v_sq > 0.01, 
            cross_prod / v_sq, 
            0.0 
        )
        next_w = np.clip(next_w, -self.omega_max, self.omega_max)

        # 4. 批量轨迹推演 (此时包含所有轨迹，包括碰撞的)
        trajectories = self._predict_trajectories_batch(
            x, y, self.curr_vx, self.curr_vy, ax_flat, ay_flat
        )
        
        # 5. 批量计算代价 (碰撞的代价被设为 np.inf)
        costs = self._calc_costs_batch(trajectories, local_goal, next_v, obstacles)
        
        # --- [新增逻辑] 6. 过滤碰撞轨迹 ---
        
        # 创建掩码：代价不为无穷大即为有效轨迹
        valid_mask = costs != float('inf')
        
        best_traj = None
        best_u = np.array([0.0, 0.0])
        best_accel = np.array([0.0, 0.0])
        
        if np.any(valid_mask):
            # 存在无碰撞的解
            
            # 利用掩码过滤数据
            valid_costs = costs[valid_mask]
            valid_trajs = trajectories[valid_mask]
            valid_v = next_v[valid_mask]
            valid_w = next_w[valid_mask]
            valid_ax = ax_flat[valid_mask]
            valid_ay = ay_flat[valid_mask]
            
            # 在有效集中寻找最优
            best_idx = np.argmin(valid_costs)
            
            best_traj = valid_trajs[best_idx]
            best_u = np.array([valid_v[best_idx], valid_w[best_idx]])
            best_accel = np.array([valid_ax[best_idx], valid_ay[best_idx]])
            
            # 【关键】更新 trajectories 列表，只返回无碰撞的轨迹
            trajectories = valid_trajs
            
        else:
            # 所有路径都碰撞：执行紧急制动 (Emergency Stop)
            # 或者原地停车，不更新状态
            print("[AccelSpaceDWA] Warning: All trajectories collided! Executing emergency stop.")
            best_u = np.array([0.0, 0.0])
            best_accel = np.array([0.0, 0.0])
            
            # 生成一条原地不动的轨迹用于可视化/输出
            best_traj = np.tile(observed_x, (self.predict_steps, 1))
            
            # 返回空列表，表示没有可用的采样轨迹
            trajectories = np.zeros((0, self.predict_steps, 3))
            
            # 强制重置速度，防止惯性继续前冲
            self.curr_v = 0.0
            self.curr_vx = 0.0
            self.curr_vy = 0.0

        # 7. 更新内部状态 (如果未完全碰撞)
        if np.any(valid_mask):
            self.curr_v = best_u[0]
            self.curr_vx += best_accel[0] * self.dt
            self.curr_vy += best_accel[1] * self.dt
            
            # 限制模长
            vel_norm = np.hypot(self.curr_vx, self.curr_vy)
            if vel_norm > self.v_max:
                scale = self.v_max / vel_norm
                self.curr_vx *= scale
                self.curr_vy *= scale
        
        return best_traj, trajectories, best_u

    def _predict_trajectories_batch(self, x0, y0, vx0, vy0, ax_flat, ay_flat):
        """批量推演轨迹 (逻辑不变)"""
        K = len(ax_flat)
        T = self.predict_steps
        
        t_steps = np.arange(1, T + 1) * self.dt
        t_steps = t_steps.reshape(1, T) 
        
        ax = ax_flat[:, None]
        ay = ay_flat[:, None]
        
        px = x0 + vx0 * t_steps + 0.5 * ax * (t_steps**2)
        py = y0 + vy0 * t_steps + 0.5 * ay * (t_steps**2)
        
        vxt = vx0 + ax * t_steps
        vyt = vy0 + ay * t_steps
        theta_t = np.arctan2(vyt, vxt)
        
        traj = np.stack([px, py, theta_t], axis=2)
        return traj

    def _calc_costs_batch(self, trajs, goal, v_sample, obstacles):
        """批量计算代价 (逻辑不变，碰撞返回 inf)"""
        K = trajs.shape[0]
        end_points = trajs[:, -1, :]
        
        dx = goal[0] - end_points[:, 0]
        dy = goal[1] - end_points[:, 1]
        target_theta = np.arctan2(dy, dx)
        angle_diff = target_theta - end_points[:, 2]
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
        cost_heading = np.abs(angle_diff)
        
        cost_dist = np.hypot(dx, dy)
        cost_vel = self.v_max - np.abs(v_sample)
        
        traj_pts_flat = trajs[:, :, :2].reshape(-1, 2)
        track_dist_sq = self._batch_get_nearest_dist_sq(traj_pts_flat)
        track_dist_sq = track_dist_sq.reshape(K, -1)
        cost_track = np.mean(track_dist_sq, axis=1)
        
        cost_obs = np.zeros(K)
        if obstacles is not None and len(obstacles) > 0:
            obs_arr = np.array(obstacles)
            tx = trajs[:, :, 0]
            ty = trajs[:, :, 1]
            min_dist_vals = np.full(K, np.inf)
            
            for obs in obs_arr:
                cx, cy, a, b, _, cos_ot, sin_ot = obs
                safe_a = a + self.safe_distance
                safe_b = b + self.safe_distance
                
                dx_v = tx - cx
                dy_v = ty - cy
                lx = dx_v * cos_ot + dy_v * sin_ot
                ly = -dx_v * sin_ot + dy_v * cos_ot
                dist_sq = (lx / safe_a)**2 + (ly / safe_b)**2
                
                dist_min_obs = np.min(dist_sq, axis=1)
                
                # 若发生碰撞，代价设为无穷大
                collision_mask = dist_min_obs <= 1.0
                cost_obs[collision_mask] = np.inf
                
                min_dist_vals = np.minimum(min_dist_vals, dist_min_obs)
            
            valid_mask = (cost_obs != np.inf) & (min_dist_vals != np.inf)
            if np.any(valid_mask):
                cost_obs[valid_mask] += 1.0 / (min_dist_vals[valid_mask] + 1e-6)

        total_cost = (self.w_heading * cost_heading +
                      self.w_dist * cost_dist +
                      self.w_vel * cost_vel +
                      self.w_track * cost_track +
                      self.w_obs * cost_obs)
        
        return total_cost

    def _get_local_goal_from_path(self, x, y):
        # ... (保持不变) ...
        search_range = 20
        start_search = max(0, self.prev_idx - 10)
        end_search = min(len(self.ref_path), self.prev_idx + search_range)
        path_segment = self.ref_path[start_search:end_search]
        if len(path_segment) == 0: return self.ref_path[-1]
        dx = x - path_segment[:, 0]
        dy = y - path_segment[:, 1]
        dist_sq = dx**2 + dy**2
        min_idx_global = start_search + np.argmin(dist_sq)
        self.prev_idx = min_idx_global
        lookahead_idx = min_idx_global
        current_dist = 0.0
        while lookahead_idx < len(self.ref_path) - 1:
            p1 = self.ref_path[lookahead_idx]
            p2 = self.ref_path[lookahead_idx + 1]
            d = np.hypot(p2[0]-p1[0], p2[1]-p1[1])
            current_dist += d
            if current_dist >= self.lookahead_dist: break
            lookahead_idx += 1
        return self.ref_path[lookahead_idx]

    def _batch_get_nearest_dist_sq(self, points):
        # ... (保持不变) ...
        search_range = 50
        start_idx = max(0, self.prev_idx - 10)
        end_idx = min(len(self.ref_path), self.prev_idx + search_range)
        ref_segment = self.ref_path[start_idx:end_idx]
        if len(ref_segment) == 0: return np.zeros(len(points))
        diff = points[:, None, :] - ref_segment[None, :, :2]
        dist_sq_mat = np.sum(diff**2, axis=2)
        min_dist_sq = np.min(dist_sq_mat, axis=1)
        return min_dist_sq