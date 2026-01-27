import numpy as np
import math

class MppiplanSolver:
    def __init__(self, 
                delta_t: float = 0.1,
                max_omega_abs: float = 3.0, # 最大角速度
                max_vel_abs: float = 5.0,   # 最大线速度
                ref_path: np.ndarray = np.array([[0.0, 0.0, 0.0, 1.0], [10.0, 0.0, 0.0, 1.0]]),
                horizon_step_T: int = 20,
                number_of_samples_K: int = 200,
                param_exploration: float = 0.0,
                param_lambda: float = 50.0,
                param_alpha: float = 1.0,
                sigma: np.ndarray = np.array([[0.5, 0.0], [0.0, 0.3]]), # [v_noise, w_noise]
                stage_cost_weight: np.ndarray = np.array([20.0, 20.0, 20.0, 1.0]),
                terminal_cost_weight: np.ndarray = np.array([50.0, 50.0, 50.0, 1.0]),
                w_obs: float = 100.0,
                safe_distance: float = 0.5,
                visualize_optimal_traj: bool = True,
                visualze_sampled_trajs: bool = True,
                **kwargs # 吸收多余参数
                 ):
        """初始化纯NumPy的MPPI参数 (差分模型)"""
        self.dim_x = 4 # [x, y, theta, last_omega]
        self.dim_u = 2 # [v, omega]
        self.T = horizon_step_T
        self.K = number_of_samples_K

        # 算法参数
        self.param_exploration = param_exploration
        self.param_lambda = param_lambda
        self.param_alpha = param_alpha
        self.param_gamma = self.param_lambda * (1.0 - self.param_alpha)
        
        # 协方差矩阵
        self.Sigma = sigma.astype(np.float32)
        self.inv_Sigma = np.linalg.inv(self.Sigma)
        
        # 成本权重
        self.stage_cost_weight = stage_cost_weight.astype(np.float32)
        self.terminal_cost_weight = terminal_cost_weight.astype(np.float32)
        self.w_obs = w_obs
        self.safe_distance = safe_distance
        
        # 车辆参数
        self.delta_t = delta_t
        self.max_omega_abs = max_omega_abs
        self.max_vel_abs = max_vel_abs
        
        # 参考路径
        self.ref_path = ref_path.astype(np.float32)
        
        # 如果输入路径只有 [x, y, theta] 3列
        if self.ref_path.ndim == 2 and self.ref_path.shape[1] == 3:
            print(f"[INFO] ref_path shape is {self.ref_path.shape}. Auto-appending max_vel ({max_vel_abs} m/s) as reference velocity.")
            # 创建一列全为 max_vel_abs 的速度
            v_ref = np.full((self.ref_path.shape[0], 1), max_vel_abs/2, dtype=np.float32)
            # 拼接到路径后面 -> [x, y, theta, v]
            self.ref_path = np.hstack([self.ref_path, v_ref])
        
        # 控制序列初始化 [T, 2]
        self.u_prev = np.zeros((self.T, self.dim_u), dtype=np.float32)
        
        self.prev_waypoints_idx = 0
        self.visualize_optimal_traj = visualize_optimal_traj
        self.visualze_sampled_trajs = visualze_sampled_trajs

    def calc_control_input(self, observed_x: np.ndarray, obstacles: list = None):
        """
        计算最优控制量
        :param observed_x: [x, y, theta]
        :param obstacles: 障碍物列表
        """
        # 状态预处理: 确保是 [1, 4] 形状
        obs_flat = observed_x.flatten()
        if len(obs_flat) >= 3:
            # 补一位 0.0 作为 omega 初始状态
            x0 = np.concatenate([obs_flat[:3], [0.0]]).reshape(1, 4).astype(np.float32)
        else:
            raise ValueError("Observed state dimension too small")

        # 查找最近点
        self._get_nearest_waypoint(x0[0, 0], x0[0, 1], update_prev_idx=True)
        
        # 障碍物转 NumPy 数组
        obs_array = None
        if obstacles is not None and len(obstacles) > 0:
            obs_array = np.array(obstacles, dtype=np.float32) # [M, 7]
        
        # 1. 生成噪声 [K, T, 2]
        epsilon = self._calc_epsilon() 
        
        # 2. 生成控制输入候选
        # 重要性采样 + 探索性采样
        u_expanded = np.tile(self.u_prev, (self.K, 1, 1)) # [K, T, 2]
        v = np.zeros_like(u_expanded)
        
        # 区分探索比例
        num_explore = int(self.K * self.param_exploration)
        num_exploit = self.K - num_explore
        
        if num_exploit > 0:
            v[:num_exploit] = u_expanded[:num_exploit] + epsilon[:num_exploit]
        if num_explore > 0:
            v[num_exploit:] = epsilon[num_exploit:]
        
        # 3. 批量推演并计算成本
        S = self._batch_compute_costs(x0, v, self.u_prev, obs_array)
        
        # 4. 计算权重
        w = self._compute_weights(S) # [K]
        
        # 5. 加权更新控制序列
        # w: [K], epsilon: [K, T, 2] -> sum(w * eps) -> [T, 2]
        w_expanded = w.reshape(-1, 1, 1)
        w_epsilon = np.sum(w_expanded * epsilon, axis=0)
        
        # 6. 平滑控制增量
        w_epsilon = self._moving_average_filter(w_epsilon, window_size=10)
        
        # 更新 self.u_prev
        self.u_prev += w_epsilon
        
        # 7. 可视化数据准备
        optimal_traj = None
        sampled_traj_list = None
        
        if self.visualize_optimal_traj:
            optimal_traj = self._compute_optimal_trajectory(x0, self.u_prev)
            
        if self.visualze_sampled_trajs:
            sampled_traj_list = self._compute_sampled_trajectories(x0, v)
        
        # 8. 滚动更新 (Shift)
        current_u = self.u_prev[0].copy()
        self.u_prev[:-1] = self.u_prev[1:]
        self.u_prev[-1] = self.u_prev[-1] # 保持最后一个控制量
        
        return optimal_traj, sampled_traj_list, current_u

    def _calc_epsilon(self):
        """生成多元正态分布噪声"""
        mean = np.zeros(self.dim_u)
        # 生成 [K, T, 2]
        epsilon = np.random.multivariate_normal(mean, self.Sigma, size=(self.K, self.T))
        return epsilon.astype(np.float32)

    def _batch_compute_costs(self, x0, v, u_base, obs_array):
        """批量推演并计算成本"""
        K, T = self.K, self.T
        total_cost = np.zeros(K, dtype=np.float32)
        
        # 初始化状态 [K, 4]
        x = np.tile(x0, (K, 1))
        
        for t in range(T):
            # 限幅控制量
            u_clamped = self._g(v[:, t, :])
            
            # 运动学更新
            x = self._F(x, u_clamped)
            
            # 阶段成本
            current_vel = u_clamped[:, 0]
            stage_cost = self._c(x, current_vel, obs_array) # [K]
            
            # 控制动作平滑成本/能量成本
            # u_base[t]: [2], v[:, t]: [K, 2]
            # diff: [K, 2]
            # term: u^T * inv_Sigma * v
            # 这里 MPPI 原始推导中通常是 (v - u) 的相关项，或者是扰动 epsilon 的成本
            # 按照原代码逻辑: u_base[t] @ inv_Sigma @ v[:, t]
            
            # 向量化矩阵乘法: (K, 1, 2) @ (1, 2, 2) @ (K, 2, 1) -> (K, 1, 1) -> (K)
            # 或者简单的: np.sum((u @ inv) * v, axis=1)
            
            u_t = u_base[t] # [2]
            v_t = v[:, t, :] # [K, 2]
            
            # term = u_t^T * inv_Sigma * v_t
            # (2,) * (2,2) -> (2,)
            temp = u_t @ self.inv_Sigma 
            # (2,) * (K, 2)^T -> (K,)
            control_cost_term = np.dot(v_t, temp) 
            
            control_cost = self.param_gamma * control_cost_term
            
            total_cost += stage_cost + control_cost
        
        # 终端成本
        total_cost += self._phi(x, current_vel)
        return total_cost

    def _F(self, x, v_in):
        """
        差分模型运动学
        x: [K, 4] (x, y, theta, w)
        v_in: [K, 2] (v, w)
        """
        x_pos = x[:, 0]
        y_pos = x[:, 1]
        theta = x[:, 2]
        
        vel = v_in[:, 0]
        omega = v_in[:, 1]
        
        dt = self.delta_t
        
        new_x = x_pos + vel * np.cos(theta) * dt
        new_y = y_pos + vel * np.sin(theta) * dt
        new_theta = theta + omega * dt
        new_theta = (new_theta + 2 * math.pi) % (2 * math.pi)
        
        # 存储当前 omega 状态
        new_w = omega
        
        return np.stack([new_x, new_y, new_theta, new_w], axis=1)

    def _g(self, v):
        """控制量限幅"""
        v_clamped = v.copy()
        v_clamped[:, 0] = np.clip(v_clamped[:, 0], -self.max_vel_abs, self.max_vel_abs)
        v_clamped[:, 1] = np.clip(v_clamped[:, 1], -self.max_omega_abs, self.max_omega_abs)
        return v_clamped

    def _c(self, x, current_vel, obs_array=None):
        """计算阶段成本 (Path + Obstacle)"""
        x_pos = x[:, 0]
        y_pos = x[:, 1]
        theta = x[:, 2]
        theta = (theta + 2 * math.pi) % (2 * math.pi)
        
        # 1. 路径成本
        _, ref_x, ref_y, ref_theta, ref_v = self._batch_get_nearest_waypoint(x_pos, y_pos)
        
        dx = x_pos - ref_x
        dy = y_pos - ref_y
        d_theta = theta - ref_theta
        # 角度差归一化到 [-pi, pi]
        d_theta = (d_theta + np.pi) % (2 * np.pi) - np.pi
        
        d_vel = current_vel - ref_v
        
        path_cost = (
            self.stage_cost_weight[0] * dx**2 +
            self.stage_cost_weight[1] * dy**2 +
            self.stage_cost_weight[2] * d_theta**2 +
            self.stage_cost_weight[3] * d_vel**2
        )
        
        # 2. 障碍物成本
        obs_cost = np.zeros_like(path_cost)
        
        if obs_array is not None:
            # x_pos: [K], cx: [M]
            # Broadcasting: [K, 1] - [1, M] = [K, M]
            
            cx = obs_array[:, 0]
            cy = obs_array[:, 1]
            a  = obs_array[:, 2]
            b  = obs_array[:, 3]
            cos_ot = obs_array[:, 5]
            sin_ot = obs_array[:, 6]
            
            safe_a = a + self.safe_distance
            safe_b = b + self.safe_distance
            
            # [K, M]
            dx_vec = x_pos[:, None] - cx[None, :]
            dy_vec = y_pos[:, None] - cy[None, :]
            
            lx = dx_vec * cos_ot[None, :] + dy_vec * sin_ot[None, :]
            ly = -dx_vec * sin_ot[None, :] + dy_vec * cos_ot[None, :]
            
            dist_sq_matrix = (lx / safe_a[None, :])**2 + (ly / safe_b[None, :])**2
            
            # 找到每条轨迹距离所有障碍物中最近的那个距离 (min over M)
            min_dist_sq = np.min(dist_sq_matrix, axis=1) # [K]
            
            collision_mask = min_dist_sq <= 1.0
            
            # 硬约束
            obs_cost[collision_mask] += 100000.0
            
            # 软约束
            safe_mask = ~collision_mask
            if np.any(safe_mask):
                obs_cost[safe_mask] += self.w_obs / (min_dist_sq[safe_mask] + 1e-6)
                
        return path_cost + obs_cost

    def _batch_get_nearest_waypoint(self, x_pos, y_pos):
        """NumPy 批量最近点查找"""
        prev_idx = self.prev_waypoints_idx
        # 限制搜索窗口
        end_idx = min(prev_idx + 200, self.ref_path.shape[0])
        ref_segment = self.ref_path[prev_idx:end_idx] # [N_seg, 4]
        
        # x_pos: [K] -> [K, 1]
        # ref_x: [N_seg] -> [1, N_seg]
        dx = x_pos[:, None] - ref_segment[:, 0][None, :]
        dy = y_pos[:, None] - ref_segment[:, 1][None, :]
        dist_sq = dx**2 + dy**2 # [K, N_seg]
        
        min_indices = np.argmin(dist_sq, axis=1) # [K]
        nearest_indices = prev_idx + min_indices
        
        # 获取对应点
        ref_x = ref_segment[min_indices, 0]
        ref_y = ref_segment[min_indices, 1]
        ref_theta = ref_segment[min_indices, 2]
        ref_v = ref_segment[min_indices, 3]
        
        return nearest_indices, ref_x, ref_y, ref_theta, ref_v

    def _phi(self, x, current_vel):
        """终端成本"""
        x_pos = x[:, 0]
        y_pos = x[:, 1]
        theta = x[:, 2]
        theta = (theta + 2 * math.pi) % (2 * math.pi)
        
        _, ref_x, ref_y, ref_theta, ref_v = self._batch_get_nearest_waypoint(x_pos, y_pos)
        
        dx = x_pos - ref_x
        dy = y_pos - ref_y
        d_theta = theta - ref_theta
        d_theta = (d_theta + np.pi) % (2 * np.pi) - np.pi
        d_vel = current_vel - ref_v
        
        return (
            self.terminal_cost_weight[0] * dx**2 +
            self.terminal_cost_weight[1] * dy**2 +
            self.terminal_cost_weight[2] * d_theta**2 +
            self.terminal_cost_weight[3] * d_vel**2
        )

    def _compute_weights(self, S):
        rho = np.min(S)
        # 减去最小值防止指数爆炸
        exp_terms = np.exp((-1.0 / self.param_lambda) * (S - rho))
        eta = np.sum(exp_terms)
        return exp_terms / eta

    def _moving_average_filter(self, xx, window_size=10):
        """NumPy 版滑动平均滤波"""
        T, dim = xx.shape
        xx_mean = np.zeros_like(xx)
        kernel = np.ones(window_size) / window_size
        
        n_left = window_size // 2 - 1
        n_right = window_size // 2
        
        for d in range(dim):
            x = xx[:, d]
            # 'valid' 模式卷积
            conv_out = np.convolve(x, kernel, mode='valid')
            
            # 手动补全边缘
            left_pad = np.full(n_left, conv_out[0])
            right_pad = np.full(n_right, conv_out[-1])
            padded_out = np.concatenate([left_pad, conv_out, right_pad])
            
            # 边缘加权修正 (模仿原代码逻辑)
            n_conv = math.ceil(window_size / 2)
            padded_out[0] *= window_size / n_conv
            for i in range(1, n_conv):
                if i < len(padded_out):
                    padded_out[i] *= window_size / (i + n_conv)
                if i < len(padded_out):
                    padded_out[-i] *= window_size / (i + n_conv - (window_size % 2))
            
            # 确保长度一致 (防止奇偶窗口导致的1位误差)
            if len(padded_out) > T:
                padded_out = padded_out[:T]
            elif len(padded_out) < T:
                padded_out = np.pad(padded_out, (0, T - len(padded_out)), mode='edge')
                
            xx_mean[:, d] = padded_out
        return xx_mean

    def _get_nearest_waypoint(self, x, y, update_prev_idx=False):
        """单点最近邻查找"""
        prev_idx = self.prev_waypoints_idx
        end_idx = min(prev_idx + 20, self.ref_path.shape[0])
        ref_segment = self.ref_path[prev_idx:end_idx]
        
        dx = x - ref_segment[:, 0]
        dy = y - ref_segment[:, 1]
        dist_sq = dx**2 + dy**2
        min_idx = np.argmin(dist_sq)
        nearest_idx = prev_idx + min_idx
        
        if update_prev_idx:
            self.prev_waypoints_idx = nearest_idx
            
        return nearest_idx

    def _compute_optimal_trajectory(self, x0, u):
        """计算最优轨迹 (用于可视化)"""
        traj = np.zeros((self.T, self.dim_x), dtype=np.float32)
        x = x0.copy() # [1, 4]
        
        for t in range(self.T):
            u_curr = u[t].reshape(1, 2)
            x = self._F(x, u_curr)
            traj[t] = x[0]
        return traj

    def _compute_sampled_trajectories(self, x0, v):
        """计算所有采样轨迹 (用于可视化)"""
        K, T = self.K, self.T
        traj = np.zeros((K, T, self.dim_x), dtype=np.float32)
        x = np.tile(x0, (K, 1))
        
        for t in range(T):
            u_clamped = self._g(v[:, t, :])
            x = self._F(x, u_clamped)
            traj[:, t, :] = x
        return traj