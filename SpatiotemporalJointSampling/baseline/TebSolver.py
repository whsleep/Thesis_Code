import casadi as ca
import numpy as np

class TebplanSolver:
    """
    自适应 TEB 规划器 (修复版)
    """
    def __init__(self, 
                 ref_path: np.ndarray,
                 safe_distance: float = 0.5,  # 稍微调小，避免过于保守
                 v_max=5.0, omega_max=2.0, a_max=2.0, 
                 lookahead_dist=5.0,
                 # 权重参数
                 w_p=0.5, w_t=2.0, w_kin=50.0, w_r=5.0, w_obs=20.0, w_goal=1.0,w_acc=0.1,w_theta =0.1, 
                 T_min=0.1, T_max=0.4):
        
        # --- 1. 基础配置 ---
        self.ref_path = ref_path
        self.lookahead_dist = lookahead_dist
        
        # 内部状态
        self.prev_idx = 0
        self.last_u = np.array([0.0, 0.0]) # [v, w]
        
        # --- 2. 优化参数 ---
        self.safe_distance = safe_distance
        self.v_max = v_max
        self.omega_max = omega_max
        self.r_min = 0.3 # 最小转弯半径，用于非完整约束惩罚
        self.a_max = a_max
        
        self.w_p = w_p
        self.w_t = w_t
        self.w_kin = w_kin
        self.w_r = w_r
        self.w_obs = w_obs
        self.w_goal = w_goal
        self.w_acc = w_acc
        self.w_theta = w_theta
        
        self.T_min = T_min
        self.T_max = T_max
        self.epsilon = 1e-4 # 避免除零

        # 动态变量
        self.n = 0
        self.trajectory = None

    def calc_control_input(self, observed_x: np.ndarray, obstacles: list = None):
        """MPC 风格接口"""
        if observed_x.ndim > 1: observed_x = observed_x.flatten()
        x0 = observed_x 
        
        # 1. 获取局部目标
        xf = self._get_local_goal_from_path(x0[0], x0[1])
        
        # 2. 自动计算 n (强制最小值为 2)
        self.n = self._auto_calculate_n(x0, xf)
        
        # 3. 障碍物预处理
        obs_np = np.array([])
        if obstacles is not None and len(obstacles) > 0:
            obs_temp = np.array(obstacles)
            # 如果输入是 [x,y,a,b...], ndim=1, 需要转为 [[x,y,a,b...]]
            if obs_temp.ndim == 1:
                obs_np = obs_temp.reshape(1, -1)
            else:
                obs_np = obs_temp
        
        try:
            # 4. 构建求解 (注意：实时构建很慢，建议后续改为参数化 Solver)
            res_dict = self._build_and_solve(x0, xf, obs_np)
            
            # 5. 提取结果
            traj, dt_seq = self._extract_trajectory(res_dict)
            self.trajectory = traj
            
            # 6. 计算控制量 (简单的差分控制)
            if len(traj) > 1 and len(dt_seq) > 0:
                dx = traj[1,0] - traj[0,0]
                dy = traj[1,1] - traj[0,1]
                dth = traj[1,2] - traj[0,2]
                # 角度归一化
                dth = (dth + np.pi) % (2*np.pi) - np.pi
                
                dt0 = float(dt_seq[0])
                if dt0 < 1e-3: dt0 = 0.1
                
                v_cmd = np.hypot(dx, dy) / dt0
                w_cmd = dth / dt0
                
                # 方向判断 (如果目标在身后，则倒车)
                cos_th = np.cos(traj[0,2])
                sin_th = np.sin(traj[0,2])
                # 点乘判断方向
                if dx * cos_th + dy * sin_th < -1e-2: 
                    v_cmd = -v_cmd
                
                # 限幅
                v_cmd = np.clip(v_cmd, -self.v_max, self.v_max)
                w_cmd = np.clip(w_cmd, -self.omega_max, self.omega_max)
                
                self.last_u = np.array([v_cmd, w_cmd])
            
            return traj, None, self.last_u
            
        except Exception as e:
            print(f"[TEB Error] Solver failed: {e}")
            # 失败时尝试简单的减速保持
            return np.array([x0]), None, self.last_u * 0.0

    def _auto_calculate_n(self, x0, xf):
        """计算点数，增加安全下限"""
        dist_total = np.linalg.norm(xf[:2] - x0[:2])
        
        # 若起点终点重合，返回最小点数
        if dist_total < 1e-2:
            return 3
        
        # 2. 估算每个时间步的最大移动距离
        T_avg = (self.T_min + self.T_max) / 2  # 平均时间步
        max_step_dist = self.v_max * T_avg     # 每步最大移动距离
        
        # 3. 基础点数：总距离 / 每步最大距离
        # TEB 的 n 是中间间隔的数量，总点数是 n+2
        base_n = int(dist_total / (max_step_dist * 0.3)) # 0.3 为安全系数，稍微多一点点
        base_n = max(3, base_n)  # 至少3个中间间隔
        base_n = min(50, base_n) # 防止过大导致求解超时
        
        return base_n

    def _build_and_solve(self, x0, xf, obs_now):
        # --- 变量定义 ---
        # x, y, theta 的点数是 n+2 (包含起点和终点)
        # dt 的段数是 n+1
        x = ca.SX.sym('x', self.n + 2)
        y = ca.SX.sym('y', self.n + 2)
        theta = ca.SX.sym('theta', self.n + 2)
        dt = ca.SX.sym('dt', self.n + 1)
        z = ca.vertcat(x, y, theta, dt)
    
        # 目标函数
        f = 0
        
        # 1. 路径平滑性与时间最优
        for i in range(self.n + 1):
            dx = x[i+1] - x[i]
            dy = y[i+1] - y[i]
            f += self.w_p * (dx**2 + dy**2)  
            f += self.w_t * dt[i]**2          

        g_eq = []
        g_ineq = []
        
        # 2. 起点终点约束 (等式约束)
        g_eq.extend([
            x[0] - x0[0],    y[0] - x0[1],    theta[0] - x0[2],
            x[-1] - xf[0],   y[-1] - xf[1],   theta[-1] - xf[2]
        ])
        
        # 3. 避障代价 (软约束添加到目标函数 f)
        if len(obs_now) > 0:
            for i in range(1, self.n + 2): # 对所有轨迹点(除起点)
                for obs in obs_now:
                    # 假设 obs 格式: [x, y, a, b, angle, cos_angle, sin_angle]
                    # 如果只有 [x, y, r]，需要适配
                    if obs.size < 2: continue
                    
                    obs_x, obs_y = obs[0], obs[1]
                    # 默认圆形障碍物处理 (如果没有椭圆参数)
                    obs_a = obs[2] if obs.size > 2 else 0.5
                    obs_b = obs[3] if obs.size > 3 else 0.5
                    cos_ot = obs[5] if obs.size > 6 else 1.0
                    sin_ot = obs[6] if obs.size > 6 else 0.0

                    dx_o = x[i] - obs_x
                    dy_o = y[i] - obs_y
                    x_rel = dx_o * cos_ot + dy_o * sin_ot
                    y_rel = -dx_o * sin_ot + dy_o * cos_ot
                    
                    a_safe = obs_a + self.safe_distance
                    b_safe = obs_b + self.safe_distance
                    
                    ellipse_val = (x_rel / a_safe)**2 + (y_rel / b_safe)**2
                    
                    # 使用 exp 函数作为惩罚项
                    f += self.w_obs * ca.exp(5.0 * (1.0 - ellipse_val)) # 稍微减小系数防梯度爆炸

        # 4. 运动学约束
        for i in range(self.n + 1):
            dx = x[i+1] - x[i]
            dy = y[i+1] - y[i]
        
            # 速度约束
            dist = ca.sqrt(dx**2 + dy**2 + self.epsilon)
            v = dist / (dt[i] + self.epsilon)
            g_ineq.extend([v - self.v_max, -v - self.v_max])

            # 角速度约束
            dth_raw = theta[i+1] - theta[i]
            # 处理角度跳变对于约束的影响较难，这里假设 dt 足够小不发生跳变，或者依赖非完整约束
            omega = dth_raw / (dt[i] + self.epsilon)
            g_ineq.extend([omega - self.omega_max, -omega - self.omega_max])    
            
            # 最小转弯半径惩罚 (软约束)
            # radius ~ v/w. 我们惩罚 w > v/r_min -> w*r_min - v > 0
            # 或者直接惩罚非完整约束误差
            
            # 加速度约束（除最后一个时间步）
            if i < self.n:
                dx2 = x[i+2] - x[i+1]
                dy2 = y[i+2] - y[i+1]
                dist_step2 = ca.sqrt(dx2**2 + dy2**2 + self.epsilon)
                v2 = dist_step2 / (dt[i+1] + self.epsilon)
                acc = (v2 - v) / (0.5*(dt[i] + dt[i+1]) + self.epsilon)
                g_ineq.extend([acc - self.a_max, -acc - self.a_max])
                f += self.w_acc * acc**2 + self.w_theta * (omega**2)

            # 非完整约束 (Non-holonomic constraint)
            # sin(th)*dx - cos(th)*dy = 0
            th_avg = theta[i] # 或者使用 (theta[i]+theta[i+1])/2
            cross = ca.sin(th_avg) * dx - ca.cos(th_avg) * dy
            f += self.w_kin * cross**2

        # 5. 约束边界设置
        g = ca.vertcat(*g_eq, *g_ineq)
        lbg = [0]*len(g_eq) + [-ca.inf]*len(g_ineq)
        ubg = [0]*len(g_eq) + [0]*len(g_ineq)
        
        # 6. 变量上下界
        lbx = -np.inf * np.ones(z.shape[0])
        ubx = np.inf * np.ones(z.shape[0])
        
        # 固定起点和终点的位置与姿态
        # x, y, theta 的布局: [x0...xn+1, y0...yn+1, th0...thn+1, dt0...dtn]
        idx_end_x = self.n + 2
        idx_end_y = 2 * (self.n + 2)
        idx_end_th = 3 * (self.n + 2)
        
        fix_indices = [
            0, idx_end_x - 1,                # x_start, x_end
            idx_end_x, idx_end_y - 1,        # y_start, y_end
            idx_end_y, idx_end_th - 1        # th_start, th_end
        ]
        
        fix_values = [
            x0[0], xf[0],
            x0[1], xf[1],
            x0[2], xf[2]
        ]
        
        lbx[fix_indices] = fix_values
        ubx[fix_indices] = fix_values
        
        # 时间步上下界
        lbx[idx_end_th:] = self.T_min
        ubx[idx_end_th:] = self.T_max
        
        # 7. 初始化猜想 (修复了调用参数和返回值)
        init_x, init_y, init_theta, init_dt = self._get_spline_initial_guess(x0, xf)

        z0 = np.zeros(z.shape[0])
        z0[0 : idx_end_x] = init_x
        z0[idx_end_x : idx_end_y] = init_y
        z0[idx_end_y : idx_end_th] = init_theta
        z0[idx_end_th : ] = init_dt
        
        # 8. 求解 NLP
        nlp = {'x': z, 'f': f, 'g': g}
        # print_level=0 关闭输出以提高速度
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'} 
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        res = solver(x0=z0, lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)
        return res

    def _get_spline_initial_guess(self, x0, xf):
        """
        修复版：增加了 x0, xf 参数，并且正确返回分开的数组
        """
        n = self.n
        n_points = n + 2
        t = np.linspace(0, 1, n_points)
        
        dist = np.linalg.norm(xf[:2] - x0[:2])
        scale = max(dist * 1.0, 0.5) 
        
        p0, pf = x0[:2], xf[:2]
        v0 = np.array([np.cos(x0[2]), np.sin(x0[2])]) * scale
        vf = np.array([np.cos(xf[2]), np.sin(xf[2])]) * scale
        
        # Hermite spline
        h00 = 2*t**3 - 3*t**2 + 1
        h10 = t**3 - 2*t**2 + t
        h01 = -2*t**3 + 3*t**2
        h11 = t**3 - t**2
        
        path_xy = (np.outer(h00, p0) + np.outer(h10, v0) + 
                   np.outer(h01, pf) + np.outer(h11, vf))
        init_x = path_xy[:, 0]
        init_y = path_xy[:, 1]
        
        # Angle interpolation
        dxs = np.diff(init_x)
        dys = np.diff(init_y)
        # 补齐最后一个点的角度
        dxs = np.append(dxs, dxs[-1])
        dys = np.append(dys, dys[-1])
        
        init_theta = np.arctan2(dys, dxs)
        # 强制修正首尾角度
        init_theta[0] = x0[2]
        init_theta[-1] = xf[2]
        init_theta = np.unwrap(init_theta)
        
        init_dt = np.ones(n + 1) * ((self.T_min + self.T_max) / 2)
        
        # 返回分开的数组，而不是 concatenate 后的
        return init_x, init_y, init_theta, init_dt

    def _get_local_goal_from_path(self, x, y):
        search_range = 15 
        start_search = max(0, self.prev_idx - 5)
        end_search = min(len(self.ref_path), self.prev_idx + search_range)
        
        path_chunk = self.ref_path[start_search:end_search]
        if len(path_chunk) == 0: return self.ref_path[-1]
        
        dists = np.hypot(path_chunk[:,0] - x, path_chunk[:,1] - y)
        min_local = np.argmin(dists)
        curr_idx = start_search + min_local
        self.prev_idx = curr_idx
        
        accum_dist = 0.0
        target_idx = curr_idx
        while target_idx < len(self.ref_path) - 1:
            d = np.hypot(self.ref_path[target_idx+1,0]-self.ref_path[target_idx,0], 
                         self.ref_path[target_idx+1,1]-self.ref_path[target_idx,1])
            accum_dist += d
            if accum_dist > self.lookahead_dist:
                break
            target_idx += 1
        return self.ref_path[target_idx]

    def _extract_trajectory(self, res):
        z_val = res['x'].full().flatten()
        n = self.n
        idx_end_x = n + 2
        idx_end_y = 2 * (n + 2)
        idx_end_th = 3 * (n + 2)
        
        x = z_val[0 : idx_end_x]
        y = z_val[idx_end_x : idx_end_y]
        th = z_val[idx_end_y : idx_end_th]
        dt = z_val[idx_end_th :]
        return np.column_stack([x, y, th]), dt