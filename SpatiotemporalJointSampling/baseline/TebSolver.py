import casadi as ca
import numpy as np

class TebplanSolver:
    """
    自适应 TEB 规划器 (融合版 - 修复维度错误)
    """
    def __init__(self, 
                 ref_path: np.ndarray,
                 safe_distance: float = 1.0,  # 稍微调小默认安全距离
                 v_max=3.0, omega_max=2.0, a_max=1.0, 
                 lookahead_dist=5.0,
                 # 权重参数
                 w_p=0.5, w_t=2.0, w_kin=50.0, w_r=5.0, w_obs=20.0,w_goal=1.0, 
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
        self.r_min = 0.3
        self.a_max = a_max
        
        self.w_p = w_p
        self.w_t = w_t
        self.w_kin = w_kin
        self.w_r = w_r
        self.w_obs = w_obs
        self.w_goal = w_goal
        
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
        
        # 2. 自动计算 n (强制最小值为 3, 建议 5 以避免边缘效应)
        self.n = self._auto_calculate_n(x0, xf)
        
        # 3. 障碍物预处理 (关键修复：确保是 2D 数组)
        obs_np = np.array([])
        if obstacles is not None and len(obstacles) > 0:
            obs_temp = np.array(obstacles)
            # 如果输入是 [x,y,a,b...], ndim=1, 需要转为 [[x,y,a,b...]]
            if obs_temp.ndim == 1:
                obs_np = obs_temp.reshape(1, -1)
            else:
                obs_np = obs_temp
        
        try:
            # 4. 构建求解
            res_dict = self._build_and_solve(x0, xf, obs_np)
            
            # 5. 提取结果
            traj, dt_seq = self._extract_trajectory(res_dict)
            self.trajectory = traj
            
            # 6. 计算控制量
            if len(traj) > 1 and len(dt_seq) > 0:
                dx = traj[1,0] - traj[0,0]
                dy = traj[1,1] - traj[0,1]
                dth = traj[1,2] - traj[0,2]
                dth = (dth + np.pi) % (2*np.pi) - np.pi
                
                dt0 = float(dt_seq[0])
                if dt0 < 1e-3: dt0 = 0.1
                
                v_cmd = np.hypot(dx, dy) / dt0
                w_cmd = dth / dt0
                
                # 方向判断
                cos_th = np.cos(traj[0,2])
                sin_th = np.sin(traj[0,2])
                if dx * cos_th + dy * sin_th < 0:
                    v_cmd = -v_cmd
                
                # 限幅
                v_cmd = np.clip(v_cmd, -self.v_max, self.v_max)
                w_cmd = np.clip(w_cmd, -self.omega_max, self.omega_max)
                
                self.last_u = np.array([v_cmd, w_cmd])
            
            return traj, None, self.last_u
            
        except Exception as e:
            # print(f"[TEB Error] Solver failed: {e}")
            # 失败时尝试简单的减速保持
            return np.array([x0]), None, self.last_u * 0.5

    def _auto_calculate_n(self, x0, xf):
        """计算点数，增加安全下限"""
        dist_total = np.linalg.norm(xf[:2] - x0[:2])
        
        # 估算每步距离
        T_avg = (self.T_min + self.T_max) / 2
        max_step_dist = self.v_max * T_avg * 0.3 # 0.5 是保守系数
        
        if max_step_dist < 0.01: max_step_dist = 0.1

        base_n = int(dist_total / max_step_dist)
        # 关键修复：强制 n 至少为 3，且避免 n=1 导致的维度计算边缘 case
        # n=1 时 x 向量长度 3，dt 向量长度 2，容易在切片时出错
        base_n = max(3, min(base_n, 40)) 
        return base_n

    def _build_and_solve(self, x0, xf, obs_now):
        # --- 变量定义 ---
        # 确保 self.n 在此处是整数
        n = int(self.n)
        
        x = ca.SX.sym('x', n + 2)
        y = ca.SX.sym('y', n + 2)
        theta = ca.SX.sym('theta', n + 2)
        dt = ca.SX.sym('dt', n + 1)
        
        z = ca.vertcat(x, y, theta, dt)
        
        # --- 目标函数 ---
        f = 0
        g_eq = []
        g_ineq = []
        
        # 起点终点约束
        g_eq.extend([
            x[0] - x0[0], y[0] - x0[1], theta[0] - x0[2]
        ])
        f+= self.w_goal * ((x[-1] - xf[0])**2 + (y[-1] - xf[1])**2)
        
        for i in range(n + 1):
            dx = x[i+1] - x[i]
            dy = y[i+1] - y[i]
            
            # 1. 路径与时间代价
            f += self.w_p * (dx**2 + dy**2)
            f += self.w_t * dt[i]**2
            
            # 2. 非完整约束 (Cross product)
            li_x, li_y = ca.cos(theta[i]), ca.sin(theta[i])
            cross = li_x * dy - li_y * dx
            f += self.w_kin * cross**2
            
            # 3. 动力学约束
            dist = ca.sqrt(dx**2 + dy**2 + self.epsilon)
            v = dist / (dt[i] + self.epsilon)
            
            dth_raw = theta[i+1] - theta[i]
            omega = dth_raw / (dt[i] + self.epsilon)
            
            # 速度约束
            g_ineq.extend([v - self.v_max])
            
            # 角速度约束
            g_ineq.extend([omega - self.omega_max, -omega - self.omega_max])
            
            # 最小半径惩罚
            radius_approx = v / (ca.fabs(omega) + self.epsilon)
            f += self.w_r * ca.fmax(0, self.r_min - radius_approx)**2
            
            # 加速度 (注意这里只到 n-1)
            if i < n:
                dt_avg = 0.5 * (dt[i] + dt[i+1])
                dx2 = x[i+2] - x[i+1]
                dy2 = y[i+2] - y[i+1]
                v2 = ca.sqrt(dx2**2 + dy2**2 + self.epsilon) / (dt[i+1] + self.epsilon)
                acc = (v2 - v) / (dt_avg + self.epsilon)
                g_ineq.extend([acc - self.a_max, -acc - self.a_max])

        # 4. 避障代价
        if len(obs_now) > 0:
            for i in range(1, n + 2): # 对所有轨迹点(除起点)
                for obs in obs_now:
                    # 维度保护：必须有7个参数
                    if obs.size < 7: continue
                    obs_x, obs_y, obs_a, obs_b = obs[0], obs[1], obs[2], obs[3]
                    cos_ot, sin_ot = obs[5], obs[6]
                    
                    dx_o = x[i] - obs_x
                    dy_o = y[i] - obs_y
                    x_rel = dx_o * cos_ot + dy_o * sin_ot
                    y_rel = -dx_o * sin_ot + dy_o * cos_ot
                    
                    a_safe = obs_a + self.safe_distance
                    b_safe = obs_b + self.safe_distance
                    
                    val = (x_rel / a_safe)**2 + (y_rel / b_safe)**2
                    f += self.w_obs * ca.exp(4.0 * (1.0 - val))

        # --- Solver Setup ---
        # 自动计算维度
        n_z = z.shape[0]
        
        lbx = -np.inf * np.ones(n_z)
        ubx = np.inf * np.ones(n_z)
        
        # DT 边界设置 (关键：使用切片时确保索引不过界)
        idx_dt_start = 3 * (n + 2)
        # 这里的切片 idx_dt_start: 到底部，对于 n=1, idx=9, len=11, [9:] 取 9,10。正确。
        lbx[idx_dt_start:] = self.T_min
        ubx[idx_dt_start:] = self.T_max
        
        g = ca.vertcat(*g_eq, *g_ineq)
        lbg = [0.0] * len(g_eq) + [-np.inf] * len(g_ineq)
        ubg = [0.0] * len(g_eq) + [0.0] * len(g_ineq)
        
        nlp = {'x': z, 'f': f, 'g': g}
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        
        # 初值猜测
        z0 = self._get_spline_initial_guess(x0, xf)
        
        # 安全检查：如果初值维度不匹配，强行 Resize
        if z0.shape[0] != n_z:
            # print(f"Warning: z0 shape {z0.shape} mismatch with vars {n_z}, recreating.")
            z0 = np.zeros(n_z) # Fallback to zeros if mismatch (rare)

        res = solver(x0=z0, lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)
        return res

    def _get_spline_initial_guess(self, x0, xf):
        n = self.n
        n_points = n + 2
        t = np.linspace(0, 1, n_points)
        
        dist = np.linalg.norm(xf[:2] - x0[:2])
        scale = max(dist * 1.0, 0.1) # 避免 scale 为 0
        
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
        # Pad last diff to keep size consistent
        dxs = np.append(dxs, dxs[-1])
        dys = np.append(dys, dys[-1])
        
        init_theta = np.arctan2(dys, dxs)
        init_theta[0] = x0[2]
        init_theta[-1] = xf[2]
        init_theta = np.unwrap(init_theta)
        
        init_dt = np.ones(n + 1) * ((self.T_min + self.T_max) / 2)
        
        return np.concatenate([init_x, init_y, init_theta, init_dt])

    def _get_local_goal_from_path(self, x, y):
        search_range = 20 
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