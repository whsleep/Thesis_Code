import casadi as ca
import numpy as np

class TebplanSolver:
    """
    简化的TEB局部路径规划器
    仅保留最基本功能：固定步长，椭圆障碍物避障
    """
    def __init__(self, 
                 ref_path: np.ndarray,
                 delta_t: float = 0.1,
                 n_teb_points: int = 20,
                 safe_distance: float = 1.5,
                 v_max: float = 10.0,
                 omega_max: float = 3.0,
                 a_max: float = 2.0,
                 theta_max: float = 1.0,
                 max_obstacles: int = 10,
                 w_path: float = 0.5,
                 w_velocity: float = 2.0,
                 w_radius: float = 2.0,
                 w_obs: float = 15.0,
                 w_kin: float = 2.0,
                 w_via_point: float = 5.0,
                 w_goal: float = 3.0,
                 lookahead_dist: float = 5.0,
                 **kwargs):
        
        # 参考路径
        self.ref_path = ref_path
        # 固定仿真时间
        self.delta_t = delta_t
        # 固定长度
        self.n_teb_points = n_teb_points
        # 安全距离
        self.safe_distance = safe_distance

        # 运动学约束
        self.v_max = v_max
        self.omega_max = omega_max
        self.a_max = a_max
        self.theta_max = theta_max
        self.max_obstacles = max_obstacles
        self.r_min = 0.1  # 最小转弯半径

        # 代价函数权重
        self.w_path = w_path
        self.w_velocity = w_velocity
        self.w_radius = w_radius
        self.w_obs = w_obs
        self.w_kin = w_kin
        self.w_via_point = w_via_point
        self.w_goal = w_goal
        self.num_via_points = 6
        self.lookahead_dist = lookahead_dist
        
        # 内部状态
        self.prev_idx = 0
        self.end_idx = self.prev_idx
        self.last_v = 0.0
        self.last_w = 0.0
        self.solver = None
        self.last_opt_z = None

        # 障碍列表
        self.obstacles = []

    def calc_control_input(self, observed_x: np.ndarray, obstacles: list = None):
        """
        计算控制量
        """
        if observed_x.ndim > 1:
            observed_x = observed_x.flatten()
        
        # 提取状态
        x, y, theta = observed_x[0], observed_x[1], observed_x[2]
        
        # 获取局部目标点
        local_goal = self._get_local_goal_from_path(x, y)
        
        # 求解轨迹
        trajectory = self._solve_teb(observed_x, local_goal, obstacles)
        
        # 提取控制量
        if trajectory is not None and len(trajectory) > 1:
            x0, y0, th0 = trajectory[0]
            x1, y1, th1 = trajectory[1]
            
            # 计算线速度
            dx = x1 - x0
            dy = y1 - y0
            v = (dx * np.cos(th0) + dy * np.sin(th0)) / self.delta_t
            v = np.clip(v, -self.v_max, self.v_max)
            
            # 计算角速度
            dth = np.arctan2(np.sin(th1 - th0), np.cos(th1 - th0))
            w = dth / self.delta_t
            w = np.clip(w, -self.omega_max, self.omega_max)
            
            best_u = np.array([v, w])
            self.last_v = v
            self.last_w = w
        else:
            best_u = np.array([self.last_v, self.last_w])
            trajectory = np.array([[x, y, theta]])
        
        # 返回格式兼容
        sampled_trajs = np.zeros((1, 1, 3))
        
        return trajectory, sampled_trajs, best_u

    def _get_local_goal_from_path(self, x, y):
        """寻找前瞻点"""
        # 限定搜索范围
        search_range = 20
        start_idx = max(0, self.prev_idx - 10)
        end_idx = min(len(self.ref_path), start_idx + search_range)
        # 提取局部路径
        path_segment = self.ref_path[start_idx:end_idx]
        if len(path_segment) == 0:
            return self.ref_path[-1]
        
        # 找到最近点
        dx = x - path_segment[:, 0]
        dy = y - path_segment[:, 1]
        dist_sq = dx**2 + dy**2
        min_idx_local = np.argmin(dist_sq)
        min_idx_global = start_idx + min_idx_local
        self.prev_idx = min_idx_global
        
        # 寻找前瞻点
        self.end_idx = min_idx_global
        current_dist = 0.0
        while self.end_idx < len(self.ref_path) - 1:
            p1 = self.ref_path[self.end_idx]
            p2 = self.ref_path[self.end_idx + 1]
            d = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
            current_dist += d
            if current_dist >= self.lookahead_dist:
                break
            self.end_idx += 1
            
        return self.ref_path[self.end_idx]

    def _solve_teb(self, start_state, goal_state, obstacles):
        """求解TEB轨迹"""
        # 固定起点终点
        x0 = start_state
        xf = goal_state
        
        # 构建优化问题
        if self.solver is None:
            self.solver = self._build_solver()
        
        # 生成初始猜测
        z0 = self._get_initial_guess(x0, xf)

        # 提取路标点
        via_indices = np.linspace(0, self.n_teb_points + 1, self.num_via_points + 2, dtype=int)[1:-1]
        via_points_flat = []
        n_vars = self.n_teb_points + 2
        for idx in via_indices:
            # z0 结构: [x_0...x_N, y_0...y_N, th_0...th_N]
            v_x = z0[idx]
            v_y = z0[n_vars + idx] # y 在 x 后面
            via_points_flat.extend([v_x, v_y])

        # 提取障碍物参数
        # 每个障碍物有 7 个参数: cx, cy, a, b, dummy, cos_ot, sin_ot
        obs_params = np.zeros(self.max_obstacles * 7)
        
        if obstacles is not None and len(obstacles) > 0:
            obs_list = np.array(obstacles) # [M, 7]
            
            # 计算距离并排序，取最近的 N 个
            dists = np.hypot(obs_list[:,0] - x0[0], obs_list[:,1] - x0[1])
            sorted_indices = np.argsort(dists)
            
            count = min(len(obstacles), self.max_obstacles)
            for i in range(count):
                idx = sorted_indices[i]
                # 填充到 flat 数组中
                obs_params[i*7 : (i+1)*7] = obs_list[idx, :]
            
            # 对于剩余的空槽位，将障碍物移到无穷远，防止干扰
            for i in range(count, self.max_obstacles):
                obs_params[i*7] = 10000.0     # cx
                obs_params[i*7+1] = 10000.0   # cy
                obs_params[i*7+2] = 0.1       # a
                obs_params[i*7+3] = 0.1       # b

        p_val = np.concatenate([x0, xf, via_points_flat, obs_params])
        try:
            # 求解
            res = self.solver(
                x0=z0, 
                p=p_val, 
                lbg=self.lbg, 
                ubg=self.ubg
                )

            # 保存最后的最优解用于热启动
            self.last_opt_z = res['x'].full().flatten()
            
            # 提取轨迹
            trajectory = self._extract_trajectory(res, x0, xf)
            return trajectory
        except:
            # 失败时返回直线
            n_points = self.n_teb_points + 2
            t = np.linspace(0, 1, n_points)
            x_traj = x0[0] + t * (xf[0] - x0[0])
            y_traj = x0[1] + t * (xf[1] - x0[1])
            theta_traj = x0[2] + t * (xf[2] - x0[2])
            return np.column_stack([x_traj, y_traj, theta_traj])

    def _build_solver(self):
        """构建求解器"""
        # 求解的步长
        n = self.n_teb_points
        epsilon = 1e-5
        
        # 目标函数
        f = 0
        # 约束条件
        g_eq = []    # 等式约束
        g_ineq = []  # 不等式约束
        
        # --- 定义边界列表 ---
        lbg_eq = []
        ubg_eq = []
        lbg_ineq = []
        ubg_ineq = []

        # 变量定义
        x = ca.SX.sym('x', n + 2)
        y = ca.SX.sym('y', n + 2)
        theta = ca.SX.sym('theta', n + 2)
        z = ca.vertcat(x, y, theta)
        
        # 输入参数
        n_p_base = 6 # Start(3) + Goal(3)
        n_p_via = 2 * self.num_via_points
        n_p_obs = 7 * self.max_obstacles # 7个参数对应 MppiSolver 的结构
        n_p = n_p_base + n_p_via + n_p_obs
        p = ca.SX.sym('p', n_p)
        x_start, y_start, theta_start = p[0], p[1], p[2]
        x_goal, y_goal, theta_goal = p[3], p[4], p[5]

        via_indices = np.linspace(0, n + 1, self.num_via_points + 2, dtype=int)[1:-1]
        # p 的结构: [Start(3), Goal(3), V1x, V1y, V2x, V2y, ...]
        n_p_base = 6 
        base_obs_idx = n_p_base + n_p_via
        for i, idx in enumerate(via_indices):
            # 获取当前 via point 的参数坐标
            p_vx = p[n_p_base + 2*i]
            p_vy = p[n_p_base + 2*i + 1]
            
            # 对应的轨迹变量坐标
            traj_x = x[idx]
            traj_y = y[idx]
            
            # 添加软约束
            dist_sq = (traj_x - p_vx)**2 + (traj_y - p_vy)**2
            f += self.w_via_point * dist_sq       


        # 1. 起点,终点松弛约束
        g_eq.extend([
            x[0] - x_start,    y[0] - y_start,    theta[0] - theta_start
        ])
        # 对应的边界：6个0
        lbg_eq.extend([0.0] * 3)
        ubg_eq.extend([0.0] * 3)

        f += self.w_goal * ((x[-1] - x_goal)**2 + (y[-1] - y_goal)**2)

        # 2. 路径代价和运动学约束
        for i in range(n + 1):
            dx = x[i+1] - x[i]
            dy = y[i+1] - y[i]
            dist_sq = dx**2 + dy**2 + epsilon
            dist = ca.sqrt(dist_sq)
            f += self.w_path * dist
            
            # 线速度约束
            v = dist / self.delta_t
            g_ineq.extend([v - self.v_max])
            lbg_ineq.extend([-np.inf])
            ubg_ineq.extend([0.0])

            # 角速度约束
            dth = ca.atan2(ca.sin(theta[i+1]-theta[i]),
                           ca.cos(theta[i+1]-theta[i]))
            omega = dth / self.delta_t
            # omega <= max  =>  omega - max <= 0
            # omega >= -max => -omega - max <= 0
            g_ineq.extend([omega - self.omega_max, -omega - self.omega_max])
            lbg_ineq.extend([-np.inf, -np.inf])
            ubg_ineq.extend([0.0, 0.0])

            # 转弯半径约束 
            radius = v / (ca.fabs(omega) + epsilon)
            f += self.w_radius * ca.fmax(0, self.r_min - radius)**2

            # 加速度约束
            if i < n:
                dx2 = x[i+2] - x[i+1]
                dy2 = y[i+2] - y[i+1]
                dist2 = ca.sqrt(dx2**2 + dy2**2 + epsilon) 
                v2 = dist2 / self.delta_t
                acc = (v2 - v) / self.delta_t # 简单的差分计算
                g_ineq.extend([acc - self.a_max, -acc - self.a_max])
                lbg_ineq.extend([-np.inf, -np.inf])
                ubg_ineq.extend([0.0, 0.0])

            # 3. 非完整约束
            dx = x[i+1] - x[i]
            dy = y[i+1] - y[i]
            li = ca.vertcat(ca.cos(theta[i]), ca.sin(theta[i]))
            li1 = ca.vertcat(ca.cos(theta[i+1]), ca.sin(theta[i+1]))
            cross = (li[0] + li1[0]) * dy - (li[1] + li1[1]) * dx
            f += self.w_kin * cross**2

            # 4.对每个时间步 i，遍历所有障碍物 j
            for j in range(self.max_obstacles):
                # 提取参数: [cx, cy, a, b, dummy, cos_ot, sin_ot]
                o_idx = base_obs_idx + j * 7
                cx = p[o_idx]
                cy = p[o_idx + 1]
                a  = p[o_idx + 2]
                b  = p[o_idx + 3]
                # param 4 is dummy/angle, ignored here as we use cos/sin directly
                cos_ot = p[o_idx + 5]
                sin_ot = p[o_idx + 6]
                
                safe_a = a + self.safe_distance
                safe_b = b + self.safe_distance
                
                # 计算相对位置
                dx_obs = x[i] - cx
                dy_obs = y[i] - cy
                
                # 旋转到障碍物坐标系 (Logic matches MppiSolver)
                # lx = dx * cos + dy * sin
                # ly = -dx * sin + dy * cos
                lx = dx_obs * cos_ot + dy_obs * sin_ot
                ly = -dx_obs * sin_ot + dy_obs * cos_ot
                
                # 椭圆方程: (x/a)^2 + (y/b)^2
                # dist_val < 1.0 意味着在障碍物内 (碰撞)
                # dist_val > 1.0 意味着安全
                ellipse_dist = (lx / safe_a)**2 + (ly / safe_b)**2
            
                
                # B. 软代价: 离得越近代价越高 (barrier function)
                f += self.w_obs * ca.exp(5.0 * (1.0 - ellipse_dist))

        # 记录约束数量
        self.n_eq = len(g_eq)
        self.n_ineq = len(g_ineq)

        # 保存边界到类成员变量 (将 list 转为 numpy array)
        self.lbg = np.concatenate([lbg_eq, lbg_ineq])
        self.ubg = np.concatenate([ubg_eq, ubg_ineq])

        # 构建NLP
        g_all = ca.vertcat(*g_eq, *g_ineq)
        
        nlp = {'x': z, 'f': f, 'g': g_all, 'p': p}
        
        opts = {
            'ipopt.print_level': 0, 
            'print_time': 0, 
            'ipopt.max_iter': 500, 
            'ipopt.tol': 1e-3,
            'ipopt.warm_start_init_point': 'yes'
        }

        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)    
        
        return solver

    def _get_initial_guess(self, x0, xf):
        """
        生成初始猜测：优先级 热启动 > 参考路径 > 赫米特插值
        """
        n = self.n_teb_points
        
        # --- 策略1：热启动 (Warm Start) ---
        # 如果上一帧求解成功，利用上一帧的轨迹
        if self.last_opt_z is not None:
            # 提取上一帧的 x, y, theta
            last_x = self.last_opt_z[:n+2]
            last_y = self.last_opt_z[n+2:2*(n+2)]
            last_th = self.last_opt_z[2*(n+2):]
            
            # 创建新的猜测数组
            init_x = np.zeros(n + 2)
            init_y = np.zeros(n + 2)
            init_theta = np.zeros(n + 2)
            
            # 平移操作：discard 第一个点 (已经是过去式)，整体前移
            # new[0]...new[n-1] = old[1]...old[n]
            init_x[:-1] = last_x[1:]
            init_y[:-1] = last_y[1:]
            init_theta[:-1] = last_th[1:]
            
            # 外推最后一个点：假设保持末端速度方向延伸
            # 简单做法：复制倒数第二个点，或者线性外推
            init_x[-1] = 2*init_x[-2] - init_x[-3]
            init_y[-1] = 2*init_y[-2] - init_y[-3]
            init_theta[-1] = init_theta[-2] # 角度保持不变
            
            # 强制修正起点和终点为当前实际值 (这一步很重要，因为求解器对初值敏感)
            init_x[0], init_y[0], init_theta[0] = x0
            init_x[-1], init_y[-1], init_theta[-1] = xf
            
            # 重新组合
            return np.concatenate([init_x, init_y, init_theta])

        # --- 策略2：基于参考路径的采样 (原有逻辑的优化版) ---
        # 线性插值位置
        if self.end_idx <= self.prev_idx:
            self.end_idx = min(self.prev_idx + 1, len(self.ref_path) - 1)
        
        segment_length = self.end_idx - self.prev_idx + 1
        
        # 如果片段太短，直接使用备用策略
        if segment_length <= 2:
            return self._get_fallback_initial_guess(x0, xf)

        # 提取片段
        ref_segment = self.ref_path[self.prev_idx:self.end_idx+1]
        
        # 使用 numpy 插值确保点数正好是 n+2
        t_original = np.linspace(0, 1, len(ref_segment))
        t_new = np.linspace(0, 1, n + 2)
        
        # 位置插值
        init_x = np.interp(t_new, t_original, ref_segment[:, 0])
        init_y = np.interp(t_new, t_original, ref_segment[:, 1])
        
        # 角度插值 (处理环绕问题)
        if ref_segment.shape[1] >= 3:
            ref_theta = ref_segment[:, 2]
            # Unwrapping 是关键，防止角度从 3.14 跳变到 -3.14 导致的插值错误
            ref_theta_unwrapped = np.unwrap(ref_theta)
            init_theta = np.interp(t_new, t_original, ref_theta_unwrapped)
        else:
            # 如果参考路径没有角度，使用反正切计算
            init_theta = np.zeros(n+2)
            dxs = np.diff(init_x)
            dys = np.diff(init_y)
            yaws = np.arctan2(dys, dxs)
            init_theta[:-1] = yaws
            init_theta[-1] = yaws[-1]
            init_theta = np.unwrap(init_theta)

        # 强制修正起点终点
        init_x[0], init_y[0], init_theta[0] = x0
        init_x[-1], init_y[-1], init_theta[-1] = xf
        
        return np.concatenate([init_x, init_y, init_theta])

    def _get_fallback_initial_guess(self, x0, xf):
        """
        备用猜测：使用三次赫米特插值 (Cubic Hermite Spline)
        相比线性插值，它能生成符合起止角度的S形曲线，更符合车辆运动学
        """
        n = self.n_teb_points
        num_points = n + 2
        t = np.linspace(0, 1, num_points)
        
        # 提取起止点信息
        p0 = np.array([x0[0], x0[1]])
        p1 = np.array([xf[0], xf[1]])
        th0 = x0[2]
        th1 = xf[2]
        
        # 计算欧几里得距离，作为切线向量的模长估算
        dist = np.linalg.norm(p1 - p0)
        # 这是一个启发式参数，通常取距离的 1.0 到 1.5 倍效果较好
        tangent_scale = dist * 1.2
        
        # 定义起止点的切线向量 (速度方向)
        m0 = np.array([np.cos(th0), np.sin(th0)]) * tangent_scale
        m1 = np.array([np.cos(th1), np.sin(th1)]) * tangent_scale
        
        # --- 赫米特基函数计算 ---
        # h00 = 2t^3 - 3t^2 + 1
        # h10 = t^3 - 2t^2 + t
        # h01 = -2t^3 + 3t^2
        # h11 = t^3 - t^2
        t2 = t * t
        t3 = t2 * t
        
        h00 = 2*t3 - 3*t2 + 1
        h10 = t3 - 2*t2 + t
        h01 = -2*t3 + 3*t2
        h11 = t3 - t2
        
        # 计算 x 和 y 轨迹
        # p(t) = h00*p0 + h10*m0 + h01*p1 + h11*m1
        init_x = h00 * p0[0] + h10 * m0[0] + h01 * p1[0] + h11 * m1[0]
        init_y = h00 * p0[1] + h10 * m0[1] + h01 * p1[1] + h11 * m1[1]
        
        # 计算角度轨迹
        # 方法1：直接插值角度（简单但可能不匹配形状）
        # 方法2：计算赫米特曲线的导数作为角度（更准确）
        # 这里使用方法2
        dx_dt = (6*t2 - 6*t)*p0[0] + (3*t2 - 4*t + 1)*m0[0] + (-6*t2 + 6*t)*p1[0] + (3*t2 - 2*t)*m1[0]
        dy_dt = (6*t2 - 6*t)*p0[1] + (3*t2 - 4*t + 1)*m0[1] + (-6*t2 + 6*t)*p1[1] + (3*t2 - 2*t)*m1[1]
        
        init_theta = np.arctan2(dy_dt, dx_dt)
        init_theta = np.unwrap(init_theta)
        
        # 强制修正起点终点
        init_theta[0] = x0[2]
        init_theta[-1] = xf[2]
        
        return np.concatenate([init_x, init_y, init_theta])

    def _extract_trajectory(self, res, x0, xf):
        """提取轨迹"""
        n = self.n_teb_points
        
        # 提取变量
        z_opt = res['x'].full().flatten()
        
        # 分割变量
        x_opt = z_opt[:n+2]
        y_opt = z_opt[n+2:2*(n+2)]
        th_opt = z_opt[2*(n+2):]
        
        # 确保起点终点正确
        # x_opt[0], y_opt[0], th_opt[0] = x0
        # x_opt[-1], y_opt[-1], th_opt[-1] = xf
        
        # 角度归一化
        for i in range(n + 2):
            th_opt[i] = (th_opt[i] + np.pi) % (2 * np.pi) - np.pi
        
        # 构建轨迹
        trajectory = np.column_stack([x_opt, y_opt, th_opt])
        
        return trajectory