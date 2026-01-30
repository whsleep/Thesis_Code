import numpy as np
import math

class DwaplanSolver:
    """
    基于动态窗口法 (DWA) 的局部路径规划器
    适配椭圆障碍物输入 (center_list)
    """
    def __init__(self, 
                 ref_path: np.ndarray,
                 delta_t: float = 0.1,
                 v_max: float = 5.0, 
                 omega_max: float = 3.0, 
                 a_max: float = 2.0, 
                 alpha_max: float = 3.0, 
                 predict_time: float = 2.0,
                 sample_v_count: int = 10,
                 sample_w_count: int = 20,
                 w_heading: float = 0.15, 
                 w_dist: float = 0.4, 
                 w_vel: float = 0.5, 
                 w_obs: float = 10.0,
                 w_track: float = 1.0, # 新增：路径跟踪权重 (Path Tracking Cost Weight)
                 safe_distance: float = 0.3,
                 lookahead_dist: float = 1.5,
                 **kwargs):
        
        # 路径与车辆参数
        self.ref_path = ref_path
        self.dt = delta_t
        self.v_max = v_max
        self.omega_max = omega_max
        self.a_max = a_max
        self.alpha_max = alpha_max
        self.predict_time = predict_time
        self.safe_distance = safe_distance
        
        # 采样分辨率
        self.sample_v_count = sample_v_count
        self.sample_w_count = sample_w_count
        
        # 权重参数
        self.w_heading = w_heading
        self.w_dist = w_dist
        self.w_vel = w_vel
        self.w_obs = w_obs
        self.w_track = w_track # 记录跟踪权重
        
        # 路径跟踪参数
        self.lookahead_dist = lookahead_dist
        self.prev_idx = 0 
        
        # 内部状态记录
        self.last_v = 0.0
        self.last_w = 0.0
        self.dim_x = 3 

    def calc_control_input(self, observed_x: np.ndarray, obstacles: list = None):
        """
        标准接口：计算控制量
        input: 
            observed_x: [x, y, theta]
            obstacles: 包含椭圆参数的列表 [cx, cy, a, b, theta, cos, sin]
        """
        if observed_x.ndim > 1:
            observed_x = observed_x.flatten()
            
        x, y, theta = observed_x[0], observed_x[1], observed_x[2]
        
        # 确定局部目标点 (更新 self.prev_idx)
        local_goal = self._get_local_goal_from_path(x, y)
        
        # 准备 DWA 状态
        current_state_dwa = np.array([x, y, theta, self.last_v, self.last_w])
        
        # 传入 obstacles 进行搜索
        best_u, best_traj, all_trajectories = self._dwa_search(current_state_dwa, local_goal, obstacles)
        
        # 更新内部状态
        self.last_v = best_u[0]
        self.last_w = best_u[1]
        
        # 格式化输出
        if len(all_trajectories) > 0:
            samp_traj_np = np.array(all_trajectories)
        else:
            samp_traj_np = np.zeros((1, 1, 3))

        return best_traj, samp_traj_np, best_u

    def _get_local_goal_from_path(self, x, y):
        """寻找前瞻点"""
        search_range = 20 
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

    def _dwa_search(self, state, goal, obstacles):
        """执行 DWA 采样"""
        x, y, theta, v, w = state
        
        dw = self._calc_dynamic_window(v, w)
        
        best_u = np.array([0.0, 0.0])
        best_traj = np.array(state[:3]).reshape(1, 3)
        min_cost = float('inf')
        all_trajectories = []
        
        v_samples = np.linspace(dw[0], dw[1], self.sample_v_count)
        w_samples = np.linspace(dw[2], dw[3], self.sample_w_count)
        
        for v_s in v_samples:
            for w_s in w_samples:
                traj = self._predict_trajectory(np.array([x, y, theta]), v_s, w_s)
                
                # 将 obstacles 传入代价计算
                cost = self._calc_cost(traj, goal, v_s, obstacles)
                
                all_trajectories.append(traj)
                
                if cost < min_cost:
                    min_cost = cost
                    best_u = np.array([v_s, w_s])
                    best_traj = traj
                    
        return best_u, best_traj, all_trajectories

    def _calc_dynamic_window(self, v, w):
        vs = [-self.v_max, self.v_max, -self.omega_max, self.omega_max]
        vd = [v - self.a_max * self.dt, v + self.a_max * self.dt,
              w - self.alpha_max * self.dt, w + self.alpha_max * self.dt]
        
        vmin = max(vs[0], vd[0])
        vmax = min(vs[1], vd[1])
        wmin = max(vs[2], vd[2])
        wmax = min(vs[3], vd[3])
        return [vmin, vmax, wmin, wmax]

    def _predict_trajectory(self, state_init, v, w):
        steps = int(self.predict_time / self.dt)
        traj = np.zeros((steps, 3))
        
        curr_x, curr_y, curr_theta = state_init
        
        for i in range(steps):
            curr_x += v * np.cos(curr_theta) * self.dt
            curr_y += v * np.sin(curr_theta) * self.dt
            curr_theta += w * self.dt
            
            traj[i, 0] = curr_x
            traj[i, 1] = curr_y
            traj[i, 2] = curr_theta
            
        return traj

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

    def _calc_cost(self, traj, goal, v, obstacles):
        """
        计算代价，包含椭圆障碍物的硬约束(碰撞)与软约束(距离评分)
        新增：全轨迹跟踪代价 (Path Tracking)
        """
        # 1. 航向代价 (只看末端)
        end_x, end_y, end_theta = traj[-1]
        dx = goal[0] - end_x
        dy = goal[1] - end_y
        target_angle = math.atan2(dy, dx)
        
        angle_diff = target_angle - end_theta
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
        cost_heading = abs(angle_diff)
        
        # 2. 距离代价 (只看末端)
        cost_dist = math.hypot(dx, dy)
        
        # 3. 速度代价
        cost_vel = self.v_max - abs(v)
        
        # 4. 路径跟踪代价 (Path Tracking Cost)
        # 对除了末端采样点以外的采样点进行计算
        cost_track = 0.0
        # traj[:-1] 排除最后一个点
        track_points = traj[:-1]
        if len(track_points) > 0:
            tx = track_points[:, 0]
            ty = track_points[:, 1]
            
            # 批量获取距离最近的参考点距离平方
            dist_sq_vals = self._batch_get_nearest_dist_sq(tx, ty)
            
            # 计算均方误差 (MSE) 作为代价
            cost_track = np.mean(dist_sq_vals)

        # 5. 障碍物代价 (硬约束 + 软约束)
        cost_obs = 0.0
        min_algebraic_dist_all = float('inf')

        if obstacles is not None and len(obstacles) > 0:
            traj_x = traj[:, 0]
            traj_y = traj[:, 1]
            
            for obs in obstacles:
                cx, cy, a, b, _, cos_ot, sin_ot = obs
                safe_a = a + self.safe_distance
                safe_b = b + self.safe_distance
                
                dx_vec = traj_x - cx
                dy_vec = traj_y - cy
                
                lx = dx_vec * cos_ot + dy_vec * sin_ot
                ly = -dx_vec * sin_ot + dy_vec * cos_ot
                
                dist_sq = (lx / safe_a)**2 + (ly / safe_b)**2
                
                min_dist_on_traj = np.min(dist_sq)

                if min_dist_on_traj < min_algebraic_dist_all:
                    min_algebraic_dist_all = min_dist_on_traj


        cost_obs = 1.0 / min_algebraic_dist_all 

        # 6. 总代价 (加入 w_obs 和 w_track)
        total_cost = (self.w_heading * cost_heading + 
                      self.w_dist * cost_dist + 
                      self.w_vel * cost_vel +
                      self.w_obs * cost_obs +
                      self.w_track * cost_track) # 加入跟踪代价
                      
        return total_cost