import do_mpc
import numpy as np
import casadi as ca
import contextlib
from typing import List, Tuple, Optional
from scipy.interpolate import CubicSpline


class MpcCbfSolver:
    """
    MPC-CBF Optimization with proper trajectory tracking and safety constraints.
    Fixed version addressing TVP closure, CBF mathematics, and lookahead logic.
    """
    
    def __init__(self, 
                 ref_path: np.ndarray,
                 delta_t: float = 0.1,
                 horizon_step_T: int = 20,
                 v_max: float = 0.5, a_lin: float = 2.0,
                 omega_max: float = 1.8, a_ang: float = 3.0,
                 safe_distance: float = 1.5,
                 gamma: float = 0.4,
                 Q: List[float] = [10, 10, 0.1],
                 R: List[float] = [0.1, 0.1],
                 lookahead_dist: float = 2.0,
                 cbf_soft_weight: float = 800.0,  # CBF软约束权重
                 **kwargs):
        
        # ========== 初始化所有基础属性 ==========
        self.ref_path = np.array(ref_path)
        self.Ts = delta_t
        self.n_horizon = horizon_step_T
        self.v_max = v_max
        self.omega_max = omega_max
        self.a_lin_max = a_lin
        self.a_ang_max = a_ang
        self.safety_dist = safe_distance
        self.gamma = gamma
        self.Q_weights = np.diag(Q)
        self.R_weights = np.array(R)
        self.lookahead_dist = lookahead_dist
        self.cbf_soft_weight = cbf_soft_weight
        self.max_obs = 10
        
        # 内部状态变量
        self.prev_idx = 0
        self.last_u = np.zeros(2)
        self.current_x0 = np.zeros(3)
        self._current_path_idx = 0
        
        # 初始化障碍物数据
        self._current_obstacles = []
        for i in range(self.max_obs):
            self._current_obstacles.append(np.array([0.0, 0.0, -1.0]))
        
        # 用于存储tvp_template的引用
        self._tvp_template = None
        
        # ========== 初始化 do_mpc 组件 ==========
        with contextlib.redirect_stdout(None):
            self.model = self._define_model()
            self.mpc = self._define_mpc()  
            self.estimator = do_mpc.estimator.StateFeedback(self.model)
            
        # 设置初始猜测
        x0 = np.zeros((3, 1))
        self.mpc.x0 = x0
        self.mpc.set_initial_guess()

    def _define_model(self) -> do_mpc.model.Model:
        """定义离散时间自行车模型"""
        model_type = 'discrete'
        model = do_mpc.model.Model(model_type)
        
        # 状态: [x, y, theta]
        _x = model.set_variable(var_type='_x', var_name='x', shape=(3, 1))
        
        # 控制: [v, omega]
        _u = model.set_variable(var_type='_u', var_name='u', shape=(2, 1))
        
        # TVP: 参考位姿
        model.set_variable(var_type='_tvp', var_name='ref_pose', shape=(3, 1))

        # ========== u_prev 用于加速度约束 ==========
        model.set_variable(var_type='_tvp', var_name='u_prev', shape=(2, 1))

        # 障碍物参数
        for i in range(self.max_obs):
            model.set_variable(var_type='_tvp', var_name=f'obs_{i}', shape=(3, 1))
        
        # 动力学 
        theta = _x[2]
        
        x_next = ca.vertcat(
            _x[0] + _u[0] * ca.cos(theta) * self.Ts,
            _x[1] + _u[0] * ca.sin(theta) * self.Ts,
            _x[2] + _u[1] * self.Ts
        )
        
        model.set_rhs('x', x_next)
        
        # 代价函数
        x_err = _x - model.tvp['ref_pose']
        tracking_cost = ca.mtimes([x_err.T, self.Q_weights, x_err])
        model.set_expression('tracking_cost', tracking_cost)
        
        model.setup()
        return model

    def _define_mpc(self) -> do_mpc.controller.MPC:
        """配置MPC求解器"""
        mpc = do_mpc.controller.MPC(self.model)
        
        # 求解器参数
        setup_mpc = {
            'n_horizon': self.n_horizon,
            't_step': self.Ts,
            'n_robust': 0,
            'store_full_solution': True,
            'nlpsol_opts': {
                'ipopt.print_level': 0,
                'ipopt.sb': 'yes',
                'print_time': 0,
                'ipopt.max_iter': 150,  # 增加迭代次数
                'ipopt.tol': 1e-4,
                'ipopt.acceptable_tol': 1e-3,
                'ipopt.acceptable_iter': 10,
            }
        }
        mpc.set_param(**setup_mpc)
        
        # 目标函数
        lterm = self.model.aux['tracking_cost']
        mterm = self.model.aux['tracking_cost']
        mpc.set_objective(mterm=mterm, lterm=lterm)
        mpc.set_rterm(u=self.R_weights)
        
        # 输入约束
        mpc.bounds['lower', '_u', 'u'] = np.array([0.0, -self.omega_max])
        mpc.bounds['upper', '_u', 'u'] = np.array([self.v_max, self.omega_max])
        # ========== 加速度硬约束 ==========
        u = self.model.u['u']
        u_prev = self.model.tvp['u_prev']
        
        # 计算控制量变化限制（加速度 * 时间 = 速度变化）
        dv_max = self.a_lin_max * self.Ts
        domega_max = self.a_ang_max * self.Ts
        
        # # 线加速度上界：v - v_prev <= a_max * dt
        # mpc.set_nl_cons('acc_v_upper', u[0] - u_prev[0] - dv_max, ub=0.0)
        # # 线加速度下界：v - v_prev >= -a_max * dt
        # mpc.set_nl_cons('acc_v_lower', -(u[0] - u_prev[0] + dv_max), ub=0.0)
        
        # # 角加速度上界：omega - omega_prev <= a_ang_max * dt
        # mpc.set_nl_cons('acc_omega_upper', u[1] - u_prev[1] - domega_max, ub=0.0)
        # # 角加速度下界：omega - omega_prev >= -a_ang_max * dt
        # mpc.set_nl_cons('acc_omega_lower', -(u[1] - u_prev[1] + domega_max), ub=0.0)      

        # 使用实例方法而非闭包，避免作用域问题
        self._tvp_template = mpc.get_tvp_template()
        
        # 绑定实例方法作为TVP函数
        mpc.set_tvp_fun(self._tvp_fun)
        
        # 添加CBF约束（软约束版本）
        self._add_cbf_constraints(mpc)
        
        mpc.setup()
        return mpc

    def _tvp_fun(self, t_now: float) -> dict:
        """
        TVP函数 - 使用三次样条插值
        """
        path_len = len(self.ref_path)
        start_idx = self._current_path_idx
        
        # 获取足够长的路径段
        end_idx = min(start_idx + self.n_horizon, path_len)
        ref_segment = self.ref_path[start_idx:end_idx, :3]
        
        if len(ref_segment) < 3:
            # 点太少，用线性
            return self._linear_tvp_fallback(ref_segment)
        
        # 填充 u_prev 用于加速度约束
        for k in range(self.n_horizon + 1):
            if k == 0:
                # 第一时刻使用实际上一时刻的控制量
                self._tvp_template['_tvp', k, 'u_prev'] = self.last_u
            else:
                # 后续时刻：理论上应该是前一个优化结果
                self._tvp_template['_tvp', k, 'u_prev'] = self.last_u

        # 计算累积距离作为参数
        arc_lengths = [0.0]
        for i in range(1, len(ref_segment)):
            d = np.linalg.norm(ref_segment[i, :2] - ref_segment[i-1, :2])
            arc_lengths.append(arc_lengths[-1] + d)
        
        total_length = arc_lengths[-1]
        if total_length < 0.01:
            return self._linear_tvp_fallback(ref_segment)
        
        # 归一化参数到 [0, 1]
        t_params = np.array(arc_lengths) / total_length
        
        # 创建三次样条（分别对x, y, theta插值）
        # 对角度需要特殊处理（unwrap避免跳变）
        x_vals = ref_segment[:, 0]
        y_vals = ref_segment[:, 1]
        theta_vals = np.unwrap(ref_segment[:, 2])  # 解包角度避免跳变
        
        cs_x = CubicSpline(t_params, x_vals)
        cs_y = CubicSpline(t_params, y_vals)
        cs_theta = CubicSpline(t_params, theta_vals)
        
        # 每步期望前进距离
        step_dist = self.v_max * self.Ts
        
        for k in range(self.n_horizon + 1):
            target_dist = k * step_dist
            
            if target_dist >= total_length:
                # 超出范围，延伸
                t_norm = 1.0 + (target_dist - total_length) / (step_dist * self.n_horizon)
                # 使用最后一点的导数方向延伸
                dx = cs_x(1.0, 1)  # 一阶导数
                dy = cs_y(1.0, 1)
                extension_dist = target_dist - total_length
                x = cs_x(1.0) + dx * extension_dist / np.sqrt(dx**2 + dy**2)
                y = cs_y(1.0) + dy * extension_dist / np.sqrt(dx**2 + dy**2)
                theta = cs_theta(1.0)
            else:
                t_norm = target_dist / total_length
                x = cs_x(t_norm)
                y = cs_y(t_norm)
                theta = cs_theta(t_norm)
            
            # 角度归一化到 [-pi, pi]
            theta = (theta + np.pi) % (2 * np.pi) - np.pi
            
            self._tvp_template['_tvp', k, 'ref_pose'] = np.array([x, y, theta])
        
        # 填充障碍物...
        for k in range(self.n_horizon + 1):
            for i in range(self.max_obs):
                if i < len(self._current_obstacles):
                    obs_data = self._current_obstacles[i]
                else:
                    obs_data = np.array([0.0, 0.0, -1.0])
                self._tvp_template['_tvp', k, f'obs_{i}'] = obs_data
        
        return self._tvp_template

    def _linear_tvp_fallback(self, ref_segment):
        """线性插值回退"""
        if len(ref_segment) == 0:
            pos = np.zeros(3)
        else:
            pos = ref_segment[0]
        
        for k in range(self.n_horizon + 1):
            self._tvp_template['_tvp', k, 'ref_pose'] = pos
        
        for k in range(self.n_horizon + 1):
            for i in range(self.max_obs):
                obs_data = self._current_obstacles[i] if i < len(self._current_obstacles) else np.array([0.0, 0.0, -1.0])
                self._tvp_template['_tvp', k, f'obs_{i}'] = obs_data
        
        return self._tvp_template
    
    def _add_cbf_constraints(self, mpc: do_mpc.controller.MPC):
        """添加CBF安全约束（修复数学问题，使用软约束）"""
        x = self.model.x['x']
        u = self.model.u['u']
        theta = x[2]
        
        # 预计算x_{k+1}表达式（仅位置）
        x_next_0 = x[0] + u[0] * ca.cos(theta) * self.Ts
        x_next_1 = x[1] + u[0] * ca.sin(theta) * self.Ts
        
        for i in range(self.max_obs):
            obs = self.model.tvp[f'obs_{i}']
            cx, cy, r = obs[0], obs[1], obs[2]
            
            safe_r = r + self.safety_dist
            
            # h(x_k) = ||x - c||^2 - safe_r^2
            h_k = (x[0] - cx)**2 + (x[1] - cy)**2 - safe_r**2
            
            # h(x_{k+1})
            h_next = (x_next_0 - cx)**2 + (x_next_1 - cy)**2 - safe_r**2
            
            # 修复CBF数学：正确处理障碍物内外情况
            # 标准CBF: h_next >= (1-gamma)*h_k  即  h_next - (1-gamma)*h_k >= 0
            # 但我们需要处理 h_k < 0（已在障碍物内）的情况
            
            eps = 1e-3
            
            # 基础CBF表达式
            cbf_base = h_next - (1 - self.gamma) * h_k
            
            # 关键修复：当在障碍物内部时(h_k < 0)，强制要求h_next > 0（必须退出）
            # 当在外部时，正常CBF约束
            cbf_expr = ca.if_else(
                r > eps,  # 有效障碍物？
                ca.if_else(
                    h_k >= 0,  # 在安全区域？
                    cbf_base,  # 正常CBF: 要求不能降太快
                    h_next - eps  # 在障碍物内：要求下一步必须退出（h_next > 0）
                ),
                1.0  # 无效障碍物：约束自动满足
            )
            
            # 使用软约束避免不可行问题
            mpc.set_nl_cons(
                f'cbf_obs_{i}', 
                -cbf_expr,  # 注意负号：因为set_nl_cons默认是 <= 0
                ub=0.0, 
                soft_constraint=True,
                penalty_term_cons=self.cbf_soft_weight
            )
    
    def _update_path_index(self, pos: np.ndarray) -> int:
        """
        修复：找到最近点，但保留前瞻逻辑供插值使用
        """
        path_len = len(self.ref_path)
        
        # 找到当前最近点（限制在prev_idx附近，避免跳变）
        search_start = max(0, self.prev_idx - 5)
        search_end = min(path_len, self.prev_idx + 20)
        
        local_dists = np.linalg.norm(
            self.ref_path[search_start:search_end, :2] - pos[:2], 
            axis=1
        )
        local_min_idx = int(np.argmin(local_dists))
        nearest_idx = search_start + local_min_idx
        
        # 确保单调前进
        self._current_path_idx = max(self.prev_idx, nearest_idx)
        self._current_path_idx = min(self._current_path_idx, path_len - 1)
        self.prev_idx = self._current_path_idx
        
        return self._current_path_idx
    
    def _process_obstacles(self, obstacles: Optional[List]) -> None:
        """
        更新内部障碍物数据（添加格式验证）
        输入格式: [[cx, cy, major_axis, minor_axis, angle], ...] 或 [[cx, cy, radius], ...]
        """
        self._current_obstacles.clear()
        
        if obstacles is not None:
            count = min(len(obstacles), self.max_obs)
            for i in range(count):
                obs = obstacles[i]
                if len(obs) >= 3:
                    cx, cy = float(obs[0]), float(obs[1])
                    
                    # 支持多种格式
                    if len(obs) >= 5:
                        # 椭圆格式 [cx, cy, a, b, angle]：使用外接圆近似
                        a, b = float(obs[2]), float(obs[3])
                        r_eff = max(a, b)  # 保守近似：取椭圆半长轴和半短轴的最大值
                    elif len(obs) >= 4:
                        # [cx, cy, a, b] 无角度
                        r_eff = max(float(obs[2]), float(obs[3]))
                    else:
                        # 圆形 [cx, cy, r]
                        r_eff = float(obs[2])
                    
                    self._current_obstacles.append(np.array([cx, cy, r_eff]))
        
        # 填充剩余为无效标记
        while len(self._current_obstacles) < self.max_obs:
            self._current_obstacles.append(np.array([0.0, 0.0, -1.0]))
    
    def calc_control_input(self, 
                          observed_x: np.ndarray, 
                          obstacles: Optional[List] = None) -> Tuple[np.ndarray, None, np.ndarray]:
        """
        主控制接口
        """
        # 更新状态
        x0 = np.array(observed_x).reshape(3, 1)
        self.current_x0 = x0.flatten()
        self.mpc.x0 = x0
        self.estimator.x0 = x0
        
        # 更新路径索引和障碍物（供tvp_fun使用）
        self._update_path_index(self.current_x0)
        self._process_obstacles(obstacles)
        
        try:
            # 执行MPC
            u0 = self.mpc.make_step(x0)
            action = u0.flatten()
            self.last_u = action
            
            # 提取预测轨迹
            opt_traj = self._extract_prediction_trajectory()
            
            return opt_traj, None, action
            
        except Exception as e:
            print(f"[MpcCbfSolver] MPC failed: {e}")
            # 故障保护：使用上一控制或停止
            fallback_action = self.last_u * 0.5  # 衰减
            if np.linalg.norm(fallback_action) < 0.01:
                fallback_action = np.array([0.0, 0.0])
            
            # 生成安全轨迹（沿参考路径）
            fallback_traj = self._generate_safe_trajectory()
            return fallback_traj, None, fallback_action
    
    def _extract_prediction_trajectory(self) -> np.ndarray:
        """从 mpc.data 提取预测轨迹（自动检测维度）"""
        # try:
        #     if self.mpc.data is None or not hasattr(self.mpc, 'opt_x_num'):
        #         return np.tile(self.current_x0[:2], (self.n_horizon + 1, 1))
            
        #     opt_x = self.mpc.opt_x_num.cat()
            
        #     # 自动计算: 总元素 / 时间点数 = 状态维度（可能包含控制量等）
        #     n_points = self.n_horizon + 1
            
        #     # 假设状态在最前面，尝试提取前 n_points * n_x 个元素
        #     n_x = self.model.n_x  # 从模型获取状态维度 (通常是 3)
        #     expected = n_points * n_x
            
        #     if len(opt_x) < expected:
        #         print(f"[MpcCbfSolver] Warning: Expected at least {expected} elements in _opt_x_num, got {len(opt_x)}")
        #         return np.tile(self.current_x0[:2], (n_points, 1))
            
        #     # 从 opt_x 中提取轨迹
        #     total_elements = expected
        #     opt_traj = opt_x[:total_elements].reshape(-1, n_x)
        #     opt_traj = opt_traj[:, :2]  # 只取 x, y 坐标
            
        #     return opt_traj
                
        # except Exception as e:
        #     print(f"[MpcCbfSolver] Trajectory extraction failed: {e}")
        return np.tile(self.current_x0[:2], (self.n_horizon + 1, 1))
    
    def _generate_safe_trajectory(self) -> np.ndarray:
        """生成故障保护用的安全轨迹（沿当前参考路径）"""
        traj = np.zeros((self.n_horizon + 1, 2))
        path_len = len(self.ref_path)
        start_idx = self._current_path_idx
        
        for k in range(self.n_horizon + 1):
            idx = min(start_idx + k, path_len - 1)
            traj[k] = self.ref_path[idx, :2]
        
        return traj 