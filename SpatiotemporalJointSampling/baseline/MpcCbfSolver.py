import do_mpc
import numpy as np
import casadi as ca
import contextlib


class MpcCbfSolver:
    """MPC-CBF Optimization problem:

    min Σ_{k=0}{N-1} 1/2*x'_k^T*Q*x'_k + 1/2*u_k^T*R*u_k   over u
    s.t.
        x_{k+1} = x_k + B*u_k*T_s
        x_min <= x_k <= x_max
        u_min <= u_k <= u_max
        x_0 = x(0)
        Δh(x_k, u_k) >= -γ*h(x_k)

    where x'_k = x_{des_k} - x_k
    """
    def __init__(self, 
                 ref_path: np.ndarray,
                 delta_t: float = 0.1,
                 horizon_step_T: int = 20,
                 v_max: float = 0.5,
                 omega_max: float = 1.8,
                 safe_distance: float = 1.5,
                 gamma: float = 0.4,
                 Q: list = [10.0, 10.0, 0.1], # x, y, theta weights
                 R: list = [1.0, 0.1],        # v, omega weights
                 lookahead_dist: float = 2.0,
                 **kwargs):
        
        # --- 参数保存 ---
        self.ref_path = ref_path
        self.Ts = delta_t
        self.n_horizon = horizon_step_T
        self.v_max = v_max
        self.omega_max = omega_max
        self.safety_dist = safe_distance
        self.gamma = gamma
        self.Q_weights = np.diag(Q)
        self.R_weights = np.array(R)
        self.lookahead_dist = lookahead_dist

        # 内部状态
        self.max_obs = 10  # 硬编码最大障碍物数量以固定计算图
        self.prev_idx = 0
        self.last_u = np.zeros(2)
        
        # 用于存储当前步的 TVP 数据 (Ref traj & Obstacles)
        self.current_ref_seq = np.zeros((self.n_horizon + 1, 3))
        self.current_obs_data = np.zeros((self.max_obs, 3)) # x, y, r

        # --- 初始化 do_mpc 组件 ---
        # 使用 suppress_stdout 避免初始化时的刷屏
        with contextlib.redirect_stdout(None):
            self.model = self._define_model()
            self.mpc = self._define_mpc()
            # 这里的 estimator 只是为了满足 do_mpc 架构，实际状态由 sim.py 提供
            self.estimator = do_mpc.estimator.StateFeedback(self.model)

    def _define_model(self):
        """定义系统动力学"""
        # 离散模型
        model_type = 'discrete'
        model = do_mpc.model.Model(model_type)
        # 1. 状态变量 [x, y, theta]
        _x = model.set_variable(var_type='_x', var_name='x', shape=(3, 1)) 
        # 2. 控制变量 [v, omega]
        _u = model.set_variable(var_type='_u', var_name='u', shape=(2, 1)) 
        # 3. 参考轨迹 TVP(时变参数)
        model.set_variable(var_type='_tvp', var_name='ref_pose', shape=(3, 1))
        # 4. 障碍物 TVP(时变参数) (x, y, r) * max_obs 
        for i in range(self.max_obs):
            model.set_variable(var_type='_tvp', var_name=f'obs_{i}', shape=(3, 1))
        # 4. 动力学方程 差分模型
        theta = _x[2]
        a = 1e-9  # 防止相对阶奇异的小量
        # B 矩阵构建 \dot{x} = Ax + Bu
        # dx = v*cos(theta) - a*sin(theta)*w
        # dy = v*sin(theta) + a*cos(theta)*w
        # dtheta = w
        x_next = ca.vertcat(
            _x[0] + (ca.cos(theta) * _u[0] - a * ca.sin(theta) * _u[1]) * self.Ts,
            _x[1] + (ca.sin(theta) * _u[0] + a * ca.cos(theta) * _u[1]) * self.Ts,
            _x[2] + _u[1] * self.Ts
        )

        # 状态更新方程 x_{k+1} = f(x_k, u_k)
        model.set_rhs(var_name='x', expr=x_next, process_noise=False)

        # 5. 代价函数表达式 (Tracking Cost)
        # J = (x - x_ref)^T Q (x - x_ref)
        x_err = _x - model.tvp['ref_pose']
        cost_expr = x_err.T @ self.Q_weights @ x_err
        
        model.set_expression(expr_name='cost', expr=cost_expr)
        model.setup()
        return model

    def _define_mpc(self):
        """配置 MPC 控制器与 CBF 约束"""
        mpc = do_mpc.controller.MPC(self.model)

        # 1. 基础参数
        setup_mpc = {
            'n_horizon': self.n_horizon,
            't_step': self.Ts,
            'n_robust': 0,
            'store_full_solution': True,
            'nlpsol_opts': {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0}
        }
        mpc.set_param(**setup_mpc)

        # 2. 目标函数
        lterm = self.model.aux['cost'] # 阶段代价
        mterm = self.model.aux['cost'] # 终端代价
        mpc.set_rterm(u=self.R_weights) # 输入代价
        mpc.set_objective(mterm=mterm, lterm=lterm) # 目标函数设置

        # 3. 输入约束
        mpc.bounds['lower', '_u', 'u'] = np.array([-self.v_max, -self.omega_max])
        mpc.bounds['upper', '_u', 'u'] = np.array([self.v_max, self.omega_max])

        # 4. TVP 函数配置 (将 Python 数据注入 CasADi)
        tvp_template = mpc.get_tvp_template()
        
        def tvp_fun(t_now):
            # do_mpc 在预测时会调用此函数获取 horizon 内每一步的参数
            # 我们在 calc_control_input 中预先计算好了 current_ref_seq 和 current_obs_data
            # do_mpc 内部逻辑比较复杂，这里简化处理：假设 t_now 对应 horizon 的第 0 步
            # 我们直接返回一个固定的 template，但 template 的值必须是动态填入的
            
            # 由于 do_mpc 是为连续仿真设计的，这里我们要 hack 一下：
            # 在 calc_control_input 中更新了 self.current_* 后，这里直接返回
            # 注意：tvp_template 是 (n_horizon+1) 长度的结构
            
            for k in range(self.n_horizon + 1):
                # 填入参考轨迹
                tvp_template['_tvp', k, 'ref_pose'] = self.current_ref_seq[k]
                
                # 填入障碍物数据
                for i in range(self.max_obs):
                    tvp_template['_tvp', k, f'obs_{i}'] = self.current_obs_data[i]
            
            return tvp_template
        # 注册 TVP (自定义)加载函数
        mpc.set_tvp_fun(tvp_fun)

        # 5. 添加 CBF 约束 (set_nl_cons)
        # 引用 mpc_cbf.py: h(x_{k+1}) >= (1-gamma)*h(x_k)
        # do_mpc 格式: expression <= ub (默认为0)
        # 变换: (1-gamma)*h(x_k) - h(x_{k+1}) <= 0
        
        # 获取 x_{k+1} 的符号表达式
        # 注意: do_mpc 的 set_nl_cons 是对当前时刻 k 施加约束，涉及 u_k
        # 这里的 model.x 是 x_k
        # 我们需要用 model 里的方程显式写出 x_{k+1}
        theta = self.model.x['x', 2]
        u_v = self.model.u['u', 0]
        u_w = self.model.u['u', 1]
        a = 1e-9
        
        # 显式动力学预测 x_{k+1}
        x_next_0 = self.model.x['x', 0] + (ca.cos(theta)*u_v - a*ca.sin(theta)*u_w) * self.Ts
        x_next_1 = self.model.x['x', 1] + (ca.sin(theta)*u_v + a*ca.cos(theta)*u_w) * self.Ts
        # CBF 只关心位置，不需要 theta_next
        for i in range(self.max_obs):
            # 获取当前步的障碍物参数 (Symbolic)
            obs_x = self.model.tvp[f'obs_{i}'][0]
            obs_y = self.model.tvp[f'obs_{i}'][1]
            obs_r = self.model.tvp[f'obs_{i}'][2]
            
            # 安全距离阈值
            safe_r = obs_r + self.safety_dist
            
            # h(x_k)
            dist_sq_k = (self.model.x['x', 0] - obs_x)**2 + (self.model.x['x', 1] - obs_y)**2
            h_k = dist_sq_k - safe_r**2
            
            # h(x_{k+1})
            dist_sq_next = (x_next_0 - obs_x)**2 + (x_next_1 - obs_y)**2
            h_next = dist_sq_next - safe_r**2
            
            # 约束: (1-gamma)*h_k - h_next <= 0
            # 增加一个开关：如果 obs_r < 0.001 (无效障碍物)，则约束失效 (设为负无穷或恒成立)
            # 这里的技巧是：如果无效，让 h 变得很大，或者让约束项变为负大值
            # 简单做法：乘以有效性系数 is_valid = if_else(obs_r > 1e-3, 1, 0)
            
            cbf_expr = (1 - self.gamma) * h_k - h_next
            # 如果 obs_r 近似 0，则忽略此约束 (cbf_expr * 0 <= 0 恒成立)
            is_valid = ca.if_else(obs_r > 1e-3, 1.0, 0.0)
            
            mpc.set_nl_cons(f'cbf_obs_{i}', cbf_expr * is_valid, ub=0)

        mpc.setup()
        return mpc

    def calc_control_input(self, observed_x: np.ndarray, obstacles: list = None):
        """
        sim.py 调用的标准接口
        observed_x: [x, y, theta]
        obstacles: [cx, cy, a, b, theta, ...]
        """
        # 1. 状态更新
        # do_mpc 需要 (3,1) 的 numpy array
        x0 = observed_x.reshape(3, 1)
        self.mpc.x0 = x0
        self.estimator.x0 = x0 
 
        # 2. 准备参考轨迹 (Reference Trajectory)
        # 逻辑：找到最近点，向后截取 n_horizon 长度
        dists = np.linalg.norm(self.ref_path[:, :2] - x0[:2].flatten(), axis=1)
        min_idx = np.argmin(dists)
        self.prev_idx = min_idx
        
        # 填充 current_ref_seq
        idx_end = min(min_idx + self.n_horizon + 1, len(self.ref_path))
        len_chunk = idx_end - min_idx
        
        self.current_ref_seq.fill(0) # 清零
        self.current_ref_seq[:len_chunk] = self.ref_path[min_idx:idx_end, :3]
        
        # 如果路径不够长，用最后一个点填充剩余部分
        if len_chunk < self.n_horizon + 1:
            self.current_ref_seq[len_chunk:] = self.ref_path[-1, :3]

        # 3. 准备障碍物数据 (Ellipses -> Circles)
        self.current_obs_data.fill(0) # 清零
        
        if obstacles:
            # sim.py 传入的 obstacles 是 list of [cx, cy, a, b, theta, cos, sin]
            count = min(len(obstacles), self.max_obs)
            for i in range(count):
                obs = obstacles[i]
                cx, cy = obs[0], obs[1]
                # 简化：取长轴作为半径
                r_max = max(obs[2], obs[3])
                self.current_obs_data[i] = [cx, cy, r_max]

        # 4. 执行控制计算
        try:
            # 关键：重置历史，防止在 sim.py 循环中内存无限增长
            self.mpc.reset_history() 
            
            u0 = self.mpc.make_step(x0)
            
            # 提取控制量 [v, omega]
            action = u0.flatten()
            self.last_u = action

            # --- 轨迹提取与维度修复 ---
            # 1. 调用 prediction 方法 (注意是圆括号调用，不是方括号索引)
            #    参数 ('_x', 'x') 表示获取状态 x 的预测值
            #    返回结果通常是一个 list，取第 0 个元素得到数组
            pred_res = self.mpc.data._opt_x_num;
            
            # 2. 转为 numpy array
            opt_traj = np.array(pred_res)
            flat_vec = opt_traj.flatten()
            total_elements = (self.n_horizon + 1) * 3  # 每个点有 3 个状态维度 (x, y, theta)
            opt_traj = flat_vec[:total_elements].reshape(-1, 3)
            opt_traj = opt_traj[:,0:2]
            
            return opt_traj, None, action

        except Exception as e:
            print(f"[MpcCbfSolver] Error: {e}")
            # 发生错误时的保底策略：返回包含当前位置的单点轨迹
            # 确保返回 (1, 3) 的二维数组，而不是 (3,) 的一维数组，否则 irsim 会报错
            fallback_traj = np.array([x0.flatten()])
            return fallback_traj, None, np.zeros(2)