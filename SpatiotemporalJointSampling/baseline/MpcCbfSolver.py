import do_mpc
import numpy as np
import casadi as ca
import contextlib
from typing import List, Tuple, Optional


class MpcCbfSolver:
    """
    MPC-CBF Optimization with proper trajectory tracking and safety constraints.
    """
    
    def __init__(self, 
                 ref_path: np.ndarray,
                 delta_t: float = 0.1,
                 horizon_step_T: int = 20,
                 v_max: float = 0.5,
                 omega_max: float = 1.8,
                 safe_distance: float = 1.5,
                 gamma: float = 0.4,
                 Q: List[float] = [1.0, 1.0, 0.5],
                 R: List[float] = [0.1, 0.1],
                 lookahead_dist: float = 2.0,
                 **kwargs):
        
        # ========== 初始化所有基础属性 ==========
        self.ref_path = np.array(ref_path)
        self.Ts = delta_t
        self.n_horizon = horizon_step_T
        self.v_max = v_max
        self.omega_max = omega_max
        self.safety_dist = safe_distance
        self.gamma = gamma
        self.Q_weights = np.diag(Q)
        self.R_weights = np.array(R)
        self.lookahead_dist = lookahead_dist
        self.max_obs = kwargs.get('max_obs', 10)
        
        # 内部状态变量
        self.prev_idx = 0
        self.last_u = np.zeros(2)
        self.current_x0 = np.zeros(3)
        self._current_path_idx = 0
        
        # 初始化障碍物数据
        self._current_obstacles = []
        for i in range(self.max_obs):
            self._current_obstacles.append(np.array([0.0, 0.0, -1.0]))
        
        # 用于存储tvp_template的引用（由_define_mpc设置）
        self._tvp_template = None
        
        # ========== 初始化 do_mpc 组件 ==========
        with contextlib.redirect_stdout(None):
            self.model = self._define_model()
            self.mpc = self._define_mpc()  # 这里会设置 self._tvp_template
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
        
        # 障碍物参数
        for i in range(self.max_obs):
            model.set_variable(var_type='_tvp', var_name=f'obs_{i}', shape=(3, 1))
        
        # 动力学
        a = 1e-6
        theta = _x[2]
        
        x_next = ca.vertcat(
            _x[0] + (_u[0] * ca.cos(theta) - a * _u[1] * ca.sin(theta)) * self.Ts,
            _x[1] + (_u[0] * ca.sin(theta) + a * _u[1] * ca.cos(theta)) * self.Ts,
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
                'ipopt.max_iter': 100,
                'ipopt.tol': 1e-4,
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
        
        # 关键修复：先获取template，然后使用闭包绑定
        tvp_template = mpc.get_tvp_template()
        self._tvp_template = tvp_template  # 保存引用供后续使用
        
        # 创建闭包函数，绑定必要的局部变量
        def tvp_fun(t_now):
            """
            TVP函数 - 闭包，绑定了tvp_template和self
            """
            # 填充参考轨迹
            path_len = len(self.ref_path)
            start_idx = self._current_path_idx
            
            for k in range(self.n_horizon + 1):
                # 参考轨迹
                path_idx = min(start_idx + k, path_len - 1)
                tvp_template['_tvp', k, 'ref_pose'] = self.ref_path[path_idx, :3]
                
                # 障碍物数据
                for i in range(self.max_obs):
                    if i < len(self._current_obstacles):
                        obs_data = self._current_obstacles[i]
                    else:
                        obs_data = np.array([0.0, 0.0, -1.0])
                    tvp_template['_tvp', k, f'obs_{i}'] = obs_data
            
            return tvp_template
        
        mpc.set_tvp_fun(tvp_fun)
        
        # 添加CBF约束
        self._add_cbf_constraints(mpc)
        
        mpc.setup()
        return mpc
    
    def _add_cbf_constraints(self, mpc: do_mpc.controller.MPC):
        """添加CBF安全约束"""
        x = self.model.x['x']
        u = self.model.u['u']
        theta = x[2]
        a = 1e-6
        
        # 预计算x_{k+1}表达式（仅位置）
        x_next_0 = x[0] + (u[0] * ca.cos(theta) - a * u[1] * ca.sin(theta)) * self.Ts
        x_next_1 = x[1] + (u[0] * ca.sin(theta) + a * u[1] * ca.cos(theta)) * self.Ts
        
        for i in range(self.max_obs):
            obs = self.model.tvp[f'obs_{i}']
            cx, cy, r = obs[0], obs[1], obs[2]
            
            safe_r = r + self.safety_dist
            
            # h(x_k)
            h_k = (x[0] - cx)**2 + (x[1] - cy)**2 - safe_r**2
            
            # h(x_{k+1})
            h_next = (x_next_0 - cx)**2 + (x_next_1 - cy)**2 - safe_r**2
            
            # CBF: h_next >= (1-gamma)*h_k  =>  (1-gamma)*h_k - h_next <= 0
            cbf_expr = (1 - self.gamma) * h_k - h_next
            
            # 无效障碍物检查 (r < 0 表示无效)
            eps = 1e-3
            cbf_expr = ca.if_else(r > eps, cbf_expr, -1.0)
            
            mpc.set_nl_cons(f'cbf_obs_{i}', cbf_expr, ub=0.0, 
                          soft_constraint=False)
    
    def _update_path_index(self, pos: np.ndarray) -> int:
        """找到并更新参考路径上最近的点"""
        dists = np.linalg.norm(self.ref_path[:, :2] - pos[:2], axis=1)
        min_idx = int(np.argmin(dists))
        self._current_path_idx = min_idx
        self.prev_idx = min_idx
        return min_idx
    
    def _process_obstacles(self, obstacles: Optional[List]) -> None:
        """更新内部障碍物数据"""
        # 清空并重新填充
        self._current_obstacles.clear()
        
        if obstacles is not None:
            count = min(len(obstacles), self.max_obs)
            for i in range(count):
                obs = obstacles[i]
                if len(obs) >= 5:
                    cx, cy = obs[0], obs[1]
                    r_eff = max(obs[2], obs[3])
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
            import traceback
            traceback.print_exc()
            # 故障保护
            fallback_action = np.array([0.0, 0.0])
            fallback_traj = np.tile(self.current_x0[:2], (self.n_horizon + 1, 1))
            return fallback_traj, None, fallback_action
    
    def _extract_prediction_trajectory(self) -> np.ndarray:
        """从mpc.data提取预测轨迹"""
        traj = np.zeros((self.n_horizon + 1, 2))
        
        try:
            for k in range(self.n_horizon + 1):
                x_pred = self.mpc.data.prediction(('_x', 'x'), t_ind=k)
                if x_pred is not None:
                    traj[k] = x_pred[:2].flatten()
                else:
                    traj[k] = self.current_x0[:2]
        except:
            traj = np.tile(self.current_x0[:2], (self.n_horizon + 1, 1))
            
        return traj