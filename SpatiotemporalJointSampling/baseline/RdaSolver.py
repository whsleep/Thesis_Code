import numpy as np
import sys
import os
from collections import namedtuple
from baseline.RDA_planner.mpc import MPC

# --- 2. 数据结构定义 ---
CarTuple = namedtuple('car', 'G h cone_type wheelbase max_speed max_acce dynamics')
ObsTuple = namedtuple('obstacle', 'center radius vertex cone_type velocity')

class RdaSolver:
    def __init__(self, 
                 ref_path: np.ndarray,
                 robot_info, 
                 delta_t: float = 0.1,
                 v_max: float = 5.0,
                 omega_max: float = 3.0,
                 a_max: float = 2.0,
                 **kwargs): # 接收但不使用 params 中的 robot_l/robot_w
        """
        初始化 RDA MPC 求解器
        """
        self.ref_path = [pt.reshape(-1, 1) for pt in ref_path]
        self.dt = delta_t
        
        # --- 3. 构建机器人模型 ---
        G = robot_info.G
        h = robot_info.h
        cone_type = robot_info.cone_type
        wheelbase = robot_info.wheelbase
        print(f"[INFO] RdaSolver loaded robot params from env: cone={cone_type}, wheelbase={wheelbase}")

        max_speed = [v_max, omega_max]
        max_acce = [a_max, 5.0]
        
        # 实例化 CarTuple
        self.car = CarTuple(G, h, cone_type, wheelbase, max_speed, max_acce, 'diff')
        
        # --- 4. 初始化 MPC ---
        self.mpc = MPC(
            self.car, 
            self.ref_path, 
            receding=10, 
            sample_time=self.dt, 
            process_num=4, 
            iter_num=2, 
            max_edge_num=4, 
            max_obs_num=5,
            obstacle_order=True, 
            wu=1.0, 
            slack_gain=10
        )
        
    def calc_control_input(self, robot_state: np.ndarray, obstacles: list = None):
        """
        计算控制量
        """
        # 1. 维度修正: (N,) -> (N, 1)
        if robot_state.ndim == 1:
            state_input = robot_state.reshape(-1, 1)
        else:
            state_input = robot_state

        # 2. 状态补全: [x,y,th] -> [x,y,th,v]
        if state_input.shape[0] < 4:
            padding = np.zeros((4 - state_input.shape[0], 1))
            state_input = np.vstack([state_input, padding])

        # 3. 障碍物转换: 椭圆参数 -> 矩形顶点
        rda_obs_list = []
        if obstacles is not None:
            for obs_params in obstacles:
                # [cx, cy, a, b, theta, ...]
                cx, cy, a, b, theta_rad = obs_params[0], obs_params[1], obs_params[2], obs_params[3], obs_params[4]
                
                # 局部顶点
                corners_local = np.array([
                    [a, b], [a, -b], [-a, -b], [-a, b]
                ])
                
                # 旋转与平移
                cos_t = np.cos(theta_rad)
                sin_t = np.sin(theta_rad)
                R = np.array([[cos_t, -sin_t], [sin_t,  cos_t]])
                
                # 变换到全局坐标并转置
                corners_global = np.array([cx, cy]) + (R @ corners_local.T).T
                vertices = corners_global.T 
                
                rda_obs_list.append(ObsTuple(None, None, vertices, 'Rpositive', 0))

        try:
            opt_vel, info = self.mpc.control(state_input, 3, rda_obs_list)
            
            action = np.array(opt_vel).flatten()
            if action.size < 2:
                action = np.array([0.0, 0.0])
                
            optimal_traj = info.get('opt_state_list', [])
            sampled_traj = []
            
            return optimal_traj, sampled_traj, action

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[Error] RdaSolver failure: {e}")
            return [], [], np.array([0.0, 0.0])