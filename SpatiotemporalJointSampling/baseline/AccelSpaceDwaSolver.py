import numpy as np

class AccelSpaceDwaSolver:
    """
    基于加速度空间 (ax, ay) 采样并直接输出 [v, w] 指令的求解器
    """
    def __init__(self, x0, xf, v_max=1.5, a_limit=2.0, N=11,
                 dt=0.1, predict_time=2.0,
                 w_heading=0.2, w_dist=0.1, w_vel=0.1, safe_distance=0.3):
        
        self.v_max = v_max        
        self.A = a_limit          # 加速度采样范围 [-A, A]
        self.N = N                # 采样点数 N*N
        self.dt = dt
        self.predict_time = predict_time
        
        self.w_heading = w_heading
        self.w_dist = w_dist
        self.w_vel = w_vel
        self.safe_distance = safe_distance
        
        # 维护当前的速度状态，对应公式中的 v0, vx0, vy0
        self.v = 0.0
        self.vx = 0.0
        self.vy = 0.0

    def solve(self, state, goal, obstacles):
        """
        state: [x, y, theta]
        """
        curr_x, curr_y, theta0 = state
        
        best_traj = None
        min_cost = float('inf')
        best_vw = [0.0, 0.0]  # 存储最优的 [v, w]
        all_trajectories = []

        # 在加速度空间采样
        a_samples = np.linspace(-self.A, self.A, self.N)

        for ax in a_samples:
            for ay in a_samples:
                # --- 执行你提供的转换公式 ---
                
                # 1. 计算线加速度 av
                # 利用加速度矢量与当前朝向 theta0 的偏差
                accel_mag = np.sqrt(ax**2 + ay**2)
                accel_dir = np.arctan2(ay, ax)
                angle_diff = accel_dir - theta0
                
                av = accel_mag * np.cos(angle_diff) 
                
                # 2. 计算线速度 v = v0 + av * dT
                next_v = self.v + av * self.dt
                next_v = np.clip(next_v, -self.v_max, self.v_max)
                
                # 3. 计算角速度 w = ||(vx, vy) x (ax, ay)||
                # 叉乘公式：|vx*ay - vy*ax|
                next_w = abs(self.vx * ay - self.vy * ax)
                next_w = np.clip(next_w, -1.5, 1.5) 

                # --- 预测轨迹 ---
                # 使用公式 s(s0, ax, ay, t) 生成轨迹点
                traj = self._predict_trajectory(curr_x, curr_y, self.vx, self.vy, ax, ay)
                all_trajectories.append(traj)
                
                # --- 计算代价 ---
                cost = self._calc_cost(traj, goal, obstacles, next_v)
                
                if cost < min_cost:
                    min_cost = cost
                    best_traj = traj
                    best_vw = [next_v, next_w]
                    best_accel = [ax, ay]

        # 更新下一帧所需的初始速度状态
        self.v = best_vw[0]
        self.vx += best_accel[0] * self.dt
        self.vy += best_accel[1] * self.dt
        
        dt_seg = np.full(len(best_traj)-1, self.dt)
        return best_traj, dt_seg, all_trajectories, best_vw

    def _predict_trajectory(self, x0, y0, vx0, vy0, ax, ay):
        """使用公式 s(s0, ax, ay, t) 生成预测轨迹"""
        t_steps = np.arange(0, self.predict_time + self.dt, self.dt)
        traj = np.zeros((len(t_steps), 3))
        
        for i, t in enumerate(t_steps):
            px = x0 + vx0 * t + 0.5 * ax * (t**2)
            py = y0 + vy0 * t + 0.5 * ay * (t**2)
            # 预测朝向
            vxt = vx0 + ax * t
            vyt = vy0 + ay * t
            traj[i] = [px, py, np.arctan2(vyt, vxt)]
            
        return traj

    def _calc_cost(self, traj, goal, obstacles, v):
        # 目标朝向代价
        goal_theta = np.arctan2(goal[1] - traj[-1, 1], goal[0] - traj[-1, 0])
        cost_heading = abs(np.arctan2(np.sin(goal_theta - traj[-1, 2]), np.cos(goal_theta - traj[-1, 2])))
        
        # 障碍物代价
        min_dist = float('inf')
        if obstacles is not None:
            for obs in obstacles:
                # 假设 obs 结构为 [cx, cy, ...]
                d = np.linalg.norm(traj[:, :2] - np.array([obs[0], obs[1]]), axis=1).min()
                min_dist = min(min_dist, d)
        
        if min_dist < self.safe_distance: return float('inf')
        cost_dist = 1.0 / min_dist if min_dist != float('inf') else 0.0
        
        # 速度代价
        cost_vel = self.v_max - v
        
        return self.w_heading * cost_heading + self.w_dist * cost_dist + self.w_vel * cost_vel