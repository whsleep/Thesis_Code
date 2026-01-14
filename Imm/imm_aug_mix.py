import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from scipy.linalg import block_diag

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
# ========================================================
# 1. 基于 7 维扩增状态的转移函数
# x = [x, y, theta, v, dx, dy, w]
# ========================================================

def fx_cv_cc_7d(x, T):
    """CV-CC (笛卡尔匀速)"""
    res = np.copy(x)
    res[0] = x[0] + x[4]*T
    res[1] = x[1] + x[5]*T
    res[2] = np.arctan2(x[5], x[4])
    res[3] = np.sqrt(x[4]**2 + x[5]**2)
    # x[4], x[5], x[6] 保持不变 (dx, dy, 0)
    res[6] = 0
    return res

def fx_cv_pc_7d(x, T):
    """CV-PC (极坐标匀速)"""
    res = np.copy(x)
    res[0] = x[0] + x[3]*T*np.cos(x[2])
    res[1] = x[1] + x[3]*T*np.sin(x[2])
    # x[2], x[3] 保持不变 (theta, v)
    res[4] = x[3]*np.cos(x[2])
    res[5] = x[3]*np.sin(x[2])
    res[6] = 0
    return res

def fx_ct_cc_7d(x, T):
    """CT-CC (笛卡尔协同转弯)"""
    res = np.copy(x)
    w = x[6]
    dx, dy = x[4], x[5]
    if abs(w) < 1e-6: return fx_cv_cc_7d(x, T)
    
    s, c = np.sin(w*T), np.cos(w*T)
    res[0] = x[0] + (s/w)*dx - ((1-c)/w)*dy
    res[1] = x[1] + ((1-c)/w)*dx + (s/w)*dy
    res[2] = np.arctan2(dx*s + dy*c, dx*c - dy*s) # 基于旋转后的速度算theta
    res[3] = np.sqrt(dx**2 + dy**2)
    res[4] = dx*c - dy*s
    res[5] = dx*s + dy*c
    return res

def fx_ct_pc_7d(x, T):
    """CT-PC (极坐标协同转弯)"""
    res = np.copy(x)
    theta, v, w = x[2], x[3], x[6]
    if abs(w) < 1e-6: return fx_cv_pc_7d(x, T)
    
    half_wT = 0.5 * w * T
    chord = (2.0 * v / w) * np.sin(half_wT)
    res[0] = x[0] + chord * np.cos(theta + half_wT)
    res[1] = x[1] + chord * np.sin(theta + half_wT)
    res[2] = theta + w * T
    res[3] = v
    res[4] = v * np.cos(res[2])
    res[5] = v * np.sin(res[2])
    return res

def hx_7d(x):
    """观测位置 x, y"""
    return x[:2]

# ========================================================
# 2. 核心 Tracker 类
# ========================================================

class IMMAugMixTracker7D:
    def __init__(self, dt=0.1):
        self.dt = dt
        # cv_cc, cv_pc, ct_cc, ct_pc
        self.mu = np.array([0.25, 0.25, 0.25, 0.25])
        self.trans_prob = np.array([
            [0.97, 0.01, 0.01, 0.01], # 从 CV-CC 保持或切换
            [0.01, 0.97, 0.01, 0.01], # 从 CV-PC 保持或切换
            [0.01, 0.01, 0.97, 0.01], # 从 CT-CC 保持或切换
            [0.01, 0.01, 0.01, 0.97]  # 从 CT-PC 保持或切换
        ])

        # 混合噪声参数设置 
        self.q_v, self.q_w = 0.2, 0.1      # 极坐标相关的加速度/角速度噪声
        self.q_ax, self.q_ay = 0.2, 0.1      # 笛卡尔相关的加速度噪声

        self.filters = [
            self._build_ukf(fx_cv_cc_7d, 0),
            self._build_ukf(fx_cv_pc_7d, 1),
            self._build_ukf(fx_ct_cc_7d, 2),
            self._build_ukf(fx_ct_pc_7d, 3)
        ]

    def _calc_Q(self, mode, theta=0.0):
            """
            严格根据 Q = G @ Sigma @ G.T 计算 7x7 过程噪声矩阵
            状态向量顺序: [x, y, theta, v, dx, dy, w]
            """
            T = self.dt
            
            # 1. 严格按照你给出的顺序构造 Sigma
            if mode in [0, 1]: # CV
                # [sigma_v, sigma_ax, sigma_ay, sigma_w]
                Sigma = np.diag([self.q_v**2, self.q_ax**2, self.q_ay**2, self.q_w**2])
            else: # CT
                # [sigma_v, sigma_ax, sigma_ay, sigma_w, sigma_omega_dot]
                Sigma = np.diag([self.q_v**2, self.q_ax**2, self.q_ay**2, self.q_w**2, 0.02**2])

            # 2. 根据模式构造噪声雅可比 G
            if mode == 0: # CV-CC
                # 噪声源: [v, dx, dy, w] -> 状态增量
                G = np.array([
                    [0, T**2/2, 0, 0],   # x: 受 ax 影响
                    [0, 0, T**2/2, 0],   # y: 受 ay 影响
                    [0, 0, 0, T**2/2],   # theta: 受 w 影响
                    [T, 0, 0, 0],         # v: 受 v 影响
                    [0, T, 0, 0],         # dx: 受 ax 影响
                    [0, 0, T, 0],         # dy: 受 ay 影响
                    [0, 0, 0, 0]          # w: 恒定
                ])
                
            elif mode == 1: # CV-PC
                # 噪声源: [v, dx, dy, w] -> 状态增量
                c, s = np.cos(theta), np.sin(theta)
                G = np.array([
                    [T**2/2*c, 0, 0, 0], # x: 受加速度在 x 方向投影影响
                    [T**2/2*s, 0, 0, 0], # y: 受加速度在 y 方向投影影响
                    [0, 0, 0, T**2/2],   # theta
                    [T, 0, 0, 0],         # v
                    [T*c, 0, 0, 0],         # dx
                    [T*s, 0, 0, 0],         # dy
                    [0, 0, 0, 0]          # w
                ])
                
            elif mode == 2: # CT-CC
                # 噪声源: [v, dx, dy, w, dot{w}] -> 状态增量
                G = np.array([
                    [0, T**2/2, 0, 0, 0],
                    [0, 0, T**2/2, 0, 0],
                    [0, 0, 0, T, 0],
                    [T, 0, 0, 0, 0],
                    [0, T, 0, 0, 0],
                    [0, 0, T, 0, 0],
                    [0, 0, 0, 0, T]      # w 受 sigma_omega 影响
                ])
                
            elif mode == 3: # CT-PC
                # 噪声源: [v, dx, dy, w, dot{w}] -> 状态增量
                c, s = np.cos(theta), np.sin(theta)
                G = np.array([
                    [(T**2/2)*c, 0, 0, 0, 0],      # x
                    [(T**2/2)*s, 0, 0, 0, 0],      # y
                    [0,          0, 0, T, 0],      # theta
                    [T,          0, 0, 0, 0],      # v
                    [T*c,        0, 0, 0, 0],      # dx
                    [T*s,        0, 0, 0, 0],      # dy
                    [0,          0, 0, 0, T]       # w
                ])


            # 3. 计算最终的 7x7 Q 矩阵
            Q = G @ Sigma @ G.T
            
            # 4. 数值稳定性补偿 (防止 Q 奇异导致 UKF 崩溃)
            return Q + np.eye(7) * 1e-9

    def _build_ukf(self, fx, mode):
        sigmas = MerweScaledSigmaPoints(n=7, alpha=0.1, beta=2., kappa=0)
        ukf = UKF(dim_x=7, dim_z=2, fx=fx, hx=hx_7d, dt=self.dt, points=sigmas)
        ukf.P = np.eye(7) * 0.2
        ukf.R = np.eye(2) * 0.2
        ukf.Q = self._calc_Q(mode=mode, theta=0.0)
        return ukf

    def step(self, z):
            # 定义模型索引 (假设: 0,1 是 CV-CC/PC; 2,3 是 CT-CC/PC)
            idx_cv = [0, 1]
            idx_ct = [2, 3]

            # 1. 计算混合概率 (Mixing Probabilities)
            c_bar = self.trans_prob.T @ self.mu
            omega_mix = np.zeros((4, 4))
            for j in range(4):
                omega_mix[:, j] = (self.trans_prob[:, j] * self.mu) / (c_bar[j] + 1e-9)

            # 2. 异构混合过程 (Heterogeneous Mixing)
            mixed_x, mixed_P = [], []
            
            for j in range(4):
                # --- 步骤 2.1: 为当前目标模型 j 准备混合状态 ---
                xj_sum = np.zeros(7)
                
                # 如果目标模型 j 是 CV 模型，我们需要先“借用”CT 模型的角速度估计来补全状态
                if j in idx_cv:
                    # 计算来自 CT 模型的角速度加权均值 (对应你提供的公式 1)
                    sum_mu_ct = np.sum([omega_mix[k, j] for k in idx_ct]) + 1e-12
                    w_from_ct = np.sum([omega_mix[k, j] * self.filters[k].x[6] for k in idx_ct]) / sum_mu_ct
                    
                    # 遍历所有来源模型 i 进行混合
                    for i in range(4):
                        xi_temp = np.copy(self.filters[i].x)
                        if i in idx_cv:
                            # 来源也是 CV，强制注入刚刚算出的加权角速度 w
                            xi_temp[6] = w_from_ct
                        xj_sum += omega_mix[i, j] * xi_temp
                else:
                    # 如果目标模型 j 本身就是 CT 模型，直接按 omega 加权 (对应公式 2)
                    for i in range(4):
                        xj_sum += omega_mix[i, j] * self.filters[i].x

                # --- 步骤 2.2: 计算混合协方差 ---
                Pj_sum = np.zeros((7, 7))
                for i in range(4):
                    # 构造补全后的来源状态协方差 Pi_temp
                    Pi_temp = np.copy(self.filters[i].P)
                    if i in idx_cv:
                        # 对于 CV 模型，角速度项是不确定的，借用 CT 的角速度协方差 (对应公式 3)
                        sum_mu_ct = np.sum([omega_mix[k, j] for k in idx_ct]) + 1e-12
                        Pww_from_ct = np.sum([omega_mix[k, j] * self.filters[k].P[6, 6] for k in idx_ct]) / sum_mu_ct
                        Pi_temp[6, 6] = Pww_from_ct
                    
                    diff = (self.filters[i].x - xj_sum).reshape(-1, 1)
                    Pj_sum += omega_mix[i, j] * (Pi_temp + diff @ diff.T)
                
                mixed_x.append(xj_sum)
                mixed_P.append(Pj_sum)

            # 3. 预测与更新 (赋值混合后的结果)
            likelihoods = []
            for i in range(4):
                self.filters[i].x, self.filters[i].P = mixed_x[i], mixed_P[i]
                
                # 自适应计算过程噪声 Q (根据当前 theta)
                current_theta = self.filters[i].x[2] 
                self.filters[i].Q = self._calc_Q(mode=i, theta=current_theta)
                
                self.filters[i].predict()
                self.filters[i].update(z)
                likelihoods.append(self.filters[i].likelihood + 1e-12)

            # 4. 模型概率更新
            self.mu = (np.array(likelihoods) * c_bar)
            self.mu /= np.sum(self.mu)

            # 5. 融合输出最终状态
            res_x = np.zeros(7)
            for i in range(4):
                res_x += self.mu[i] * self.filters[i].x
                
            return res_x

def generate_soft_switch_case(dt=0.1):
    """
    生成一个复杂的运动轨迹，用于测试模型间的软切换。
    状态：[x, y, theta, v, dx, dy, w]
    """
    # 初始状态：位置(0,0)，朝向0度(东)，速度5m/s
    s = np.array([0.0, 0.0, 0.0, 5.0, 5.0, 0.0, 0.0])
    traj = []
    
    # 段落1: 匀速直线 (3秒) - 验证 CV-CC/CV-PC 的稳定性
    for _ in range(int(1.0/dt)):
        s[0] += s[4]*dt; s[1] += s[5]*dt
        traj.append(s.copy())
        
    # 段落2: 协同转弯 (4秒, w=0.4rad/s) - 验证向 CT 模型的软切换
    w = 1.2
    s[6] = w
    for _ in range(int(10.0/dt)):
        theta_old = s[2]
        s[2] += w*dt
        s[0] += s[3]*dt*np.cos(theta_old + 0.5*w*dt)
        s[1] += s[3]*dt*np.sin(theta_old + 0.5*w*dt)
        s[4] = s[3]*np.cos(s[2])
        s[5] = s[3]*np.sin(s[2])
        traj.append(s.copy())
        
    # 段落3: 突然减速并反向转弯 (3秒, w=-0.6) - 验证极坐标模型的鲁棒性
    s[6] = -1.0
    for _ in range(int(10.0/dt)):
        s[3] = max(1.0, s[3] - 1.0*dt) # 减速
        theta_old = s[2]
        s[2] += s[6]*dt
        s[0] += s[3]*dt*np.cos(theta_old + 0.5*s[6]*dt)
        s[1] += s[3]*dt*np.sin(theta_old + 0.5*s[6]*dt)
        s[4] = s[3]*np.cos(s[2])
        s[5] = s[3]*np.sin(s[2])
        traj.append(s.copy())

    return np.array(traj)

# ========================================================
# 2. 预测函数 (用于显示软切换时的预判能力)
# ========================================================

def predict_future_7d(tracker, steps=10):
    """备份状态并利用当前混合概率预测未来轨迹"""
    orig_states = [(f.x.copy(), f.P.copy()) for f in tracker.filters]
    preds = []
    for _ in range(steps):
        step_pos = np.zeros(2)
        for i, f in enumerate(tracker.filters):
            f.predict()
            step_pos += tracker.mu[i] * f.x[:2]
        preds.append(step_pos)
    # 还原
    for i, f in enumerate(tracker.filters):
        f.x, f.P = orig_states[i]
    return np.array(preds)

# ========================================================
# 3. 执行测试与可视化
# ========================================================

def run_test_and_visualize():
    dt = 0.1
    tracker = IMMAugMixTracker7D(dt=dt)
    
    # 1. 生成数据
    true_traj = generate_soft_switch_case(dt)
    # 添加适量噪声以观察滤波效果，noise=0 则退化为验证动力学一致性
    measurements = true_traj[:, :2] + np.random.normal(0, 0.0, (len(true_traj), 2))
    
    # 初始化
    init_z = measurements[0]
    for f in tracker.filters:
        f.x = np.array([init_z[0], init_z[1], 0.0, 5.0, 5.0, 0.0, 0.0])

    # 2. 设置复杂的画布布局
    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(2, 2, width_ratios=[1.5, 1], height_ratios=[1, 1], figure=fig)
    
    ax_map = fig.add_subplot(gs[:, 0])      # 左侧全图：轨迹与预测
    ax_prob = fig.add_subplot(gs[0, 1])     # 右上：模式概率
    ax_error = fig.add_subplot(gs[1, 1])    # 右下：实时估计误差

    # 3. 初始化绘图元素
    ax_map.plot(true_traj[:, 0], true_traj[:, 1], 'k:', alpha=0.3, label="Ideal Path")
    est_path, = ax_map.plot([], [], 'b-', lw=1.5, label="IMM Estimate")
    pred_path, = ax_map.plot([], [], 'r--', alpha=0.7, label="1s Horizon Prediction")
    curr_point = ax_map.scatter([], [], c='red', s=50, zorder=5)
    
    ax_map.set_title("IMM-AugMix: Trajectory & Prediction", fontsize=12)
    ax_map.legend(loc='upper left')
    ax_map.grid(True, alpha=0.3)
    ax_map.set_aspect('equal')

    model_names = ['CV-CC', 'CV-PC', 'CT-CC', 'CT-PC']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    prob_lines = [ax_prob.plot([], [], color=colors[i], label=model_names[i])[0] for i in range(4)]
    
    ax_prob.set_title("Mode Probability Dynamic", fontsize=12)
    ax_prob.set_ylim(-0.02, 1.02)
    ax_prob.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax_prob.grid(True, ls='--')

    error_line, = ax_error.plot([], [], 'm-', label="Euclidean Error")
    ax_error.set_title("Estimation Error (m)", fontsize=12)
    ax_error.set_ylim(0, 1.0)
    ax_error.grid(True, ls='--')

    # 4. 数据记录
    history_est = []
    history_mu = [[] for _ in range(4)]
    history_err = []

    plt.ion()
    print(">>> 开始软切换压力测试...")

    for i in range(len(measurements)):
        z = measurements[i]
        
        # 核心更新步
        x_state = tracker.step(z)
        
        # 记录数据
        history_est.append(x_state[:2])
        for j in range(4):
            history_mu[j].append(tracker.mu[j])
        
        err = np.linalg.norm(x_state[:2] - true_traj[i, :2])
        history_err.append(err)

        # 预测未来轨迹
        preds = predict_future_7d(tracker, steps=10)
        
        # 5. 增量更新 UI (避免使用 ax.clear 以提高帧率)
        est_arr = np.array(history_est)
        est_path.set_data(est_arr[:, 0], est_arr[:, 1])
        pred_path.set_data(preds[:, 0], preds[:, 1])
        curr_point.set_offsets(x_state[:2])

        # 概率图更新
        times = np.arange(len(history_mu[0])) * dt
        for j in range(4):
            prob_lines[j].set_data(times, history_mu[j])
        ax_prob.set_xlim(0, max(5, times[-1]))

        # 误差图更新
        error_line.set_data(times, history_err)
        ax_error.set_xlim(0, max(5, times[-1]))

        # 视角局部跟随 (窗口平移)
        ax_map.set_xlim(x_state[0]-8, x_state[0]+8)
        ax_map.set_ylim(x_state[1]-8, x_state[1]+8)

        if i % 2 == 0:  # 隔帧刷新降低绘图开销
            plt.pause(0.001)
        
        if not plt.fignum_exists(fig.number):
            break

    plt.ioff()
    plt.show()
    print(">>> 测试结束。")

if __name__ == "__main__":
    run_test_and_visualize()