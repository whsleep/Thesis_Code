import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from matplotlib.gridspec import GridSpec
# ========================================================
# 1. 运动与观测模型 (5维状态: [x, y, dx, dy, w])
# ========================================================

def fx_cv_cc(x, dt):
    """CV-CC: 匀速运动"""
    res = np.copy(x)
    res[0] = x[0] + x[2] * dt
    res[1] = x[1] + x[3] * dt
    return res

def fx_ct_cc(x, dt):
    """CT-CC: 协同转弯"""
    w = x[4]
    if abs(w) < 1e-6: return fx_cv_cc(x, dt)
    res = np.copy(x)
    dx, dy = x[2], x[3]
    s, c = np.sin(w*dt), np.cos(w*dt)
    s_2 = np.sin(w*dt*0.5)
    res[0] = x[0] + (s_2/w)*dx - ((1-c)/w)*dy
    res[1] = x[1] + ((1-c)/w)*dx + (s_2/w)*dy
    res[2] = c*dx - s*dy
    res[3] = s*dx + c*dy
    return res

def hx(x):
    return x[:2]

# ========================================================
# 2. 跟踪器实现 (结构参考 7D Mix Tracker)
# ========================================================

class IMMAugCCTracker:
    def __init__(self, dt=0.1):
        self.dt = dt
        # 模式概率: 0: CV-CC, 1: CT-CC
        self.mu = np.array([0.5, 0.5]) 
        self.trans_prob = np.array([
            [0.95, 0.05],
            [0.05, 0.95]
        ])
        
        # 过程噪声超参数
        self.q_ax = 0.2  # x轴加速度噪声
        self.q_ay = 0.2  # y轴加速度噪声
        self.q_w  = 0.1  # 角速度演化噪声
        
        self.filters = [
            self._build_ukf(fx_cv_cc, mode=0),
            self._build_ukf(fx_ct_cc, mode=1)
        ]

    def _calc_Q(self, mode):
        """
        严格根据 Q = G @ Sigma @ G.T 计算 5x5 过程噪声矩阵
        状态顺序 (Row): [x, y, dx, dy, w]
        噪声顺序 (Col): [ax_noise, ay_noise, w_noise]
        """
        T = self.dt
        
        # 1. 构造噪声协方差 Sigma  构造噪声雅可比 G 
        if mode == 0: # CV 模式
            # CV 模式下 w 噪声极小，强制保持 w 恒定
            Sigma = np.diag([self.q_ax**2, self.q_ay**2])
            G = np.array([
                            [T**2/2, 0],  # x <- ax
                            [0,      T**2/2],  # y <- ay
                            [T,      0],  # dx <- ax
                            [0,      T],  # dy <- ay
                            [0,      0]   # w <- w_noise
                        ])
        else: # CT 模式
            Sigma = np.diag([self.q_ax**2, self.q_ay**2, self.q_w**2])
            G = np.array([
                            [T**2/2, 0, 0],  # x <- ax
                            [0,      T**2/2, 0],  # y <- ay
                            [T,      0, 0],  # dx <- ax
                            [0,      T, 0],  # dy <- ay
                            [0,      0, T]   # w <- w_noise
                        ])

        # 3. 计算 Q = G @ Sigma @ G.T
        Q = G @ Sigma @ G.T
        
        # 数值稳定性补偿
        return Q + np.eye(5) * 1e-9

    def _build_ukf(self, fx_func, mode):
        points = MerweScaledSigmaPoints(n=5, alpha=0.1, beta=2., kappa=0)
        ukf = UKF(dim_x=5, dim_z=2, fx=fx_func, hx=hx, dt=self.dt, points=points)
        ukf.P = np.eye(5) * 0.2
        ukf.R = np.eye(2) * 0.2
        # 初始化时计算 Q
        ukf.Q = self._calc_Q(mode)
        return ukf

    def step(self, z):
        # 1. 状态扩增 (Augmentation): 核心物理逻辑
        # 强制同步 CT 滤波器的 w 及其方差到 CV 滤波器
        self.filters[0].x[4] = self.filters[1].x[4]
        self.filters[0].P[4,4] = self.filters[1].P[4,4]

        # 2. 交互混合 (Mixing)
        c_bar = self.trans_prob.T @ self.mu
        omega = np.zeros((2, 2))
        for j in range(2):
            omega[:, j] = (self.trans_prob[:, j] * self.mu) / c_bar[j]

        mixed_x, mixed_P = [], []
        for j in range(2):
            xj = np.zeros(5)
            Pj = np.zeros((5, 5))
            for i in range(2):
                xj += omega[i, j] * self.filters[i].x
            for i in range(2):
                diff = (self.filters[i].x - xj).reshape(-1, 1)
                Pj += omega[i, j] * (self.filters[i].P + diff @ diff.T)
            mixed_x.append(xj)
            mixed_P.append(Pj)

        # 3. 预测与更新
        likelihoods = []
        for i in range(2):
            self.filters[i].x, self.filters[i].P = mixed_x[i], mixed_P[i]
            
            # 动态更新 Q (如果运动状态发生剧变，可在此重新计算)
            self.filters[i].Q = self._calc_Q(mode=i)
            
            self.filters[i].predict()
            self.filters[i].update(z)
            likelihoods.append(self.filters[i].likelihood + 1e-12)

        # 4. 模式概率更新
        self.mu = (np.array(likelihoods) * c_bar)
        self.mu /= np.sum(self.mu)

        # 5. 融合输出
        res_x = self.mu[0] * self.filters[0].x + self.mu[1] * self.filters[1].x
        return res_x

# ========================================================
# 1. 辅助预测函数 (用于显示 IMM 的预判能力)
# ========================================================

def predict_future_5d(tracker, steps=10):
    """备份当前状态，利用混合概率预测未来 1 秒的轨迹"""
    # 深度备份所有滤波器的状态
    orig_states = [(f.x.copy(), f.P.copy()) for f in tracker.filters]
    preds = []
    
    for _ in range(steps):
        step_pos = np.zeros(2)
        for i, f in enumerate(tracker.filters):
            f.predict()  # 仅预测
            step_pos += tracker.mu[i] * f.x[:2] # 按当前概率加权融合位置
        preds.append(step_pos)
        
    # 状态还原
    for i, f in enumerate(tracker.filters):
        f.x, f.P = orig_states[i]
    return np.array(preds)

# ========================================================
# 2. 执行测试与动态可视化
# ========================================================

def run_test_and_visualize_5d():
    dt = 0.1
    tracker = IMMAugCCTracker(dt=dt)
    
    # --- 数据生成 ---
    # 初始状态: [x, y, dx, dy, w]
    x_true = np.array([0.0, 0.0, 4.0, 1.0, 0.0])
    true_traj, measurements = [], []
    
    for t in range(150):
        if 50 <= t <= 100:
            x_true[4] = 0.6  # 转弯阶段
            x_true = fx_ct_cc(x_true, dt)
        else:
            x_true[4] = 0.0  # 直线阶段
            x_true = fx_cv_cc(x_true, dt)
        
        true_traj.append(x_true.copy())
        measurements.append(x_true[:2] + np.random.normal(0, 0.1, 2))
    
    true_traj = np.array(true_traj)
    measurements = np.array(measurements)

    # --- 可视化设置 ---
    plt.ion()
    fig = plt.figure(figsize=(15, 7))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.5, 1])
    
    ax1 = fig.add_subplot(gs[:, 0])      # 轨迹追踪
    ax2 = fig.add_subplot(gs[0, 1])     # 模式概率
    ax3 = fig.add_subplot(gs[1, 1])     # 速度估计

    # 初始化绘图句柄
    ax1.plot(true_traj[:, 0], true_traj[:, 1], 'k:', alpha=0.3, label="Ground Truth")
    line_est, = ax1.plot([], [], 'b-', lw=1.5, label="IMM Estimate")
    line_pred, = ax1.plot([], [], 'r--', alpha=0.8, label="1s Prediction")
    dot_meas = ax1.scatter([], [], c='gray', s=10, alpha=0.4)
    
    ax1.set_title("5D IMM-AUG-CC Tracking (Physical Q)")
    ax1.legend(loc='upper left')
    ax1.axis('equal')

    model_names = ['CV-CC', 'CT-CC']
    mu_history = [[] for _ in range(2)]
    v_est_history = []

    print(">>> 开始动态验证测试...")

    # --- 循环运行跟踪 ---
    for i in range(len(measurements)):
        z = measurements[i]
        
        # 跟踪器更新
        if i == 0:
            for f in tracker.filters: f.x[:2] = z
        
        x_res = tracker.step(z)
        
        # 记录数据
        for j in range(2): mu_history[j].append(tracker.mu[j])
        v_abs = np.sqrt(x_res[2]**2 + x_res[3]**2)
        v_est_history.append(v_abs)
        
        # 获取预测轨迹
        preds = predict_future_5d(tracker, steps=10)

        # 更新绘图
        est_arr = np.array(true_traj[:i+1]) # 此处为了显示平滑，实际绘图可用历史估计
        line_est.set_data(true_traj[:i, 0], true_traj[:i, 1]) # 简化处理显示
        line_pred.set_data(preds[:, 0], preds[:, 1])
        dot_meas.set_offsets(z)
        
        # 概率图更新
        ax2.clear()
        ax2.plot(mu_history[0], label=model_names[0], color='steelblue')
        ax2.plot(mu_history[1], label=model_names[1], color='darkorange')
        ax2.set_title("Mode Probabilities")
        ax2.set_ylim(-0.05, 1.05)
        ax2.legend(loc='upper right')
        ax2.grid(True, ls=':')

        # 速度图更新
        ax3.clear()
        ax3.plot(v_est_history, color='green', label='Estimated Speed')
        ax3.set_title("Estimated Velocity Magnitude")
        ax3.set_ylim(0, 7)
        ax3.legend()
        ax3.grid(True, ls=':')

        # 局部视角跟随
        ax1.set_xlim(x_res[0]-15, x_res[0]+15)
        ax1.set_ylim(x_res[1]-15, x_res[1]+15)

        plt.pause(0.01)
        if not plt.fignum_exists(fig.number): break

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    run_test_and_visualize_5d()