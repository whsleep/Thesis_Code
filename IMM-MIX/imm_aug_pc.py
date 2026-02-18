import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints

from matplotlib.gridspec import GridSpec
# ========================================================
# 1. 极坐标运动模型定义 (5维: [x, y, theta, v, w])
# ========================================================

def fx_cv_pc(x, dt):
    """CV-PC: 极坐标匀速"""
    res = np.copy(x)
    res[0] = x[0] + x[3] * dt * np.cos(x[2])
    res[1] = x[1] + x[3] * dt * np.sin(x[2])
    return res

def fx_ct_pc(x, dt):
    """CT-PC: 极坐标协同转弯 (精确弦长公式)"""
    theta, v, w = x[2], x[3], x[4]
    if abs(w) < 1e-6: return fx_cv_pc(x, dt)
    
    res = np.copy(x)
    half_w_dt = 0.5 * w * dt
    chord = (2.0 * v / w) * np.sin(half_w_dt)
    
    res[0] = x[0] + chord * np.cos(theta + half_w_dt)
    res[1] = x[1] + chord * np.sin(theta + half_w_dt)
    res[2] = theta + w * dt
    return res

def hx(x):
    return x[:2]

# ========================================================
# 2. 跟踪器实现 (Q 的计算参考 7D 结构)
# ========================================================

class IMMAugPCTracker:
    def __init__(self, dt=0.1):
        self.dt = dt
        self.mu = np.array([0.5, 0.5]) 
        self.trans_prob = np.array([[0.95, 0.05], 
                                    [0.05, 0.95]])
        
        # 极坐标噪声超参数
        self.q_v_dot = 0.2  # 切向加速度噪声 (作用于 v)
        self.q_w_dot = 0.1  # 角加速度噪声 (作用于 w)
        
        self.filters = [
            self._build_ukf(fx_cv_pc, mode=0),
            self._build_ukf(fx_ct_pc, mode=1)
        ]

    def _calc_Q(self, mode, theta):
        """
        根据 Q = G @ Sigma @ G.T 计算 5x5 过程噪声矩阵
        状态: [x, y, theta, v, w]
        噪声源: [q_v_dot, q_w_dot]
        """
        T = self.dt
        c, s = np.cos(theta), np.sin(theta)
        
        # 1. 噪声强度 Sigma
        if mode == 0: # CV-PC
            # CV 模式下角速度变化噪声极小 , 噪声雅可比 G (将切向和角向噪声映射到全局坐标)
            Sigma = np.diag([self.q_v_dot**2, 0.001**2])
            G = np.array([
                [(T**2/2)*c,  0],       # x 受速度变化投影影响
                [(T**2/2)*s,  0],       # y 受速度变化投影影响
                [0,           T**2/2],  # theta 受角加速度影响
                [T,           0],       # v
                [0,           0]        # w
            ])
        else: # CT-PC
            Sigma = np.diag([self.q_v_dot**2, self.q_w_dot**2])
            G = np.array([
                [(T**2/2)*c,  0],       # x 受速度变化投影影响
                [(T**2/2)*s,  0],       # y 受速度变化投影影响
                [0,           T**2/2],  # theta 受角加速度影响
                [T,           0],       # v
                [0,           T]        # w
            ])

        return G @ Sigma @ G.T + np.eye(5) * 1e-9

    def _build_ukf(self, fx_func, mode):
        points = MerweScaledSigmaPoints(n=5, alpha=0.1, beta=2., kappa=0)
        ukf = UKF(dim_x=5, dim_z=2, fx=fx_func, hx=hx, dt=self.dt, points=points)
        ukf.P = np.eye(5) * 0.2
        ukf.R = np.eye(2) * 0.2
        ukf.Q = self._calc_Q(mode, 0.0)
        return ukf

    def step(self, z):
        # 1. 状态扩增 (Augmentation)
        self.filters[0].x[4] = self.filters[1].x[4]
        self.filters[0].P[4,4] = self.filters[1].P[4,4]

        # 2. 交互混合 (Mixing)
        c_bar = self.trans_prob.T @ self.mu
        omega = (self.trans_prob * self.mu[:, np.newaxis]) / c_bar
        
        mixed_x, mixed_P = [], []
        for j in range(2):
            xj = np.dot(omega[:, j], [f.x for f in self.filters])
            Pj = np.zeros((5, 5))
            for i in range(2):
                diff = (self.filters[i].x - xj).reshape(-1, 1)
                Pj += omega[i, j] * (self.filters[i].P + diff @ diff.T)
            mixed_x.append(xj)
            mixed_P.append(Pj)

        # 3. 预测与更新
        likelihoods = []
        for i in range(2):
            self.filters[i].x, self.filters[i].P = mixed_x[i], mixed_P[i]
            
            # 重要：动态更新 Q 以适应当前的航向 theta
            self.filters[i].Q = self._calc_Q(mode=i, theta=self.filters[i].x[2])
            
            self.filters[i].predict()
            self.filters[i].update(z)
            likelihoods.append(self.filters[i].likelihood + 1e-12)

        # 4. 模式概率更新
        self.mu = (np.array(likelihoods) * c_bar)
        self.mu /= np.sum(self.mu)

        return self.mu[0] * self.filters[0].x + self.mu[1] * self.filters[1].x

# ========================================================
# 3. 辅助预测与可视化函数
# ========================================================

def predict_future_5d_pc(tracker, steps=10):
    orig_states = [(f.x.copy(), f.P.copy()) for f in tracker.filters]
    preds = []
    for _ in range(steps):
        step_pos = np.zeros(2)
        for i, f in enumerate(tracker.filters):
            f.predict()
            step_pos += tracker.mu[i] * f.x[:2]
        preds.append(step_pos)
    for i, f in enumerate(tracker.filters):
        f.x, f.P = orig_states[i]
    return np.array(preds)

def run_test_and_visualize_5d_pc():
    dt = 0.1
    tracker = IMMAugPCTracker(dt=dt)
    
    # --- 修正后的数据生成 (使用 PC 模型生成真值) ---
    # 初始状态: [x, y, theta, v, w]
    x_true = np.array([0.0, 0.0, np.pi/4, 4.0, 0.0])
    true_traj, measurements = [], []
    
    for t in range(150):
        if 50 <= t <= 100:
            x_true[4] = 0.5  # 转弯阶段 (rad/s)
            x_true = fx_ct_pc(x_true, dt)
        else:
            x_true[4] = 0.0  # 直线阶段
            x_true = fx_cv_pc(x_true, dt)
        
        true_traj.append(x_true.copy())
        measurements.append(x_true[:2] + np.random.normal(0, 0.1, 2))
    
    true_traj = np.array(true_traj)
    measurements = np.array(measurements)

    # --- 可视化 ---
    plt.ion()
    fig = plt.figure(figsize=(15, 7))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.5, 1])
    
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])

    ax1.plot(true_traj[:, 0], true_traj[:, 1], 'k:', alpha=0.3, label="Ground Truth")
    line_est, = ax1.plot([], [], 'b-', lw=1.5, label="IMM-AUG-PC Estimate")
    line_pred, = ax1.plot([], [], 'r--', alpha=0.8, label="1s Prediction")
    dot_meas = ax1.scatter([], [], c='gray', s=10, alpha=0.4)
    ax1.set_title("5D IMM-AUG-PC Tracking (Adaptive Q)")
    ax1.legend(loc='upper left')
    ax1.axis('equal')

    model_names = ['CV-PC', 'CT-PC']
    mu_history = [[] for _ in range(2)]
    v_est_history = []
    est_traj = []

    for i in range(len(measurements)):
        z = measurements[i]
        
        if i == 0:
            for f in tracker.filters: f.x[:2] = z
        
        x_res = tracker.step(z)
        est_traj.append(x_res[:2])
        
        # 记录概率
        for j in range(2): mu_history[j].append(tracker.mu[j])
        # 在 PC 模型中，状态 x[3] 直接就是速率 v
        v_est_history.append(x_res[3])
        
        preds = predict_future_5d_pc(tracker, steps=10)

        # 更新绘图
        curr_est_np = np.array(est_traj)
        line_est.set_data(curr_est_np[:, 0], curr_est_np[:, 1])
        line_pred.set_data(preds[:, 0], preds[:, 1])
        dot_meas.set_offsets(z)
        
        ax2.clear()
        ax2.plot(mu_history[0], label=model_names[0], color='steelblue')
        ax2.plot(mu_history[1], label=model_names[1], color='darkorange')
        ax2.set_title("Mode Probabilities")
        ax2.set_ylim(-0.05, 1.05)
        ax2.legend(loc='upper right'); ax2.grid(True, ls=':')

        ax3.clear()
        ax3.plot(v_est_history, color='green', label='Estimated v (PC State)')
        ax3.axhline(4.0, color='r', linestyle='--', alpha=0.5, label='True v')
        ax3.set_title("Estimated Velocity (v)")
        ax3.set_ylim(0, 7); ax3.legend(); ax3.grid(True, ls=':')

        ax1.set_xlim(x_res[0]-15, x_res[0]+15)
        ax1.set_ylim(x_res[1]-15, x_res[1]+15)

        plt.pause(0.01)
        if not plt.fignum_exists(fig.number): break

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    run_test_and_visualize_5d_pc()