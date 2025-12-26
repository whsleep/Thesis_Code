import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')  # 屏蔽除以0等警告

# ===================== 核心参数配置 =====================
N_MC = 10000        # 蒙特卡罗实验次数
T = 0.1             # 状态转移时间差 (s)
omega_fixed = 0.1   # 固定角速度 (rad/s)
theta0 = np.pi/4    # 初始方位角 (rad)

# 噪声标准差（与论文一致）
sigma_theta = 0.1   # 方位角噪声 (rad)
sigma_v = 0.3       # 线速度噪声 (m/s)
sigma_omega = 0.1   # 角速度噪声 (rad/s)

# 线速度变化范围：0~100m/s（对数刻度采样，避免低速点稀疏）
v_list = np.logspace(np.log10(0.01), np.log10(100), 50)  # 从0.01开始避免纯零

# 结果存储数组（CT-CC/CT-PC 分别存储位置/方位角误差的中位数、10%、90%分位数）
# CT-CC 结果
cc_pos_med, cc_pos_10, cc_pos_90 = [], [], []
cc_theta_med, cc_theta_10, cc_theta_90 = [], [], []

# CT-PC 结果
pc_pos_med, pc_pos_10, pc_pos_90 = [], [], []
pc_theta_med, pc_theta_10, pc_theta_90 = [], [], []

# ===================== 运动模型定义 =====================
def ct_cc_transition(x_prev, omega, T):
    """
    CT-CC 状态转移函数（笛卡尔坐标系协调转弯）
    x_prev: 前一状态 [x, y, dx, dy, omega]
    omega: 角速度 (rad/s)
    T: 时间差 (s)
    返回：预测状态
    """
    x, y, dx, dy, _ = x_prev
    if np.abs(omega) < 1e-8:  # 角速度接近0时退化到CV模型
        x_new = x + dx * T
        y_new = y + dy * T
        dx_new = dx
        dy_new = dy
    else:
        # CT-CC 转移公式
        x_new = x + (dx/omega)*np.sin(omega*T/2) - (dy/omega)*(1 - np.cos(omega*T))
        y_new = y + (dy/omega)*np.sin(omega*T/2) + (dx/omega)*(1 - np.cos(omega*T))
        dx_new = dx * np.cos(omega*T) - dy * np.sin(omega*T)
        dy_new = dx * np.sin(omega*T) + dy * np.cos(omega*T)
    return np.array([x_new, y_new, dx_new, dy_new, omega])

def ct_pc_transition(x_prev, T):
    """
    CT-PC 状态转移函数（极坐标系协调转弯）
    x_prev: 前一状态 [x, y, theta, v, omega]
    T: 时间差 (s)
    返回：预测状态
    """
    x, y, theta, v, omega = x_prev
    if np.abs(omega) < 1e-8:  # 角速度接近0时退化到CV模型
        x_new = x + v * np.cos(theta) * T
        y_new = y + v * np.sin(theta) * T
        theta_new = theta
    else:
        # CT-PC 转移公式
        x_new = x + (2*v/omega) * np.sin(omega*T/2) * np.cos(theta + omega*T/2)
        y_new = y + (2*v/omega) * np.sin(omega*T/2) * np.sin(theta + omega*T/2)
        theta_new = theta + omega * T
    return np.array([x_new, y_new, theta_new, v, omega])

# ===================== 蒙特卡罗仿真 =====================
for v in v_list:
    print(f"正在计算线速度: {v:.4f} m/s")

    # -------------------- 初始状态初始化 --------------------
    # 初始位置固定为(0,0)
    # CT-CC 初始状态 [x, y, dx, dy, omega]
    dx0 = v * np.cos(theta0)
    dy0 = v * np.sin(theta0)
    x0_cc = np.array([0.0, 0.0, dx0, dy0, omega_fixed])
    
    # CT-PC 初始状态 [x, y, theta, v, omega]
    x0_pc = np.array([0.0, 0.0, theta0, v, omega_fixed])
    
    # -------- 修正1：计算无噪声的真实状态（理想预测结果） --------
    # 真实的CT-CC预测位置（无噪声）
    x_true_cc = ct_cc_transition(x0_cc, omega_fixed, T)
    true_pos_cc = (x_true_cc[0], x_true_cc[1])  # 真实位置
    true_theta_cc = np.arctan2(x_true_cc[3], x_true_cc[2])  # 真实方位角
    
    # 真实的CT-PC预测位置（无噪声）
    x_true_pc = ct_pc_transition(x0_pc, T)
    true_pos_pc = (x_true_pc[0], x_true_pc[1])  # 真实位置
    true_theta_pc = x_true_pc[2]  # 真实方位角
    
    # -------------------- 生成噪声 --------------------
    # CT-CC 噪声：dx/dy噪声（与v噪声等价） + omega噪声
    noise_dx = np.random.normal(0, sigma_v * np.cos(theta0), N_MC)
    noise_dy = np.random.normal(0, sigma_v * np.sin(theta0), N_MC)
    noise_omega_cc = np.random.normal(0, sigma_omega, N_MC)
    
    # CT-PC 噪声：v噪声 + omega噪声 + 初始theta噪声
    noise_v = np.random.normal(0, sigma_v, N_MC)
    noise_omega_pc = np.random.normal(0, sigma_omega, N_MC)
    noise_theta = np.random.normal(0, sigma_theta, N_MC)
    
    # -------------------- CT-CC 模型预测与误差计算 --------------------
    cc_pos_errors = []  # 位置误差（欧氏距离）
    cc_theta_errors = []# 方位角误差（归一化到[0,π]）
    
    for i in range(N_MC):
        # 注入噪声的初始状态
        x0_noisy = x0_cc.copy()
        x0_noisy[2] += noise_dx[i]    # dx噪声
        x0_noisy[3] += noise_dy[i]    # dy噪声
        x0_noisy[4] += noise_omega_cc[i]  # omega噪声
        
        # 状态预测
        x_pred = ct_cc_transition(x0_noisy, x0_noisy[4], T)
        
        # -------- 修正2：计算预测位置与真实位置的误差 --------
        # 位置误差：预测位置 vs 无噪声的真实位置
        pos_error = np.sqrt((x_pred[0] - true_pos_cc[0])**2 + (x_pred[1] - true_pos_cc[1])**2)
        cc_pos_errors.append(pos_error)
        
        # 方位角误差：由dx/dy计算预测方位角，与真实方位角的差
        if np.abs(x_pred[2]) < 1e-10 and np.abs(x_pred[3]) < 1e-10:
            pred_theta = true_theta_cc  # 速度为0时避免arctan2报错
        else:
            pred_theta = np.arctan2(x_pred[3], x_pred[2])
        theta_error = np.abs(pred_theta - true_theta_cc)
        theta_error = np.min([theta_error, 2*np.pi - theta_error])  # 归一化到[0,π]
        cc_theta_errors.append(theta_error)
    
    # -------------------- CT-PC 模型预测与误差计算 --------------------
    pc_pos_errors = []
    pc_theta_errors = []
    
    for i in range(N_MC):
        # 注入噪声的初始状态
        x0_noisy = x0_pc.copy()
        x0_noisy[2] += noise_theta[i]   # theta噪声
        x0_noisy[3] += noise_v[i]       # v噪声
        x0_noisy[4] += noise_omega_pc[i]# omega噪声
        
        # 状态预测
        x_pred = ct_pc_transition(x0_noisy, T)
        
        # -------- 修正2：计算预测位置与真实位置的误差 --------
        # 位置误差：预测位置 vs 无噪声的真实位置
        pos_error = np.sqrt((x_pred[0] - true_pos_pc[0])**2 + (x_pred[1] - true_pos_pc[1])**2)
        pc_pos_errors.append(pos_error)
        
        # 方位角误差：预测方位角 vs 无噪声的真实方位角
        pred_theta = x_pred[2]
        theta_error = np.abs(pred_theta - true_theta_pc)
        theta_error = np.min([theta_error, 2*np.pi - theta_error])
        pc_theta_errors.append(theta_error)
    
    # -------------------- 误差分位数统计 --------------------
    # CT-CC 统计
    cc_pos_med.append(np.median(cc_pos_errors))
    cc_pos_10.append(np.percentile(cc_pos_errors, 10))
    cc_pos_90.append(np.percentile(cc_pos_errors, 90))
    cc_theta_med.append(np.median(cc_theta_errors))
    cc_theta_10.append(np.percentile(cc_theta_errors, 10))
    cc_theta_90.append(np.percentile(cc_theta_errors, 90))
    
    # CT-PC 统计
    pc_pos_med.append(np.median(pc_pos_errors))
    pc_pos_10.append(np.percentile(pc_pos_errors, 10))
    pc_pos_90.append(np.percentile(pc_pos_errors, 90))
    pc_theta_med.append(np.median(pc_theta_errors))
    pc_theta_10.append(np.percentile(pc_theta_errors, 10))
    pc_theta_90.append(np.percentile(pc_theta_errors, 90))

# ===================== 结果可视化 =====================
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'Times New Roman'
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# -------------------- 位置误差图（对数刻度） --------------------
ax1.set_xscale('log')
ax1.set_yscale('log')
# CT-CC 位置误差
ax1.plot(v_list, cc_pos_med, 'b-', label='CT-CC (Median)', linewidth=2)
ax1.fill_between(v_list, cc_pos_10, cc_pos_90, color='b', alpha=0.2, label='CT-CC (10%-90%)')
# CT-PC 位置误差
ax1.plot(v_list, pc_pos_med, 'r-', label='CT-PC (Median)', linewidth=2)
ax1.fill_between(v_list, pc_pos_10, pc_pos_90, color='r', alpha=0.2, label='CT-PC (10%-90%)')

ax1.set_xlabel('Linear Velocity (m/s) [log]')
ax1.set_ylabel('Position Error (m) [log]')
ax1.set_title(f'Position Error(ω={omega_fixed} rad/s)')
ax1.legend(loc='upper left')
ax1.grid(True, which="both", ls="--")

# -------------------- 方位角误差图（对数刻度） --------------------
ax2.set_xscale('log')
ax2.set_yscale('log')
# CT-CC 方位角误差
ax2.plot(v_list, cc_theta_med, 'b-', label='CT-CC (Median)', linewidth=2)
ax2.fill_between(v_list, cc_theta_10, cc_theta_90, color='b', alpha=0.2, label='CT-CC (10%-90%)')
# CT-PC 方位角误差
ax2.plot(v_list, pc_theta_med, 'r-', label='CT-PC (Median)', linewidth=2)
ax2.fill_between(v_list, pc_theta_10, pc_theta_90, color='r', alpha=0.2, label='CT-PC (10%-90%)')

ax2.set_xlabel('Linear Velocity (m/s) [log]')
ax2.set_ylabel('Angular Error (rad) [log]')
ax2.set_title(f'Angular Error(ω={omega_fixed} rad/s)')
ax2.legend(loc='upper right')
ax2.grid(True, which="both", ls="--")

plt.tight_layout()
plt.show()
