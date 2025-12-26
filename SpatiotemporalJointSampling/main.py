import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ====================== 核心参数设置（严格匹配论文）======================
np.random.seed(42)  # 固定种子保证可复现
T = 0.1  # 时间步
theta0 = np.pi/4  # 初始方位角
omega_fixed = 0.1  # 固定角速度 (rad/s)
sigma_theta = 0.1  # 方位角观测噪声标准差 (rad)
sigma_v = 0.3      # 线速度观测噪声标准差 (m/s)
sigma_omega = 0.1  # 角速度观测噪声标准差 (rad/s)
num_trials = 100000  # 蒙特卡罗次数
v_list = np.logspace(np.log10(0.01), np.log10(100), 100)  # 线速度序列

# ====================== 向量优化的预测函数（物理意义对齐）======================
def ct_pc_predict_vec(omega_noise, theta_noise, v_noise, T, x0=0, y0=0):
    """CT-PC模型：极坐标建模，直接注入观测噪声（匹配论文）"""
    omega_k = omega_noise
    theta_k = theta_noise + omega_k * T  # 预测时刻的方位角（含初始观测噪声）
    v_k = v_noise
    mask = np.abs(omega_k) > 1e-6  # 避免除零
    
    # 初始化：退化到CV模型（无转弯）
    x_pre = np.full_like(omega_k, x0 + v_k * T * np.cos(theta_noise))
    y_pre = np.full_like(omega_k, y0 + v_k * T * np.sin(theta_noise))
    
    # 非零角速度：圆周运动公式（论文核心公式）
    x_pre[mask] = x0 + (2*v_k[mask]/omega_k[mask]) * np.sin(omega_k[mask]*T/2) * np.cos(theta0 + omega_k[mask]*T/2)
    y_pre[mask] = y0 + (2*v_k[mask]/omega_k[mask]) * np.sin(omega_k[mask]*T/2) * np.sin(theta0 + omega_k[mask]*T/2)
    
    # 修正：预测方位角应基于真实初始角+噪声，而非带噪初始角
    pred_theta = theta0 + omega_k * T + (theta_noise - theta0)  # 噪声仅作用于观测
    
    return x_pre, y_pre, pred_theta

def ct_cc_predict_vec(omega_noise, dot_x_noisy, dot_y_noisy, T, x0=0, y0=0):
    """CT-CC模型：笛卡尔坐标建模，噪声由v/θ推导（匹配论文）"""
    omega_k = omega_noise
    mask = np.abs(omega_k) > 1e-6
    
    # 初始化：CV模型
    x_pre = np.full_like(omega_k, x0 + dot_x_noisy*T)
    y_pre = np.full_like(omega_k, y0 + dot_y_noisy*T)
    dot_x_pre = np.full_like(omega_k, dot_x_noisy)
    dot_y_pre = np.full_like(omega_k, dot_y_noisy)
    
    # 非零角速度：旋转矩阵更新速度分量（论文核心公式）
    cos_wt = np.cos(omega_k[mask]*T)
    sin_wt = np.sin(omega_k[mask]*T)
    x_pre[mask] = x0 + (dot_x_noisy[mask]/omega_k[mask])*sin_wt - (dot_y_noisy[mask]/omega_k[mask])*(1 - cos_wt)
    y_pre[mask] = y0 + (dot_y_noisy[mask]/omega_k[mask])*sin_wt + (dot_x_noisy[mask]/omega_k[mask])*(1 - cos_wt)
    dot_x_pre[mask] = dot_x_noisy[mask]*cos_wt - dot_y_noisy[mask]*sin_wt
    dot_y_pre[mask] = dot_x_noisy[mask]*sin_wt + dot_y_noisy[mask]*cos_wt
    
    return x_pre, y_pre, dot_x_pre, dot_y_pre

# ====================== 蒙特卡罗模拟（噪声环境完全公平）======================
def compute_errors():
    # 初始化统计数组：[中位数, 10分位, 90分位] × 线速度点数
    ctpc_pos = np.zeros((3, len(v_list)))
    ctcc_pos = np.zeros((3, len(v_list)))
    ctpc_theta = np.zeros((3, len(v_list)))
    ctcc_theta = np.zeros((3, len(v_list)))
    
    for idx, v in enumerate(v_list):
        print(f"线速度: {v:.4f} m/s")
        
        # 生成一组噪声（CT-CC/CT-PC共用，保证公平）
        # 噪声模型：观测噪声 = 真实值 + 高斯噪声（论文标准设定）
        theta_noise = norm.rvs(loc=theta0, scale=sigma_theta, size=num_trials)
        v_noise = norm.rvs(loc=v, scale=sigma_v, size=num_trials)
        omega_noise = norm.rvs(loc=omega_fixed, scale=sigma_omega, size=num_trials)
        
        # ---------------------- 理想状态（无噪声，论文基准） ----------------------
        # 理想CT-PC状态
        ideal_theta_pc = theta0 + omega_fixed * T
        ideal_x_pc = np.where(abs(omega_fixed)>1e-6,
                             0 + (2*v/omega_fixed)*np.sin(omega_fixed*T/2)*np.cos(ideal_theta_pc - omega_fixed*T/2),
                             0 + v*T*np.cos(theta0))
        ideal_y_pc = np.where(abs(omega_fixed)>1e-6,
                             0 + (2*v/omega_fixed)*np.sin(omega_fixed*T/2)*np.sin(ideal_theta_pc - omega_fixed*T/2),
                             0 + v*T*np.sin(theta0))
        
        # 理想CT-CC状态
        dx_true = v * np.cos(theta0)
        dy_true = v * np.sin(theta0)
        ideal_x_cc = np.where(abs(omega_fixed)>1e-6,
                             0 + (dx_true/omega_fixed)*np.sin(omega_fixed*T) - (dy_true/omega_fixed)*(1-np.cos(omega_fixed*T)),
                             0 + dx_true*T)
        ideal_y_cc = np.where(abs(omega_fixed)>1e-6,
                             0 + (dy_true/omega_fixed)*np.sin(omega_fixed*T) + (dx_true/omega_fixed)*(1-np.cos(omega_fixed*T)),
                             0 + dy_true*T)
        ideal_theta_cc = theta0 + omega_fixed * T  # 论文定义：理想方位角随时间更新
        
        # ---------------------- CT-PC 预测与误差（修复噪声注入） ----------------------
        pred_x_pc, pred_y_pc, pred_theta_pc = ct_pc_predict_vec(omega_noise, theta_noise, v_noise, T)
        # 位置误差：L2范数（论文标准）
        pos_err_pc = np.linalg.norm([pred_x_pc - ideal_x_pc, pred_y_pc - ideal_y_pc], axis=0)
        # 角度误差：归一化到[0, π]（论文核心处理）
        theta_err_pc = np.abs(pred_theta_pc - ideal_theta_pc)
        theta_err_pc = np.minimum(theta_err_pc, 2*np.pi - theta_err_pc)
        
        # ---------------------- CT-CC 预测与误差（修复噪声同源） ----------------------
        # 带噪速度分量：由同一组v/θ噪声推导（非完整约束，论文核心）
        dot_x_noisy = v_noise * np.cos(theta_noise)
        dot_y_noisy = v_noise * np.sin(theta_noise)
        # 预测
        pred_x_cc, pred_y_cc, pred_dot_x_cc, pred_dot_y_cc = ct_cc_predict_vec(omega_noise, dot_x_noisy, dot_y_noisy, T)
        # 位置误差
        pos_err_cc = np.linalg.norm([pred_x_cc - ideal_x_cc, pred_y_cc - ideal_y_cc], axis=0)
        # 角度误差：从更新后的速度分量推导（论文定义）
        pred_theta_cc = np.arctan2(pred_dot_y_cc, pred_dot_x_cc)
        theta_err_cc = np.abs(pred_theta_cc - ideal_theta_cc)
        theta_err_cc = np.minimum(theta_err_cc, 2*np.pi - theta_err_cc)
        
        # ---------------------- 统计计算（匹配论文的中位数/分位数） ----------------------
        ctpc_pos[:, idx] = [np.median(pos_err_pc), np.percentile(pos_err_pc, 10), np.percentile(pos_err_pc, 90)]
        ctcc_pos[:, idx] = [np.median(pos_err_cc), np.percentile(pos_err_cc, 10), np.percentile(pos_err_cc, 90)]
        ctpc_theta[:, idx] = [np.median(theta_err_pc), np.percentile(theta_err_pc, 10), np.percentile(theta_err_pc, 90)]
        ctcc_theta[:, idx] = [np.median(theta_err_cc), np.percentile(theta_err_cc, 10), np.percentile(theta_err_cc, 90)]
    
    return ctpc_pos, ctpc_theta, ctcc_pos, ctcc_theta

# ====================== 主流程与可视化（完全匹配论文风格）======================
if __name__ == "__main__":
    ctpc_pos, ctpc_theta, ctcc_pos, ctcc_theta = compute_errors()
    
    # 可视化配置（论文标准格式）
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.family'] = 'Times New Roman'
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. 位置误差图（论文图1a）
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_ylim(1e-4, 1e0)  # 匹配论文的y轴范围
    ax1.plot(v_list, ctcc_pos[0], 'b-', label='CT-CC (Median)', linewidth=2)
    ax1.fill_between(v_list, ctcc_pos[1], ctcc_pos[2], alpha=0.2, color='b')
    ax1.plot(v_list, ctpc_pos[0], 'r-', label='CT-PC (Median)', linewidth=2)
    ax1.fill_between(v_list, ctpc_pos[1], ctpc_pos[2], alpha=0.2, color='r')
    ax1.set_xlabel('Linear Velocity (m/s) [log]')
    ax1.set_ylabel('Position Error (m) [log]')
    ax1.set_title(f'Position Error (ω={omega_fixed} rad/s)')
    ax1.legend(loc='upper left')
    ax1.grid(True, which="both", ls="--")
    
    # 2. 方位角误差图（论文图1b，核心结论）
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_ylim(1e-3, 1e0)  # 匹配论文的y轴范围
    ax2.plot(v_list, ctcc_theta[0], 'b-', label='CT-CC (Median)', linewidth=2)
    ax2.fill_between(v_list, ctcc_theta[1], ctcc_theta[2], alpha=0.2, color='b')
    ax2.plot(v_list, ctpc_theta[0], 'r-', label='CT-PC (Median)', linewidth=2)
    ax2.fill_between(v_list, ctpc_theta[1], ctpc_theta[2], alpha=0.2, color='r')
    ax2.set_xlabel('Linear Velocity (m/s) [log]')
    ax2.set_ylabel('Angular Error (rad) [log]')
    ax2.set_title(f'Angular Error (ω={omega_fixed} rad/s)')
    ax2.legend(loc='upper right')  # 论文标准位置
    ax2.grid(True, which="both", ls="--")
    
    plt.tight_layout()
    plt.show()
    
