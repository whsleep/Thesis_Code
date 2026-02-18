import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ====================== 核心参数设置 ======================
np.random.seed(42)
T = 0.1  # 时间步
theta0 = np.pi/4  # 初始方位角
v0 = 10.0  # 固定线速度
sigma_theta, sigma_v, sigma_omega = 0.1, 0.3, 0.1  # 噪声标准差
num_trials = 100000  # 蒙特卡罗次数
omega_list = np.logspace(np.log10(0.01), np.log10(20), 100)  # 角速度对数序列
dot_x0_true, dot_y0_true = v0*np.cos(theta0), v0*np.sin(theta0)  # 无噪声真实初始速度

# ====================== 向量优化的预测函数 ======================
def ct_pc_predict_vec(omega_noise, theta_noise, v_noise, T, x0=0, y0=0):
    """CT-PC模型向量预测（批量处理，含theta/v噪声注入）"""
    omega_k = omega_noise
    theta = theta_noise + omega_k * T  # 注入初始方位角噪声
    v = v_noise  # 注入线速度噪声
    mask = np.abs(omega_k) > 1e-6  # 避免除零
    
    # 初始化预测位置（退化到CV模型）
    x_pre = np.full_like(omega_k, x0 + v * T * np.cos(theta_noise))
    y_pre = np.full_like(omega_k, y0 + v * T * np.sin(theta_noise))
    
    # 非零角速度时的计算
    x_pre[mask] = x0 + (2*v[mask]/omega_k[mask]) * np.sin(omega_k[mask]*T/2) * np.cos(theta[mask])
    y_pre[mask] = y0 + (2*v[mask]/omega_k[mask]) * np.sin(omega_k[mask]*T/2) * np.sin(theta[mask])
    
    return x_pre, y_pre, theta

def ct_cc_predict_vec(omega_noise, dot_x_noisy, dot_y_noisy, T, x0=0, y0=0):
    """CT-CC模型向量预测（批量处理，接收带噪初始速度）"""
    omega_k = omega_noise
    mask = np.abs(omega_k) > 1e-6
    
    # 初始化预测位置和速度（用带噪初始速度）
    x_pre = np.full_like(omega_k, x0 + dot_x_noisy*T)
    y_pre = np.full_like(omega_k, y0 + dot_y_noisy*T)
    dot_x_pre = np.full_like(omega_k, dot_x_noisy)
    dot_y_pre = np.full_like(omega_k, dot_y_noisy)
    
    # 非零角速度时的计算
    cos_wt = np.cos(omega_k[mask]*T)
    sin_wt = np.sin(omega_k[mask]*T)
    x_pre[mask] = x0 + (dot_x_noisy[mask]/omega_k[mask])*sin_wt - (dot_y_noisy[mask]/omega_k[mask])*(1 - cos_wt)
    y_pre[mask] = y0 + (dot_y_noisy[mask]/omega_k[mask])*sin_wt + (dot_x_noisy[mask]/omega_k[mask])*(1 - cos_wt)
    dot_x_pre[mask] = dot_x_noisy[mask]*cos_wt - dot_y_noisy[mask]*sin_wt
    dot_y_pre[mask] = dot_x_noisy[mask]*sin_wt + dot_y_noisy[mask]*cos_wt
    
    return x_pre, y_pre, dot_x_pre, dot_y_pre

# ====================== 蒙特卡罗模拟（完善CT-CC噪声注入）======================
def compute_errors():
    ctpc_pos_stats = np.zeros((3, len(omega_list)))  # 中位数, q10, q90
    ctcc_pos_stats = np.zeros((3, len(omega_list)))
    ctpc_theta_stats = np.zeros((3, len(omega_list)))
    ctcc_theta_stats = np.zeros((3, len(omega_list)))
    
    for idx, omega in enumerate(omega_list):
        print(f"正在计算角速度: {omega:.4f} rad/s")
        # 生成批量噪声样本（所有样本形状均为 (num_trials,)）
        theta_noise = norm.rvs(theta0, sigma_theta, num_trials)  # θ噪声
        v_noise = norm.rvs(v0, sigma_v, num_trials)              # v噪声
        omega_noise = norm.rvs(omega, sigma_omega, num_trials)    # ω噪声
        
        # ---------------------- CT-CC 计算 ----------------------
        # 1. 由v和θ的噪声推导带噪初始速度（符合非完整约束）
        dot_x_noisy = v_noise * np.cos(np.random.normal(0, sigma_theta, num_trials) + theta0)
        dot_y_noisy = v_noise * np.sin(np.random.normal(0, sigma_theta, num_trials) + theta0)
        
        # 2. 理想位置（无噪声）
        ideal_x_cc = np.where(abs(omega)>1e-6,
                             0 + (dot_x0_true/omega)*np.sin(omega*T) - (dot_y0_true/omega)*(1-np.cos(omega*T)),
                             0 + dot_x0_true*T)
        ideal_y_cc = np.where(abs(omega)>1e-6,
                             0 + (dot_y0_true/omega)*np.sin(omega*T) + (dot_x0_true/omega)*(1-np.cos(omega*T)),
                             0 + dot_y0_true*T)
        ideal_theta_cc = np.arctan2(dot_y0_true, dot_x0_true)
        
        # 3. 预测位置（注入初始速度噪声+角速度噪声）
        pred_x_cc, pred_y_cc, pred_dot_x_cc, pred_dot_y_cc = ct_cc_predict_vec(omega_noise, dot_x_noisy, dot_y_noisy, T)
        # 位置误差
        pos_err_cc = np.linalg.norm([pred_x_cc - ideal_x_cc, pred_y_cc - ideal_y_cc], axis=0)
        # 方位角误差（从带噪速度分量推导）
        pred_theta_cc = np.arctan2(pred_dot_y_cc, pred_dot_x_cc)
        theta_err_cc = np.minimum(np.abs(pred_theta_cc - ideal_theta_cc), 2*np.pi - np.abs(pred_theta_cc - ideal_theta_cc))
        
        # ---------------------- CT-PC 计算 ----------------------
        # 理想位置（无噪声）
        ideal_theta_pc = theta0 + omega * T
        ideal_x_pc = np.where(abs(omega)>1e-6,
                             0 + (2*v0/omega)*np.sin(omega*T/2)*np.cos(ideal_theta_pc),
                             0 + v0*T*np.cos(theta0))
        ideal_y_pc = np.where(abs(omega)>1e-6,
                             0 + (2*v0/omega)*np.sin(omega*T/2)*np.sin(ideal_theta_pc),
                             0 + v0*T*np.sin(theta0))
        
        # 预测位置（注入theta/v/omega噪声）
        pred_x_pc, pred_y_pc, pred_theta_pc = ct_pc_predict_vec(omega_noise, theta_noise, v_noise, T)
        # 位置误差
        pos_err_pc = np.linalg.norm([pred_x_pc - ideal_x_pc, pred_y_pc - ideal_y_pc], axis=0)
        # 方位角误差
        theta_err_pc = np.minimum(np.abs(pred_theta_pc - ideal_theta_pc), 2*np.pi - np.abs(pred_theta_pc - ideal_theta_pc))
        
        # ---------------------- 统计结果 ----------------------
        for stats, err in zip([ctpc_pos_stats, ctpc_theta_stats, ctcc_pos_stats, ctcc_theta_stats],
                             [pos_err_pc, theta_err_pc, pos_err_cc, theta_err_cc]):
            stats[:, idx] = [np.median(err), np.percentile(err, 10), np.percentile(err, 90)]
    
    return ctpc_pos_stats, ctpc_theta_stats, ctcc_pos_stats, ctcc_theta_stats

# ====================== 主流程执行 ======================
if __name__ == "__main__":
    # 计算误差统计（向量运算，速度高效）
    ctpc_pos, ctpc_theta, ctcc_pos, ctcc_theta = compute_errors()
    
    # 可视化（遵循参考代码风格）
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.family'] = 'Times New Roman'
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 位置误差图
    ax1.set_ylim(1e-3, 1)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.plot(omega_list, ctcc_pos[0], 'b-', label='CT-CC (Median)', linewidth=2)
    ax1.fill_between(omega_list, ctcc_pos[1], ctcc_pos[2], alpha=0.2, color='b', label='CT-CC (10%-90%)')
    ax1.plot(omega_list, ctpc_pos[0], 'r-', label='CT-PC (Median)', linewidth=2)
    ax1.fill_between(omega_list, ctpc_pos[1], ctpc_pos[2], alpha=0.2, color='r', label='CT-PC (10%-90%)')
    ax1.set_xlabel('Angular Velocity (rad/s) [log]')
    ax1.set_ylabel('Position Error (m) [log]')
    ax1.set_title(f'Position Error (v={v0} m/s)')
    ax1.legend(loc='upper left')
    ax1.grid(True, which="both", ls="--")
    
    # 方位角误差图
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.plot(omega_list, ctcc_theta[0], 'b-', label='CT-CC (Median)', linewidth=2)
    ax2.fill_between(omega_list, ctcc_theta[1], ctcc_theta[2], alpha=0.2, color='b', label='CT-CC (10%-90%)')
    ax2.plot(omega_list, ctpc_theta[0], 'r-', label='CT-PC (Median)', linewidth=2)
    ax2.fill_between(omega_list, ctpc_theta[1], ctpc_theta[2], alpha=0.2, color='r', label='CT-PC (10%-90%)')
    ax2.set_xlabel('Angular Velocity (rad/s) [log]')
    ax2.set_ylabel('Angular Error (rad) [log]')
    ax2.set_title(f'Angular Error (v={v0} m/s)')
    ax2.legend(loc='upper left')
    ax2.grid(True, which="both", ls="--")

    plt.tight_layout()
    plt.show()
    