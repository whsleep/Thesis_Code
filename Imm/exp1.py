import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# ---------------------------------------------------------
# 实验参数设置 (严格匹配论文 Experiment 1)
# ---------------------------------------------------------
num_trials = 100000
T = 0.1
omega_fixed = 0.1
v_range = np.logspace(-3, 2, 100) # 覆盖从 0.001 到 100 m/s

# 噪声标准差定义
sigma_theta = 0.1   # 仅用于 PC 模型 (rad)
sigma_v = 0.3       # 速度标准差 (m/s)
sigma_omega = 0.1   # 角速度标准差 (rad/s)

def run_strict_experiment_1():
    results = {
        'v': v_range,
        'pc_pos': [], 'cc_pos': [],
        'pc_ori': [], 'cc_ori': []
    }

    for v_true in v_range:
        # 初始真实值
        theta_init = np.pi / 4
        theta_next_true = theta_init + omega_fixed * T
        
        # 理想位置真值 (用于计算误差)，适配 CT-CC 模型
        dx_true = (v_true / omega_fixed) * (np.sin(theta_next_true) - np.sin(theta_init))
        dy_true = (v_true / omega_fixed) * (np.cos(theta_init) - np.cos(theta_next_true))
        pos_true = np.array([dx_true, dy_true])

        # 共享角速度采样
        om_s = omega_fixed + np.random.normal(0, sigma_omega, num_trials)
        om_s[np.abs(om_s) < 1e-6] = 1e-6

        # --- 模型 A: CT-PC (极坐标表示) ---
        # 它的噪声直接定义在 v 和 theta 上
        v_s_pc = v_true + np.random.normal(0, sigma_v, num_trials)
        th_s_pc = theta_init + np.random.normal(0, sigma_theta, num_trials)
        
        dist_pc = (2 * v_s_pc / om_s) * np.sin(om_s * T / 2)
        x_pc = dist_pc * np.cos(th_s_pc + om_s * T / 2)
        y_pc = dist_pc * np.sin(th_s_pc + om_s * T / 2)
        theta_pc_est = th_s_pc + om_s * T

        # --- 模型 B: CT-CC (笛卡尔表示) ---
        # 核心修改：它接收的是 vx, vy，且噪声独立加在分量上
        vx_true = v_true * np.cos(theta_init)
        vy_true = v_true * np.sin(theta_init)
        
        # 模拟笛卡尔速度噪声 sigma_x = sigma_y = sigma_v (符合非完整约束下的各向同性假设)
        vx_s = vx_true + np.random.normal(0, sigma_v, num_trials)
        vy_s = vy_true + np.random.normal(0, sigma_v, num_trials)
        
        # CT-CC 预测方程 (Table I 第三列)
        x_cc = (vx_s / om_s) * np.sin(om_s * T) - (vy_s / om_s) * (1 - np.cos(om_s * T))
        y_cc = (vy_s / om_s) * np.sin(om_s * T) + (vx_s / om_s) * (1 - np.cos(om_s * T))
        
        # 计算预测后的速度分量以提取方位角
        vx_next_cc = vx_s * np.cos(om_s * T) - vy_s * np.sin(om_s * T)
        vy_next_cc = vx_s * np.sin(om_s * T) + vy_s * np.cos(om_s * T)
        theta_cc_est = np.arctan2(vy_next_cc, vx_next_cc)

        # --- 误差计算与统计 ---
        def get_stats(err):
            return [np.median(err), np.percentile(err, 10), np.percentile(err, 90)]

        err_p_pc = np.linalg.norm(np.stack([x_pc, y_pc], axis=1) - pos_true, axis=1)
        err_p_cc = np.linalg.norm(np.stack([x_cc, y_cc], axis=1) - pos_true, axis=1)
        
        def angle_err(est, true):
            return np.rad2deg(np.abs(np.arctan2(np.sin(est - true), np.cos(est - true))))

        results['pc_pos'].append(get_stats(err_p_pc))
        results['cc_pos'].append(get_stats(err_p_cc))
        results['pc_ori'].append(get_stats(angle_err(theta_pc_est, theta_next_true)))
        results['cc_ori'].append(get_stats(angle_err(theta_cc_est, theta_next_true)))

    return {k: np.array(v) for k, v in results.items()}

# ---------------------------------------------------------
# 可视化结果
# ---------------------------------------------------------
data = run_strict_experiment_1()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))


for ax, title, key, ylabel in zip([ax1, ax2], 
                                  ['Position Error', 'Orientation Error'], 
                                  ['pos', 'ori'], 
                                  ['Error [m]', 'Error [deg]']):
    ax.set_xscale('log')
    ax.set_yscale('log')

    # CT-PC (红色实线)
    ax.plot(data['v'], data[f'pc_{key}'][:, 0], 'r-', label='CT-PC', linewidth=2)
    ax.fill_between(data['v'], data[f'pc_{key}'][:, 1], data[f'pc_{key}'][:, 2], color='r', alpha=0.15)
    
    # CT-CC (蓝色虚线)
    ax.plot(data['v'], data[f'cc_{key}'][:, 0], 'b--', label='CT-CC', linewidth=2)
    ax.fill_between(data['v'], data[f'cc_{key}'][:, 1], data[f'cc_{key}'][:, 2], color='b', alpha=0.1)
    
    ax.set_title(f'{title} ($\omega=0.1$ rad/s)')
    ax.set_xlabel('Linear Velocity v [m/s]')
    ax.set_ylabel(ylabel)
    
    # 关键修正：确保 y 轴显示范围合理，防止因为 1e-10 级别的误差导致坐标轴标识混乱
    if key == 'pos':
        ax.set_ylim(1e-4, 100) # 位置误差范围
    else:
        ax.set_ylim(1e-2, 200) # 角度误差范围 (deg)

    ax.legend()
    # 开启网格，包括次要刻度
    ax.grid(True, which="both", alpha=0.3, linestyle='--')

plt.tight_layout()
plt.show()