import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 实验参数设置 (严格匹配论文 Experiment 2)
# ---------------------------------------------------------
num_trials = 100000
T = 1.0
v_fixed = 10.0                # 固定线速度 10 m/s
om_range = np.logspace(-2, 1.3, 50) # 覆盖从 0.01 到 20 rad/s

# 噪声标准差定义
sigma_theta = 0.1   # 仅用于 PC 模型初值 (rad)
sigma_v = 0.3       # 速度标准差 (m/s)
sigma_omega = 0.1   # 角速度标准差 (rad/s)

def run_strict_experiment_2():
    results = {
        'om': om_range,
        'pc_pos': [], 'cc_pos': [],
        'pc_ori': [], 'cc_ori': []
    }

    for om_true in om_range:
        # 初始真实值
        theta_init = np.pi / 4
        theta_next_true = theta_init + om_true * T
        
        # 理想位置真值 (基于理想 CT 运动学推导)
        # 使用统一的物理真值作为基准：x_k = x_{k-1} + v/w * (sin(theta+wT) - sin(theta))
        dx_true = (v_fixed / om_true) * (np.sin(theta_next_true) - np.sin(theta_init))
        dy_true = (v_fixed / om_true) * (np.cos(theta_init) - np.cos(theta_next_true))
        pos_true = np.array([dx_true, dy_true])

        # 共享角速度采样
        om_s = om_true + np.random.normal(0, sigma_omega, num_trials)
        om_s[np.abs(om_s) < 1e-6] = 1e-6 # 避免除零

        # --- 模型 A: CT-PC (极坐标表示) ---
        v_s_pc = v_fixed + np.random.normal(0, sigma_v, num_trials)
        th_s_pc = theta_init + np.random.normal(0, sigma_theta, num_trials)
        
        # CT-PC 预测方程 (Table I 第四列)
        dist_pc = (2 * v_s_pc / om_s) * np.sin(om_s * T / 2)
        x_pc = dist_pc * np.cos(th_s_pc + om_s * T / 2)
        y_pc = dist_pc * np.sin(th_s_pc + om_s * T / 2)
        theta_pc_est = th_s_pc + om_s * T

        # --- 模型 B: CT-CC (笛卡尔表示) ---
        # 将速度噪声从极坐标映射到笛卡尔，这样 vx 和 vy 的噪声就包含了 theta 和 v 的耦合
        vx_s = (v_fixed + np.random.normal(0, sigma_v, num_trials)) * np.cos(theta_init + np.random.normal(0, sigma_theta, num_trials))
        vy_s = (v_fixed + np.random.normal(0, sigma_v, num_trials)) * np.sin(theta_init + np.random.normal(0, sigma_theta, num_trials))
        
        # CT-CC 预测方程 (Table I 第三列)
        x_cc = (vx_s / om_s) * np.sin(om_s * T) - (vy_s / om_s) * (1 - np.cos(om_s * T))
        y_cc = (vy_s / om_s) * np.sin(om_s * T) + (vx_s / om_s) * (1 - np.cos(om_s * T))
        
        # 提取方位角用于对比
        vx_next_cc = vx_s * np.cos(om_s * T) - vy_s * np.sin(om_s * T)
        vy_next_cc = vx_s * np.sin(om_s * T) + vy_s * np.cos(om_s * T)
        theta_cc_est = np.arctan2(vy_next_cc, vx_next_cc)

        # --- 统计计算 ---
        def get_stats(err):
            return [np.median(err), np.percentile(err, 10), np.percentile(err, 90)]

        # 位置欧氏距离误差
        err_p_pc = np.linalg.norm(np.stack([x_pc, y_pc], axis=1) - pos_true, axis=1)
        err_p_cc = np.linalg.norm(np.stack([x_cc, y_cc], axis=1) - pos_true, axis=1)
        
        # 方位角绝对误差 (deg)
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
data = run_strict_experiment_2()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for ax, title, key, ylabel in zip([ax1, ax2], 
                                  ['Position Error', 'Orientation Error'], 
                                  ['pos', 'ori'], 
                                  ['Error [m]', 'Error [deg]']):
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # CT-PC (红色实线)
    ax.plot(data['om'], data[f'pc_{key}'][:, 0], 'r-', label='CT-PC', linewidth=2)
    ax.fill_between(data['om'], data[f'pc_{key}'][:, 1], data[f'pc_{key}'][:, 2], color='r', alpha=0.15)
    
    # CT-CC (蓝色虚线)
    ax.plot(data['om'], data[f'cc_{key}'][:, 0], 'b--', label='CT-CC', linewidth=2)
    ax.fill_between(data['om'], data[f'cc_{key}'][:, 1], data[f'cc_{key}'][:, 2], color='b', alpha=0.1)
    
    ax.set_title(f'{title} ($v=10$ m/s)')
    ax.set_xlabel('Angular Velocity $\omega$ [rad/s]')
    ax.set_ylabel(ylabel)
    
    # 范围调整
    if key == 'pos':
        ax.set_ylim(1e-2, 10) 
    else:
        ax.set_ylim(1e-1, 200)

    ax.legend()
    ax.grid(True, which="both", alpha=0.3, linestyle='--')

plt.tight_layout()
plt.show()