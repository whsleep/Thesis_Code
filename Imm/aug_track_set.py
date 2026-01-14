import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from imm_aug_mix import IMMAugMixTracker7D
from imm_aug_cc import IMMAugCCTracker
from imm_aug_pc import IMMAugPCTracker

# ========================================================
# 1. 论文 V.A 节运动模型 (用于数据生成)
# ========================================================

def fx_cv_pc(x, dt):
    """极坐标匀速模型: [x, y, theta, v, w]"""
    res = np.copy(x)
    res[0] = x[0] + x[3] * dt * np.cos(x[2])
    res[1] = x[1] + x[3] * dt * np.sin(x[2])
    return res

def fx_ct_pc(x, dt):
    """极坐标协同转弯模型"""
    theta, v, w = x[2], x[3], x[4]
    if abs(w) < 1e-6: return fx_cv_pc(x, dt)
    res = np.copy(x)
    half_w_dt = 0.5 * w * dt
    chord = (2.0 * v / w) * np.sin(half_w_dt)
    res[0] = x[0] + chord * np.cos(theta + half_w_dt)
    res[1] = x[1] + chord * np.sin(theta + half_w_dt)
    res[2] = theta + w * dt
    return res

def generate_paper_synthetic_data(target_v=5.0, T=0.1):
    state = np.array([0.0, 0.0, 0.0, target_v, 0.0]) 
    traj = []
    
    # 1. 直线 (30s)
    for _ in range(int(30/T)):
        state = fx_cv_pc(state, T); traj.append(state.copy())
        
    # 2. Stop-and-go (15s)
    accel = target_v / 5.0
    for _ in range(int(5/T)): # 减速
        state[3] = max(0, state[3] - accel * T); state = fx_cv_pc(state, T); traj.append(state.copy())
    for _ in range(int(5/T)): # 停止
        state[3] = 0.0; state = fx_cv_pc(state, T); traj.append(state.copy())
    for _ in range(int(5/T)): # 加速
        state[3] += accel * T; state = fx_cv_pc(state, T); traj.append(state.copy())
        
    # 3. 协同转弯 (±0.1 rad/s)
    for w in [0.1, -0.1]:
        state[4] = w
        turn_duration = int((np.pi/2) / abs(w) / T)
        for _ in range(turn_duration):
            state = fx_ct_pc(state, T); traj.append(state.copy())
        state[4] = 0.0
        for _ in range(int(10/T)):
            state = fx_cv_pc(state, T); traj.append(state.copy())

    # 4. 急转弯 (瞬间改变航向)
    for dw in [np.pi/2, -np.pi/2]:
        state[2] += dw
        for _ in range(int(10/T)):
            state = fx_cv_pc(state, T); traj.append(state.copy())

    return np.array(traj)

# ========================================================
# 2. 实验对比逻辑
# ========================================================

def run_comparison_experiment(target_speed=5.0):
    dt = 0.05
    noise_std = 0.3 # 观测噪声
    
    true_traj = generate_paper_synthetic_data(target_v=target_speed, T=dt)
    true_pos = true_traj[:, :2]
    num_steps = len(true_pos)
    measurements = true_pos + np.random.normal(0, noise_std, true_pos.shape)

    def run_tracker(tracker_class, mode_type):
        tracker = tracker_class(dt=dt)
        z0 = measurements[0]
        
        # 针对不同维度的状态向量进行初始化修复
        for f in tracker.filters:
            if mode_type == "PC": # 5D [x, y, theta, v, w]
                f.x = np.array([z0[0], z0[1], 0.0, 0.0, 0.0])
            elif mode_type == "CC": # 5D [x, y, dx, dy, w]
                f.x = np.array([z0[0], z0[1], 0.0, 0.0, 0.0])
            elif mode_type == "MIX": # 7D [x, y, theta, v, dx, dy, w]
                f.x = np.array([z0[0], z0[1], 0.0, 0.0, 0.0, 0.0, 0.0])
        
        ests, probs, rmses = [], [], []
        for i in range(num_steps):
            z = measurements[i]
            x_est = tracker.step(z)
            ests.append(x_est[:2]) # 仅提取位置用于 RMSE
            probs.append(tracker.mu.copy())
            rmses.append(np.linalg.norm(true_pos[i] - x_est[:2]))
        return np.array(ests), np.array(probs), np.array(rmses)

    # 运行三个跟踪器
    est_mix, prob_mix, rmse_mix = run_tracker(IMMAugMixTracker7D, "MIX")
    est_cc, prob_cc, rmse_cc = run_tracker(IMMAugCCTracker, "CC")
    est_pc, prob_pc, rmse_pc = run_tracker(IMMAugPCTracker, "PC")

    # 2. 按照要求分成三个字典
    est_dict = {'CC': est_cc, 'PC': est_pc, 'MIX': est_mix}
    prob_dict = {'CC': prob_cc, 'PC': prob_pc, 'MIX': prob_mix}
    rmse_dict = {'CC': rmse_cc, 'PC': rmse_pc, 'MIX': rmse_mix}

    # 可视化结果
    plot_integrated_dashboard(true_pos=true_pos, measurements=measurements, est_dict=est_dict, prob_dict=prob_dict, rmse_dict=rmse_dict)


def plot_integrated_dashboard(true_pos, measurements, est_dict, prob_dict, rmse_dict):
    """
    est_dict: {'CC': est_cc, 'PC': est_pc, 'MIX': est_mix}
    prob_dict: {'CC': prob_cc, 'PC': prob_pc, 'MIX': prob_mix}
    rmse_dict: {'CC': rmse_cc, 'PC': rmse_pc, 'MIX': rmse_mix}
    """
    fig = plt.figure(figsize=(18, 12))
    # 创建 3x2 网格
    gs = GridSpec(3, 2, figure=fig, width_ratios=[2, 1], hspace=0.3, wspace=0.2)

    # --- 1. 左侧大图：跟踪轨迹图 (占据 [0:2, 0]) ---
    ax_traj = fig.add_subplot(gs[0:2, 0])
    ax_traj.plot(est_dict['CC'][:, 0], est_dict['CC'][:, 1], 'r--', lw=5, label='IMM-CC', alpha=0.7)
    ax_traj.plot(est_dict['PC'][:, 0], est_dict['PC'][:, 1], 'g--', lw=5, label='IMM-PC', alpha=0.7)
    ax_traj.plot(est_dict['MIX'][:, 0], est_dict['MIX'][:, 1], 'b-', lw=5, label='IMM-Mix (Hybrid)', alpha=0.7)
    ax_traj.plot(true_pos[:, 0], true_pos[:, 1], 'k-', lw=1., label='Ground Truth')
    ax_traj.set_title("Trajectory Tracking Performance", fontsize=14, fontweight='bold')
    ax_traj.set_xlabel("X Position [m]"); ax_traj.set_ylabel("Y Position [m]")
    ax_traj.legend(); ax_traj.axis('equal'); ax_traj.grid(True, ls=':')


    # --- 绝对定位局部放大图 (手动创建) ---
    # [left, bottom, width, height] 均为相对于主图 ax_traj 的比例
    axins = ax_traj.inset_axes([0.05, 0.3, 0.4, 0.4]) 

    # 在小窗里画同样的内容
    axins.plot(true_pos[:,0], true_pos[:,1], 'k--', alpha=0.6)
    axins.plot(est_dict['MIX'][:,0], est_dict['MIX'][:,1], 'b-', lw=2)
    axins.plot(est_dict['CC'][:,0], est_dict['CC'][:,1], 'y-', lw=1.5)
    axins.plot(est_dict['PC'][:,0], est_dict['PC'][:,1], 'r-', lw=1.5)

    # 设置你想要放大的具体区域 (例如第 450 步的 Sharp Turn)
    zoom_idx = 1928
    x_c, y_c = true_pos[zoom_idx, 0], true_pos[zoom_idx, 1]
    zoom_range = 10  # 放大窗口显示的范围大小 (单位: 米)
    axins.set_xlim(x_c - zoom_range, x_c + zoom_range)
    axins.set_ylim(y_c - zoom_range, y_c + zoom_range)

    # 修饰小窗：去掉刻度标签，增加显眼的边框
    axins.set_xticklabels([]); axins.set_yticklabels([])
    for spine in axins.spines.values():
        spine.set_edgecolor('gray')
        spine.set_linewidth(2)


    # --- 2. 右侧三张小图：概率切换 (占据 [0,1], [1,1], [2,1]) ---
    # (a) CC 概率
    ax_p_cc = fig.add_subplot(gs[0, 1])
    for i, lbl in enumerate(['CV-CC', 'CT-CC']):
        ax_p_cc.plot(prob_dict['CC'][:, i], label=lbl)
    ax_p_cc.set_title("Probabilities: CC Tracker"); ax_p_cc.set_ylim(-0.05, 1.05); ax_p_cc.legend(loc='right', fontsize='x-small')

    # (b) PC 概率
    ax_p_pc = fig.add_subplot(gs[1, 1])
    for i, lbl in enumerate(['CV-PC', 'CT-PC']):
        ax_p_pc.plot(prob_dict['PC'][:, i], label=lbl, color=['#2ca02c', '#d62728'][i])
    ax_p_pc.set_title("Probabilities: PC Tracker"); ax_p_pc.set_ylim(-0.05, 1.05); ax_p_pc.legend(loc='right', fontsize='x-small')

    # (c) MIX 概率
    ax_p_mix = fig.add_subplot(gs[2, 1])
    for i, lbl in enumerate(['CV-CC', 'CV-PC', 'CT-CC', 'CT-PC']):
        ax_p_mix.plot(prob_dict['MIX'][:, i], label=lbl)
    ax_p_mix.set_title("Probabilities: Mix-7D Tracker"); ax_p_mix.set_ylim(-0.05, 1.05); ax_p_mix.legend(loc='lower left', ncol=2, fontsize='x-small')

    # --- 3. 最下面一行左侧：RMSE 对比图 (占据 [2, 0]) ---
    ax_rmse = fig.add_subplot(gs[2, 0])
    ax_rmse.plot(rmse_dict['CC'], 'r', alpha=0.4, label=f"CC (Mean: {np.mean(rmse_dict['CC']):.3f})")
    ax_rmse.plot(rmse_dict['PC'], 'g', alpha=0.4, label=f"PC (Mean: {np.mean(rmse_dict['PC']):.3f})")
    ax_rmse.plot(rmse_dict['MIX'], 'b', lw=1.5, label=f"Mix-7D (Mean: {np.mean(rmse_dict['MIX']):.3f})")
    ax_rmse.set_title("Real-time RMSE Comparison", fontsize=12)
    ax_rmse.set_xlabel("Time Steps"); ax_rmse.set_ylabel("Error [m]")
    ax_rmse.legend(loc='upper right', fontsize='small'); ax_rmse.grid(True, ls=':', alpha=0.5)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_comparison_experiment(target_speed=5.0)