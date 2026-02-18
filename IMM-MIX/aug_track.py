import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from imm_aug_mix import IMMAugMixTracker7D
from imm_aug_cc import IMMAugCCTracker
from imm_aug_pc import IMMAugPCTracker

def run_real_data_comparison(file_path='groundtruth.txt'):
    # 1. 加载数据
    try:
        data = np.loadtxt(file_path)
    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    # 提取时间、X、Y
    timestamps = data[:, 0]
    true_pos = data[:, 1:3]
    num_steps = len(true_pos)
    
    # 计算实际平均 dt
    dt_list = np.diff(timestamps)
    dt = np.mean(dt_list) if len(dt_list) > 0 else 0.1
    print(f"检测到数据点: {num_steps}, 平均采样间隔 dt: {dt:.4f}s")

    # 2. 初始化跟踪器
    tracker_mix = IMMAugMixTracker7D(dt=dt)
    tracker_cc = IMMAugCCTracker(dt=dt)
    tracker_pc = IMMAugPCTracker(dt=dt)

    # 统一初始化状态 (假设初始速度为0)
    z0 = true_pos[0]
    for t in [tracker_cc, tracker_pc, tracker_mix]:
        for f in t.filters:
            if hasattr(f, 'x'):
                if len(f.x) == 7: # MIX
                    f.x = np.array([z0[0], z0[1], 0.0, 0.0, 0.0, 0.0, 0.0])
                else: # CC/PC 5D
                    f.x = np.array([z0[0], z0[1], 0.0, 0.0, 0.0])

    # 3. 运行跟踪
    results = {
        'MIX': {'pos': [], 'mu': [], 'err': []},
        'CC':  {'pos': [], 'mu': [], 'err': []},
        'PC':  {'pos': [], 'mu': [], 'err': []}
    }
    
    # 添加噪声  (假设测量噪声标准差为0.1)
    noise_std = 0.1
    noise = np.random.normal(0, noise_std, size=true_pos.shape)
    measurements = true_pos + noise

    print("正在处理数据...")
    for i in range(num_steps):
        z = measurements[i]
        
        # 运行各个跟踪器
        for key, tracker in zip(['MIX', 'CC', 'PC'], [tracker_mix, tracker_cc, tracker_pc]):
            x_est = tracker.step(z)
            results[key]['pos'].append(x_est[:2])
            results[key]['mu'].append(tracker.mu.copy())
            results[key]['err'].append(np.linalg.norm(x_est[:2] - z))

    # 4. 转换数据格式以匹配绘图函数
    est_dict = {k: np.array(v['pos']) for k, v in results.items()}
    prob_dict = {k: np.array(v['mu']) for k, v in results.items()}
    rmse_dict = {k: np.array(v['err']) for k, v in results.items()}

    # 5. 调用仪表盘绘图
    plot_integrated_dashboard(true_pos, est_dict, prob_dict, rmse_dict)

def plot_integrated_dashboard(true_pos, est_dict, prob_dict, rmse_dict):
    fig = plt.figure(figsize=(18, 12))
    # 3x2 布局: 左侧 2/3 宽度给轨迹和误差，右侧 1/3 给概率
    gs = GridSpec(3, 2, figure=fig, width_ratios=[2, 1], hspace=0.3, wspace=0.2)
    
    # --- [左侧: 轨迹图] (占据 0,1 行的第 0 列) ---
    ax_traj = fig.add_subplot(gs[0:2, 0])
    ax_traj.plot(est_dict['MIX'][:, 0], est_dict['MIX'][:, 1], 'b-', lw=2, label='IMM-Mix (Hybrid)', alpha=0.7)
    ax_traj.plot(est_dict['CC'][:, 0], est_dict['CC'][:, 1], 'r--', lw=2, label='IMM-CC', alpha=0.7)
    ax_traj.plot(est_dict['PC'][:, 0], est_dict['PC'][:, 1], 'g--', lw=2, label='IMM-PC', alpha=0.7)
    ax_traj.plot(true_pos[:, 0], true_pos[:, 1], 'k-', lw=1., label='Ground Truth') 

    ax_traj.set_title("Trajectory Tracking Performance", fontsize=14, fontweight='bold')
    ax_traj.set_xlabel("X Position [m]"); ax_traj.set_ylabel("Y Position [m]")
    ax_traj.axis('equal'); ax_traj.grid(True, ls=':')
    ax_traj.legend(loc='best')

    # --- [右侧: 概率切换图] (纵向排列在第 1 列) ---
    # CC 概率
    ax_p_cc = fig.add_subplot(gs[0, 1])
    ax_p_cc.plot(prob_dict['CC'][:, 0], label='CV-CC')
    ax_p_cc.plot(prob_dict['CC'][:, 1], label='CT-CC')
    ax_p_cc.set_title("Probabilities: CC Tracker"); ax_p_cc.set_ylim(-0.05, 1.05)
    ax_p_cc.legend(loc='right', fontsize='x-small'); ax_p_cc.grid(True, ls=':', alpha=0.5)

    # PC 概率
    ax_p_pc = fig.add_subplot(gs[1, 1])
    ax_p_pc.plot(prob_dict['PC'][:, 0], label='CV-PC')
    ax_p_pc.plot(prob_dict['PC'][:, 1], label='CT-PC')
    ax_p_pc.set_title("Probabilities: PC Tracker"); ax_p_pc.set_ylim(-0.05, 1.05)
    ax_p_pc.legend(loc='right', fontsize='x-small'); ax_p_pc.grid(True, ls=':', alpha=0.5)

    # MIX 概率 (4个模型)
    ax_p_mix = fig.add_subplot(gs[2, 1])
    mix_lbls = ['CV-CC', 'CV-PC', 'CT-CC', 'CT-PC']
    for i in range(4):
        ax_p_mix.plot(prob_dict['MIX'][:, i], label=mix_lbls[i])
    ax_p_mix.set_title("Probabilities: Mix-7D Tracker"); ax_p_mix.set_ylim(-0.05, 1.05)
    ax_p_mix.set_xlabel("Steps")
    ax_p_mix.legend(loc='lower left', ncol=2, fontsize='xx-small'); ax_p_mix.grid(True, ls=':', alpha=0.5)

    # --- [底部: RMSE 对比图] (占据第 2 行第 0 列) ---
    ax_rmse = fig.add_subplot(gs[2, 0])
    ax_rmse.plot(rmse_dict['CC'], 'r', alpha=0.4, label=f"CC (Mean: {np.mean(rmse_dict['CC']):.3f})")
    ax_rmse.plot(rmse_dict['PC'], 'g', alpha=0.4, label=f"PC (Mean: {np.mean(rmse_dict['PC']):.3f})")
    ax_rmse.plot(rmse_dict['MIX'], 'b', lw=1.5, label=f"MIX (Mean: {np.mean(rmse_dict['MIX']):.3f})")
    
    ax_rmse.set_title("Real-time RMSE Comparison", fontsize=8)
    ax_rmse.set_xlabel("Time Steps"); ax_rmse.set_ylabel("Error [m]")
    ax_rmse.legend(loc='upper right', fontsize='small'); ax_rmse.grid(True, ls=':', alpha=0.5)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_real_data_comparison('groundtruth.txt')