import time
import numpy as np
from imm_aug_mix import IMMAugMixTracker7D
import irsim

env = irsim.make("../robot_world.yaml")
tracker_mix = IMMAugMixTracker7D(dt=0.01)

# 用于存储跟踪结果
tracking_active = False

# 存储历史轨迹用于绘制 - 确保每个点是 2x1 的列向量
history_estimated = []  # 估计轨迹，每个元素为 2x1 numpy 数组

for _i in range(1000):
    env.step()
    start_time = time.time()
    
    obs_list = env.get_obstacle_info_list()
    
    if len(obs_list) > 0:
        # 获取第一个障碍物的中心位置 [x, y]
        input_center = obs_list[0].center
        
        # 转换为 numpy 数组 (确保是 [x, y] 格式)
        if isinstance(input_center, (tuple, list)):
            z = np.array([float(input_center[0]), float(input_center[1])])
        else:
            z = np.array([float(input_center[0]), float(input_center[1])])
        
        # 第一次观测时初始化跟踪器
        if not tracking_active:
            for f in tracker_mix.filters:
                # 初始化状态: [x, y, theta, v, dx, dy, w]
                f.x = np.array([z[0], z[1], 0.0, 0.0, 0.0, 0.0, 0.0])
            tracking_active = True
            estimated_state = np.array([z[0], z[1], 0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            for i in range(10): # 迭代多次以确保滤波器收敛
                # 执行跟踪步骤
                estimated_state = tracker_mix.step(z)
        
        estimated_pos = estimated_state[:2]
        
        # 记录历史轨迹 - 转换为 2x1 列向量
        # draw_trajectory 期望每个点是 2x1 或 (2, N) 格式
        pos_col = np.array([[estimated_pos[0]], [estimated_pos[1]]])
        history_estimated.append(pos_col)
        pre_traj = tracker_mix.predict_future_7d(steps=200)

        # 绘制估计轨迹 (蓝色实线)
        if len(history_estimated) > 1:
            env.draw_trajectory(pre_traj.T, traj_type="b-", linewidth=4, refresh=True)
        
        print(f"Step {_i}:")
        print(f"  Time taken: {time.time() - start_time:.6f} seconds")
        print(f"  Measurement: [{z[0]:.3f}, {z[1]:.3f}]")
        print(f"  Estimated:   [{estimated_pos[0]:.3f}, {estimated_pos[1]:.3f}]")
        print(f"  Model probs: CV-CC={tracker_mix.mu[0]:.3f}, CV-PC={tracker_mix.mu[1]:.3f}, "
              f"CT-CC={tracker_mix.mu[2]:.3f}, CT-PC={tracker_mix.mu[3]:.3f}")
        print("-" * 60)
    else:
        print("No obstacles detected")
    
    # 渲染环境
    env.render()
    
    if env.done():
        break

env.end()