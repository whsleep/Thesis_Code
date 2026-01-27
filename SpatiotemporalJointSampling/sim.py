import numpy as np
import time
import sys
from irsim.env import EnvBase
import cv2
from sklearn.cluster import DBSCAN


from baseline.TebSolver import TebplanSolver
from baseline.DwaplanSolver import DwaplanSolver
from baseline.AccelSpaceDwaSolver import AccelSpaceDwaSolver
from baseline.MppiSolver import MppiplanSolver


class SIM_ENV:
    def __init__(self, world_file="robot_world.yaml", render=False):

        # 初始化环境
        self.env = EnvBase(world_file, display=render, disable_all_plot=not render,save_ani = True)
        # 环境参数
        self.robot_goal = self.env.get_robot_info(0).goal
        # 读取参考路径
        npy_path = sys.path[0] + '/dynamic_obs.npy'
        self.ref_path = list(np.load(npy_path, allow_pickle=True))
        self.env.draw_trajectory(self.ref_path, traj_type='-k') 
        self.ref_path = np.array(self.ref_path).squeeze()

        # 局部求解器
        # self.solver = DwaplanSolver(ref_path=self.ref_path, 
        #                             delta_t=0.1,
        #                             v_max=10.0, 
        #                             omega_max=3.0, 
        #                             a_max=8.0, 
        #                             alpha_max=5.0, 
        #                             predict_time=2.0,
        #                             sample_v_count=10,
        #                             sample_w_count=20,
        #                             w_heading=0.15, 
        #                             w_dist=1, 
        #                             w_vel=0.8,
        #                             w_obs=10.0, 
        #                             w_track=5,
        #                             safe_distance=0.5,
        #                             lookahead_dist=5.0)
        # self.solver = MppiplanSolver(ref_path=self.ref_path,
        #                              delta_t=0.1,
        #                              max_omega_abs=3.0,
        #                              max_vel_abs=10.0,
        #                              horizon_step_T=20,
        #                              number_of_samples_K=200,
        #                              param_exploration= 0.3,
        #                              param_lambda=50.0,
        #                              param_alpha=1.0,
        #                              sigma=np.array([[0.5, 0.0], [0.0, 1.0]]),
        #                              stage_cost_weight=np.array([20.0, 20.0, 20.0, 1.0]),
        #                              terminal_cost_weight=np.array([50.0, 50.0, 50.0, 1.0]),
        #                              w_obs=100.0,
        #                              safe_distance=0.5,
        #                              visualize_optimal_traj=True,
        #                              visualze_sampled_trajs=True)
        # self.solver = TebplanSolver(ref_path=self.ref_path)
        self.solver = AccelSpaceDwaSolver(ref_path=self.ref_path)

    def step(self,):
        # 环境可视化
        if self.env.display:
            self.env.render()

        # 获取机器人状态
        self.robot_state = self.env.get_robot_state()
        scan_data = self.env.get_lidar_scan()
        obs_list, center_list = self.scan_ellipse(self.robot_state,scan_data)
        # 绘制障碍
        for obs in obs_list:
            self.env.draw_box(obs, refresh=True, color= "-b")

        # 计算求解时间
        star_time = time.time()
        opt_traj ,samp_traj, action_input_list = self.solver.calc_control_input(self.robot_state,
                                                                                obstacles=center_list)
        end_time = time.time()
        # print(" DWA solver time:", end_time - star_time)
        # print(" MMPI solver time:", end_time - star_time)
        print(" TEB solver time:", end_time - star_time)
        # 绘制采样轨迹
        for i in range(samp_traj.shape[0]):
            list_of_arrays = [np.array(row).reshape(-1, 1) for row in samp_traj[i]]
            self.env.draw_trajectory(list_of_arrays, traj_type='-y', lw=0.5, refresh=True)
        # 绘制最优轨迹
        list_of_arrays = [np.array(row).reshape(-1, 1) for row in opt_traj]
        self.env.draw_trajectory(list_of_arrays, traj_type='-r', lw=2.0, refresh=True)

        # 执行动作
        # [v, steer]
        self.env.step(action_id=0, action=action_input_list)

        # 是否抵达
        if self.env.robot.arrive:
            print("Goal reached")
            return True
        
        # 是否碰撞
        if self.env.robot.collision:
            print("collision !!!")
            return True
        
        return False
    
    def scan_ellipse(self, state, scan_data):
        ranges = np.array(scan_data['ranges'])
        angles = np.linspace(scan_data['angle_min'], scan_data['angle_max'], len(ranges))
        point_list = []
        obstacle_list = []
        center_list = []  # 在这里，center_list 将存储椭圆的完整参数 [cx, cy, a, b, theta]
        for i in range(len(ranges)):
            scan_range = ranges[i]
            angle = angles[i]
            if scan_range < (scan_data['range_max'] - 0.1):
                # 激光雷达坐标系下的点
                point = np.array([[scan_range * np.cos(angle)], [scan_range * np.sin(angle)]])
                point_list.append(point)
        if len(point_list) < 5:  # 拟合椭圆至少需要5个点
            return obstacle_list, center_list
        else:
            point_array = np.hstack(point_list).T
            # 使用 DBSCAN 聚类
            labels = DBSCAN(eps=0.4, min_samples=1).fit_predict(point_array)
            for label in np.unique(labels):
                if label == -1:
                    continue
                
                point_array2 = point_array[labels == label]
                
                # 拟合椭圆需要至少 5 个非共线点
                if len(point_array2) < 5:
                    continue
                # 1. 拟合局部坐标系下的椭圆
                # ellipse 返回格式: ((中心x, 中心y), (长短轴直径w, h), 旋转角度deg)
                ellipse = cv2.fitEllipse(point_array2.astype(np.float32))
                (lc_x, lc_y), (w, h), angle_deg = ellipse
                # 2. 坐标变换：从机器人局部坐标系转到全局坐标系
                trans = state[0:2] # [x, y]
                rot = state[2, 0]  # theta
                R = np.array([[np.cos(rot), -np.sin(rot)], 
                            [np.sin(rot),  np.cos(rot)]])
                # 转换中心点
                center_local = np.array([[lc_x], [lc_y]])
                center_global = trans + R @ center_local
                # 转换旋转角 (OpenCV 角度是顺时针，需注意与机器人坐标系的映射)
                # 全局角度 = 局部椭圆角度 + 机器人当前朝向
                angle_rad = np.deg2rad(angle_deg) + rot
                obs_a = max(w / 2.0, 0.05)
                obs_b = max(h / 2.0, 0.05)
                
                # --- 预计算三角函数 ---
                cos_ot = np.cos(angle_rad)
                sin_ot = np.sin(angle_rad)
                
                # 扩展返回参数：[cx, cy, a, b, theta, cos_theta, sin_theta]
                ellipse_params = [
                    center_global[0, 0], 
                    center_global[1, 0], 
                    obs_a, 
                    obs_b, 
                    angle_rad,
                    cos_ot,
                    sin_ot
                ]
                center_list.append(ellipse_params)
                # 4. 为了可视化，仍然可以计算矩形顶点（用于 draw_box）
                rect = cv2.minAreaRect(point_array2.astype(np.float32))
                box = cv2.boxPoints(rect)
                global_vertices = trans + R @ box.T
                obstacle_list.append(global_vertices)
            return obstacle_list, center_list