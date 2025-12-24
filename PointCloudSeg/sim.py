import numpy as np
from irsim.env import EnvBase
import matplotlib.pyplot as plt
from DynamicPointCloudProcessor import DynamicPointCloudProcessor as DPCP
from DynamicObstacleTracker import DynamicObstacleTracker as DOT    

class SIM_ENV:
    def __init__(self, world_file="robot_world.yaml", render=False):

        # 初始化环境
        self.env = EnvBase(world_file, display=render, disable_all_plot=not render,save_ani = False)
        # 环境参数
        self.robot_goal = self.env.get_robot_info(0).goal
        
        # 速度指令
        self.v = 0.0
        self.w = 0.0

        # 点云处理器
        grid_map = self.env.get_map()
        self.dpcp = DPCP(dt=0.1) # 假设 10Hz
        self.dpcp.set_map(grid_map)
        self.dot = DOT(dt=0.1)


    def step(self, lin_velocity=0.0, ang_velocity=0.0):
        # 环境单步仿真
        self.env.step(action_id=0, action=np.array([[self.v], [self.w]]))
        # 环境可视化
        if self.env.display:
            self.env.render()

        # 获取机器人姿态及环境信息
        robot_state = self.env.get_robot_state()
        scan_data = self.env.get_lidar_scan()

        # 处理点云，分离动态和静态点
        detect_data = self.dpcp.process(scan_data, robot_state)

        # 跟踪器更新
        tracked_dynamics = self.dot.update(
            obstacles=detect_data['obstacles'],
        )
        # 假设在你的主运行逻辑中（处理器+跟踪器更新后）
        if tracked_dynamics is not None and len(tracked_dynamics) > 0:
            # 颜色池（可扩展，确保不同track_id对应不同颜色）
            COLOR_PALETTE = ["r", "g", "b", "y", "m", "c"]
            track_colors = {}  # {track_id: 颜色} 确保同一跟踪目标颜色一致
            
            # 第一步：先遍历所有动态障碍物，分配颜色并绘制（refresh=False 避免清除）
            for track_id, obs_info in tracked_dynamics.items():
                # 为每个track_id分配唯一颜色（循环使用颜色池）
                if track_id not in track_colors:
                    track_colors[track_id] = COLOR_PALETTE[track_id % len(COLOR_PALETTE)]
                color = track_colors[track_id]
                
                # 获取矩形框顶点（确保格式是2x4或4x2，适配draw_box要求）
                box_vertex = obs_info['box']
                cx, cy, _, _, _ = obs_info['rect']
                if box_vertex is None:
                    continue
                
                # 绘制矩形框：color参数格式为 "-颜色"（边框颜色），refresh=False 不清除之前的框
                self.env.draw_box(vertex=box_vertex, refresh=True, color=f"-{color}")
                self.env.draw_points(points=np.array([[cx], [cy]]), s=10, c=color, refresh=False)

                
        # 是否抵达
        if self.env.robot.arrive:
            print("Goal reached")
            return True
        
        # 是否碰撞
        if self.env.robot.collision:
            print("collision !!!")
            return True
        
        return False

    def visualize_dynamic_static_points(self, point_cloud_result):
            """
            使用内置 draw_points 函数可视化动静态点云
            
            Args:
                point_cloud_result: 包含 'static_points' 和 'dynamic_points' 的字典
            """
            static_points = point_cloud_result['static_points']   # 形状 (N, 2)
            dynamic_points = point_cloud_result['dynamic_points'] # 形状 (M, 2)

            # 1. 绘制静态点云
            # refresh=True: 清除上一帧的所有点，开始绘制当前帧
            if len(static_points) > 0:
                self.env.draw_points(
                    points=static_points.T,  # 转换为 (2, N) 以符合函数要求
                    s=20, 
                    c='blue', 
                    refresh=True,            # 第一步绘制需刷新画布
                    alpha=1.0
                )
            
            # 2. 绘制动态点云
            # refresh=False: 在保留刚才绘制的静态点的基础上，叠加绘制动态点
            if len(dynamic_points) > 0:
                self.env.draw_points(
                    points=dynamic_points.T, # 转换为 (2, M)
                    s=25, 
                    c='red', 
                    refresh=True,           # 关键：设置为 False 避免清除刚才绘制的蓝色静态点
                    alpha=0.9
                )