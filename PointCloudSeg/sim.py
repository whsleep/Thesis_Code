import numpy as np
from irsim.env import EnvBase
import matplotlib.pyplot as plt
from DynamicPointCloudProcessor import DynamicPointCloudProcessor as DPCP

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
        point_cloud_result = self.dpcp.process(scan_data, robot_state)
        self.visualize_dynamic_static_points(point_cloud_result)

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