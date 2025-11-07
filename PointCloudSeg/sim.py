import numpy as np
from irsim.env import EnvBase
import DynamicPointCloudProcessor as DPCP

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
        self.dpcp = DPCP.DynamicPointCloudProcessor()
        
    def step(self, lin_velocity=0.0, ang_velocity=0.0):
        # 环境单步仿真
        self.env.step(action_id=0, action=np.array([[self.v], [self.w]]))
        # 环境可视化
        if self.env.display:
            self.env.render()

        # 获取机器人姿态及环境信息
        robot_state = self.env.get_robot_state()
        scan_data = self.env.get_lidar_scan()

        # 处理点云数据
        point_cloud_result = self.dpcp.process_frame_global(scan_data, robot_state, time_interval=0.1)
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
        使用内置draw_points函数可视化动静态点云
        
        Args:
            point_cloud_result: 动态点云处理结果，包含static_points和dynamic_points
        """
        # 提取静态和动态点云
        static_points = point_cloud_result['static_points']  # 形状: (N1, 2)
        dynamic_points = point_cloud_result['dynamic_points']  # 形状: (N2, 2)
        
        # 1. 将NumPy数组转换为列表格式
        # 使用tolist()方法保持二维结构 [[x1,y1], [x2,y2], ...]
        if len(static_points) > 0:
            static_points_list = static_points.tolist()  # 转换为列表格式
            self.env.draw_points(points=static_points_list, s=10, c='blue', refresh=True, alpha=0.6)
        
        # 2. 叠加绘制动态点云（红色，较大，不清除静态点云）
        if len(dynamic_points) > 0:
            dynamic_points_list = dynamic_points.tolist()  # 转换为列表格式
            self.env.draw_points(points=dynamic_points_list, s=20, c='red', refresh=True, alpha=0.8)


