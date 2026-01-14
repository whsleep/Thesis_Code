import matplotlib.pyplot as plt
import numpy as np
from SimpleMapper import njit_bresenham_line

def plot_grid_map_update():
    # 模拟一个10x10栅格地图（0=未知，1=自由，2=障碍）
    grid = np.zeros((10, 10), dtype=int)
    robot_pos = (5, 5)  # 机器人栅格坐标
    obstacle_pos = (8, 6)  # 障碍物栅格坐标
    
    # 计算激光射线路径（Bresenham算法）
    path = njit_bresenham_line(robot_pos[0], robot_pos[1], obstacle_pos[0], obstacle_pos[1])
    
    # 更新栅格：路径为自由空间，终点为障碍
    for (x, y) in path[:-1]:
        grid[y, x] = 1  # 自由空间
    grid[obstacle_pos[1], obstacle_pos[0]] = 2  # 障碍物
    
    # 绘图
    plt.figure(figsize=(8, 8))
    cmap = plt.cm.colors.ListedColormap(['white', 'lightgreen', 'red'])  # 未知=白，自由=绿，障碍=红
    plt.imshow(grid, cmap=cmap, origin='lower')
    
    # 标记机器人位置
    plt.scatter(robot_pos[0], robot_pos[1], c='blue', s=100, marker='o', label='机器人')
    # 标记激光路径
    path_x = [p[0] for p in path]
    path_y = [p[1] for p in path]
    plt.plot(path_x, path_y, 'k--', linewidth=2, label='激光路径')
    
    plt.xlabel('栅格X')
    plt.ylabel('栅格Y')
    plt.title('栅格地图更新示意图（Bresenham算法）')
    plt.legend()
    plt.grid(True)
    plt.show()

# 运行绘图
plot_grid_map_update()