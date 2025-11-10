# 实验环境配置

## 雷达配置

使用 `ir-sim` 仿真2D雷达, 雷达参数配置在 [robot_world.yaml](/PointCloudSeg/robot_world.yaml)

```yaml
    sensors:
      - type: 'lidar2d'
        range_min: 0
        range_max: 6
        angle_range: 6.28
        number: 360
        noise: True
        std: 0.02
        angle_std: 0.05
        alpha: 0.4
```

仿真步长为 `0.1s` ,同样在 [robot_world.yaml](/PointCloudSeg/robot_world.yaml) 配置

```yaml
  step_time: 0.1
  sample_time: 0.1
```
## 环境配置

一个固定直角边界,用于获取静态点云

```yaml
  - shape: {name: 'linestring', vertices: [[4, 2], [6, 2], [6, 4]] }  # vertices
    state: [0, 0, 0] 
    unobstructed: False
```

随机位置和形状的移动障碍

```yaml
  - number: 10
    kinematics: {name: 'omni'}
    distribution: {name: 'random', range_low: [1.5, 1.5, -3.14], range_high: [8.5, 8.5, 3.14]}
    behavior: {name: 'rvo', wander: True, range_low: [1.5, 1.5, -3.14], range_high: [8.5, 8.5, 3.14], vxmax: 0.5, vymax: 0.5, factor: 1.0}
    vel_max: [0.5, 0.5]
    vel_min: [-0.5, -0.5]
    shape:
    shape:
      - {name: 'circle', radius: 0.3, random_shape: False}
      - {name: 'polygon', random_shape: true, avg_radius_range: [0.2, 0.3], irregularity_range: [0, 0.2], spikeyness_range: [0, 0.2], num_vertices_range: [4, 6]}
```

## 雷达信息格式

**Python字典**，其中包含了激光雷达（LIDAR）传感器的数据


```python
{
    'angle_min': -3.14,           # 扫描角度最小值（弧度）
    'angle_max': 3.14,            # 扫描角度最大值（弧度）
    'angle_increment': 0.017444444444444446,  # 角度增量
    'time_increment': 0.0002776369562825286,  # 时间增量
    'scan_time': 0.1,             # 扫描时间
    'range_min': 0,               # 最小距离
    'range_max': 6,               # 最大距离
    'ranges': array([5.97138451, 5.97054631, ...]),  # 距离测量数组
    'intensities': None,          # 强度数据（未提供）
    'velocity': array([[0., 0., ...], [0., 0., ...]])  # 速度数据
}
```

## 状态信息格式

`3\times 1` 的二维数组，每个元素依次代表

- `x` 轴坐标
- `y` 轴坐标
- `theta` 车辆朝向与 `x` 轴夹角，弧度制

```python
array([[3.5],
       [4. ],
       [0. ]])
```

### 关键组件

1. **基础参数**：
   - `angle_min`/`angle_max`: 扫描范围（-π到π，覆盖360度）
   - `angle_increment`: 相邻测量点的角度间隔（约1度）
   - `scan_time`: 完成一次完整扫描的时间（0.1秒）

2. **数据数组**：
   - `ranges`: **NumPy数组**，包含每个角度的距离测量值
   - `velocity`: **2D NumPy数组**，包含速度信息

3. **数值类型**：
   - 所有数值都是浮点数
   - `array()`表示这些是**NumPy数组**对象


## 动态点云的提取

考虑点云，距离和角度的零均值高斯误差

```yaml
noise: True
std: 0.02
angle_std: 0.05
```

### 极坐标点云转换全局二维点云

#### 坐标转换基础

首先，激光雷达的原始观测值是在极坐标系下的，

每个点由距离  `r`  和角度  `theta`  表示。它们转换到二维笛卡尔坐标系的公式是：

$$
x = r \cos\theta\\
y = r \sin\theta
$$

#### 方差转换的核心推导

测量值  `r`  和  `theta`  本身是存在误差的随机变量，

通常假设其方差已知，分别为  $\sigma_r^2$  （距离方差）和  $\sigma_\theta^2$  （角度方差）。

同时，我们常假设 `r`  和  `theta`  的测量误差是相互独立的，即它们的协方差 $ \operatorname{Cov}(r, \theta) = 0 $。

误差传播的核心工具是一阶泰勒展开（Delta方法）。

通过线性化，二维点坐标 $(x, y)$ 的协方差矩阵 $\Sigma_{xy}$  可以通过雅可比矩阵 `J` 从极坐标的协方差矩阵  $\Sigma_{r\theta}$  变换得到：

$$
\Sigma_{xy} = J \Sigma_{r\theta} J^T
$$

其中，极坐标的协方差矩阵为对角矩阵

$$
\Sigma_{r\theta} = \begin{bmatrix} \sigma_r^2 & 0 \\ 0 & \sigma_\theta^2 \end{bmatrix}
$$

雅可比矩阵 `J` 包含了转换函数  $x = r\cos\theta, y = r\sin\theta$  对每个输入变量 $(r, \theta)$ 的偏导数：

$$ 
J = \begin{bmatrix} 
\frac{\partial x}{\partial r} & \frac{\partial x}{\partial \theta} \\ 
\frac{\partial y}{\partial r} & \frac{\partial y}{\partial \theta} 
\end{bmatrix} 
= 
\begin{bmatrix} 
\cos\theta & -r\sin\theta \\
\sin\theta & r\cos\theta 
\end{bmatrix} 
$$

将 `J` 和 $\Sigma_{r\theta}$ 代入误差传播公式，经过矩阵运算，可以得到二维点坐标 $(x, y)$ 的协方差矩阵 $\Sigma_{xy}$ 的最终表达式。

这个矩阵描述了转换后点云在x和y方向上的方差 $\sigma_x^2, \sigma_y^2$ 以及它们之间的协方差 $\operatorname{Cov}(x,y)$。

下表总结了 $\Sigma_{xy}$ 中各元素的具体表达式及其物理意义：

| 协方差矩阵元素 | 数学表达式 | 物理意义 |
| - | - | - |
| $\sigma_x^2$ | $\cos^2\theta \cdot \sigma_r^2 + (r\sin\theta)^2 \cdot \sigma_\theta^2$ | 点在x方向上的位置不确定性 |
| $\sigma_y^2$ | $\sin^2\theta \cdot \sigma_r^2 + (r\cos\theta)^2 \cdot \sigma_\theta^2$ | 点在y方向上的位置不确定性 |
| $\operatorname{Cov}(x, y)$ | $[\sin\theta \cos\theta \cdot \sigma_r^2] - [r^2 \sin\theta \cos\theta \cdot \sigma_\theta^2]$ | x和y方向误差的关联性 |

从极坐标到笛卡尔坐标的转换基础公式是  $x = r \cos\theta, y = r \sin\theta$  

#### 重要结论与物理意义

从上面的公式可以得出几个关键结论：

1.  不确定性随距离增大：二维点位置的方差 $\sigma_x^2$  和  $\sigma_y^2$ 中都包含一项 $r^2 \sigma_\theta^2$ 。这意味着，点离雷达越远（r越大），角度测量误差  $\sigma_\theta$  对最终位置不确定性的贡献就越大，且是平方倍增长。这是影响精度的主要因素。
2.  误差椭圆：转换后的点云不确定性通常不再是在x和y方向上对称的。在雷达的径向（远离雷达的方向），不确定性主要受  $\sigma_r$  影响；在切向（垂直于径向的方向），不确定性则主要由  $r \sigma_\theta$  主导。这形成了一个误差椭圆，其方向和大小随点的位置而变化。
3.  误差相关性：协方差 $\operatorname{Cov}(x, y)$ 通常不为零，说明x和y方向上的误差是相关的。在进行后续处理（如卡尔曼滤波）时，使用这个完整的协方差矩阵能比简单地使用独立的方差值获得更准确的结果。

### DBSCAN 聚类点云簇

### 使用 Kd-tree 构建每一帧点云

### 根据时序差分计算每个点云速度矢量

### 根据速度矢量区分动静点云簇

## 时变地图构建

该类根据以下内容初始化

- 栅格分辨率 `thrould` 将实际尺寸在  `thrould` 的范围离散成一个栅格处理
- `lidar` 扫描范围 `r` $m$，构建尺寸 `rxr` $m$ 的栅格地图
- 栅格地图根据 `state = [x,y,theta]` 转换到全局坐标系下(不对地图做旋转处理)

### 时变栅格内容

每一个小栅格，使用一个类表示，主要包含以下内容

- 该栅格是障碍的概率
- 该栅格占用概率的衰减方式


