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

### 点云转换全局点云

### DBSCAN 聚类点云簇

### 使用 Kd-tree 构建每一帧点云

### 根据时序差分计算每个点云速度矢量

### 根据速度矢量区分动静态点云

## 