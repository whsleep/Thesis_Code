### MPPI 算法核心流程概览

MPPI 的核心思想是通过大量随机采样控制序列，并在模型中推演其轨迹，根据轨迹的成本（Cost）计算权重，最后通过加权平均得到最优控制序列。

---

### 1. 初始化与状态准备 (Initialization)

在计算控制量之前，首先需要准备当前的车辆状态和参考路径。

* **代码对应：** `calc_control_input` 方法开头。
* **逻辑：**
1. 接收观测状态 `observed_x` ，并补充初始角速度。
2. 找到参考路径上距离当前位置最近的点 `_get_nearest_waypoint`，用于后续计算路径跟踪误差。



### 2. 生成噪声与采样 (Noise Sampling)

MPPI 通过在上一时刻的控制序列上叠加随机噪声来生成候选控制序列。

* **数学公式：**




其中  是标称控制量， 是噪声， 是采样后的控制量， 是采样索引。
* **代码对应：**
* **生成噪声：** `_calc_epsilon` 方法。
```python
# 生成 [K, T, 2] 的多元正态分布噪声
epsilon = np.random.multivariate_normal(mean, self.Sigma, size=(self.K, self.T))

```


* **生成候选控制：** `calc_control_input` 中的 Step 2。
```python
# 利用性采样 (Exploitation): 在上一时刻最优控制 u_prev 基础上加噪声
v[:num_exploit] = u_expanded[:num_exploit] + epsilon[:num_exploit]
# 探索性采样 (Exploration): 纯噪声 (代码中的变体策略)
v[num_exploit:] = epsilon[num_exploit:]

```





### 3. 动力学模型推演 (Rollout / Forward Simulation)

将生成的  组控制序列输入到车辆运动学模型中，预测未来的状态轨迹。

* **数学公式 (差分模型)：**




* **代码对应：**
* **推演循环：** `_batch_compute_costs` 中的 `for t in range(T):` 循环。
* **运动学模型：** `_F` 方法。
```python
new_x = x_pos + vel * np.cos(theta) * dt
new_y = y_pos + vel * np.sin(theta) * dt
new_theta = theta + omega * dt

```


* **限幅：** 在推演前通过 `_g` 方法对速度和角速度进行物理约束。



### 4. 成本计算 (Trajectory Evaluation)

计算每条采样轨迹的代价值（Cost）。成本越低，该轨迹越优。

* **数学公式：**
总成本  通常由三部分组成：


1. **阶段成本 (Stage Cost) :** 包含路径跟踪误差和障碍物惩罚。
2. **控制成本 (Control Cost):** 惩罚过大的控制量或与先验控制的偏差。
3. **终端成本 (Terminal Cost) :** 惩罚终点误差。


* **代码对应：** `_batch_compute_costs` 方法。
* **阶段成本：** 调用 `_c` 方法。
* *路径误差：* 计算与参考点的距离和角度差 (加权平方和)。
* *障碍物：* 计算到障碍物的距离，使用势场法（距离越近成本越高，小于安全距离给予巨大惩罚）。


* **控制成本：** 代码中计算项为 `self.param_gamma * (u_t @ inv_Sigma @ v_t)`。
```python
temp = u_t @ self.inv_Sigma 
control_cost_term = np.dot(v_t, temp) 
control_cost = self.param_gamma * control_cost_term

```


* **终端成本：** 循环结束后调用 `_phi` 方法。



### 5. 权重计算 (Weight Calculation)

根据成本计算每条轨迹的权重。成本越低的轨迹，权重越大（指数形式）。

* **数学公式：**



其中  是温度参数， 是最小成本（用于数值稳定性）。
* **代码对应：** `_compute_weights` 方法。
```python
rho = np.min(S)
# 减去最小值防止指数爆炸
exp_terms = np.exp((-1.0 / self.param_lambda) * (S - rho))
eta = np.sum(exp_terms)
return exp_terms / eta

```



### 6. 控制量更新 (Control Update)

利用计算出的权重，对噪声进行加权平均，更新控制序列。

* **数学公式：**



或者直接更新控制量：



*注意：此代码采用的是更新扰动量的方式 (`u + w * epsilon`)。*
* **代码对应：** `calc_control_input` 中的 Step 5。
```python
# w: [K], epsilon: [K, T, 2] -> sum(w * eps) -> [T, 2]
w_expanded = w.reshape(-1, 1, 1)
w_epsilon = np.sum(w_expanded * epsilon, axis=0)

# 更新 self.u_prev
self.u_prev += w_epsilon

```



### 7. 平滑与滚动时域 (Smoothing & Receding Horizon)

为了保证控制的连续性，通常会对计算出的控制序列进行平滑，并执行滚动时域操作。

* **平滑：** 代码中使用了 `_moving_average_filter` (滑动平均滤波器) 来平滑 `w_epsilon`，这有助于减少控制抖动。
* **滚动时域 (Shift)：** 执行完当前一步后，将控制序列向前移动一步，末尾复制补全，作为下一帧的初始猜测。
* **代码对应：**
```python
current_u = self.u_prev[0].copy() # 输出给机器人的当前控制量
self.u_prev[:-1] = self.u_prev[1:] # 整体前移
self.u_prev[-1] = self.u_prev[-1]  # 保持最后一位

```
