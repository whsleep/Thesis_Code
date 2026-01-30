import pandas as pd
import datetime
import os
import numpy as np

class SimulationLogger:
    """
    通用仿真记录器
    负责收集每一步的数据并保存为 CSV
    """
    def __init__(self, log_dir="logs"):
        self.data_buffer = []
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
    def log_step(self, step_time, solver_name, compute_time, robot_state, action, collision, goal_reached, timeout):
        """
        记录单步数据
        :param robot_state: [x, y, theta] 或 [x, y, theta, v, w]
        :param action: [v_cmd, w_cmd]
        """
        # 数据清洗：确保 tensor 或 numpy array 被转为标量或列表
        state_flat = robot_state.flatten() if hasattr(robot_state, 'flatten') else robot_state
        action_flat = action.flatten() if hasattr(action, 'flatten') else action
        
        # 兼容不同长度的 state (有的环境包含速度，有的只包含位置)
        # 假设 state 至少包含 [x, y, theta]
        pos_x = float(state_flat[0])
        pos_y = float(state_flat[1])
        heading = float(state_flat[2])
        
        record = {
            "step_time": step_time,
            "solver": solver_name,
            "compute_time_ms": compute_time * 1000,
            "pos_x": pos_x,
            "pos_y": pos_y,
            "heading": heading,
            "cmd_v": float(action_flat[0]),
            "cmd_w": float(action_flat[1]),
            "collision": 1 if collision else 0,
            "goal_reached": 1 if goal_reached else 0,
            "timeout": 1 if timeout else 0,
        }
        self.data_buffer.append(record)

    def save_to_csv(self, filename_prefix="sim_log"):
        """保存缓冲区数据到 CSV"""
        if not self.data_buffer:
            print("[Logger] No data to save.")
            return

        df = pd.DataFrame(self.data_buffer)
        filename = f"{self.log_dir}/{filename_prefix}.csv"
        
        try:
            df.to_csv(filename, index=False)
            print(f"[Logger] Data saved successfully to: {filename}")
            
            # 打印简要性能统计
            avg_time = df["compute_time_ms"].mean()
            max_time = df["compute_time_ms"].max()
            print(f"[Logger] Stats -> Avg Time: {avg_time:.2f}ms, Max Time: {max_time:.2f}ms")
        except Exception as e:
            print(f"[Logger] Failed to save CSV: {e}") 