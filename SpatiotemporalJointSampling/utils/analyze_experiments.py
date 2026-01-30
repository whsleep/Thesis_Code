import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path

# ================= 路径配置 =================
# 1. 获取当前脚本所在的绝对路径
CURRENT_DIR = Path(__file__).resolve().parent

# 2. 获取父级目录
PROJECT_ROOT = CURRENT_DIR.parent

# 3. 定位实验数据目录
EXPERIMENT_ROOT = PROJECT_ROOT / "experiment_results"

# 4. 定义输出目录
OUTPUT_DIR = EXPERIMENT_ROOT / "performance"

# 定义要分析的算法
ALGORITHMS = ['accdwa', 'dwa', 'mppi']
# ===========================================

def calculate_path_length(df):
    """计算轨迹总长度"""
    if len(df) < 2:
        return 0.0
    x = df['pos_x'].values
    y = df['pos_y'].values
    dx = np.diff(x)
    dy = np.diff(y)
    distances = np.sqrt(dx**2 + dy**2)
    return np.sum(distances)

def calculate_smoothness(df):
    """
    计算轨迹平滑度 (E_jerk)
    定义: Integral of square jerk (Jerk平方的积分)
    公式: sum(jerk^2 * dt)
    注意: 这里计算的是总量，没有除以步长
    """
    if len(df) < 5:
        return np.nan
    
    points = df[['pos_x', 'pos_y']].values
    
    # 自动计算采样时间 dt
    dt_array = np.diff(df['step_time'].values)
    valid_dt = dt_array[dt_array > 0.0001]
    if len(valid_dt) == 0:
        return np.nan
    dt = np.mean(valid_dt)
    
    # 1. 计算三阶导数 (Jerk)
    jerk_vectors = np.diff(points, n=3, axis=0) / (dt ** 3)
    
    # 2. 计算 Jerk 的平方
    squared_jerk_norms = np.sum(jerk_vectors ** 2, axis=1)
    
    # 3. 计算积分 (Sum * dt)
    e_jerk = np.sum(squared_jerk_norms) * dt / len(dt_array)
    
    return e_jerk

def analyze_single_file(filepath):
    """分析单个 CSV 文件"""
    try:
        df = pd.read_csv(filepath)
        if len(df) == 0:
            return None
        
        last_row = df.iloc[-1]
        
        status = 'unknown'
        if last_row.get('goal_reached') == 1:
            status = 'success'
        elif last_row.get('collision') == 1:
            status = 'collision'
        elif last_row.get('timeout') == 1:
            status = 'timeout'
        
        metrics = {
            'status': status,
            'nav_time': last_row['step_time'],
            'path_length': 0.0,
            'smoothness': 0.0
        }
        
        # 仅对成功案例计算详细指标
        if status == 'success':
            metrics['path_length'] = calculate_path_length(df)
            metrics['smoothness'] = calculate_smoothness(df)
            
        return metrics

    except Exception as e:
        print(f"Error processing {filepath.name}: {e}")
        return None

def main():
    print(f"Searching for data in: {EXPERIMENT_ROOT}")
    
    if not EXPERIMENT_ROOT.exists():
        print(f"Error: Directory not found: {EXPERIMENT_ROOT}")
        return

    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    all_summaries = []

    for algo in ALGORITHMS:
        algo_path = EXPERIMENT_ROOT / algo
        print(f"Processing algorithm: {algo}...")
        
        if not algo_path.exists():
            print(f"  Warning: Directory {algo_path} does not exist. Skipping.")
            continue

        csv_files = list(algo_path.glob('*.csv'))
        if not csv_files:
            print(f"  No .csv files found in {algo_path}")
            continue
            
        results = []
        for f in csv_files:
            res = analyze_single_file(f)
            if res:
                results.append(res)
        
        if not results:
            continue
            
        res_df = pd.DataFrame(results)
        total_runs = len(res_df)
        
        # --- 1. 计算比率 ---
        success_count = len(res_df[res_df['status'] == 'success'])
        collision_count = len(res_df[res_df['status'] == 'collision'])
        timeout_count = len(res_df[res_df['status'] == 'timeout'])
        
        success_rate = success_count / total_runs
        collision_rate = collision_count / total_runs
        timeout_rate = timeout_count / total_runs
        
        # --- 2. 统计均值和方差 (仅针对成功数据) ---
        success_df = res_df[res_df['status'] == 'success']
        
        if len(success_df) > 0:
            # 计算均值
            time_mean = success_df['nav_time'].mean()
            len_mean = success_df['path_length'].mean()
            smooth_mean = success_df['smoothness'].mean()
            
            # 计算方差 (样本数需 > 1)
            if len(success_df) > 1:
                time_var = success_df['nav_time'].var()
                len_var = success_df['path_length'].var()
                smooth_var = success_df['smoothness'].var()
            else:
                time_var, len_var, smooth_var = 0.0, 0.0, 0.0
        else:
            time_mean, len_mean, smooth_mean = 0.0, 0.0, 0.0
            time_var, len_var, smooth_var = 0.0, 0.0, 0.0

        # --- 3. 汇总数据 (确保包含方差) ---
        summary = {
            'Algorithm': algo,
            'Total_Runs': total_runs,
            'Success_Rate': success_rate,
            'Collision_Rate': collision_rate,
            'Timeout_Rate': timeout_rate,
            'Time_Mean': time_mean,
            'Time_Var': time_var,       # <--- 已保存方差
            'Length_Mean': len_mean,
            'Length_Var': len_var,      # <--- 已保存方差
            'Smoothness_Mean': smooth_mean,
            'Smoothness_Var': smooth_var # <--- 已保存方差
        }
        all_summaries.append(summary)

    # --- 4. 保存文件 ---
    if all_summaries:
        final_df = pd.DataFrame(all_summaries)
        output_file = OUTPUT_DIR / 'metrics_summary.csv'
        
        # 保存到 CSV
        final_df.to_csv(output_file, index=False, float_format='%.4f')
        print(f"\nAnalysis complete! Results saved to: {output_file}")
        
        # 终端预览 (显式展示方差列)
        print("\n=== Result Preview ===")
        print(final_df.to_string()) # 直接打印所有列，确保你能看到 Variance
    else:
        print("No valid data processed.")

if __name__ == '__main__':
    main()