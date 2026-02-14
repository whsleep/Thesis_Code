import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ================= 路径配置 =================
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
EXPERIMENT_ROOT = PROJECT_ROOT / "experiment_results"
INPUT_FILE = EXPERIMENT_ROOT / "performance" / "metrics_summary.csv"
OUTPUT_DIR = EXPERIMENT_ROOT / "performance" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# ===========================================

# 配色方案
COLORS = {
    'accdwa': '#E63946',   # 红色
    'dwa': '#F4A261',      # 橙色  
    'mppi': '#2A9D8F',     # 青色
    'teb': '#264653',      # 深蓝
    'rda': '#9B5DE5'       # 紫色
}

def load_data():
    """加载数据"""
    df = pd.read_csv(INPUT_FILE)
    df['color'] = df['Algorithm'].map(COLORS)
    return df

def robust_normalize(series, invert=False, target_range=(0.2, 0.9)):
    """
    稳健归一化：使用均值±2倍标准差作为边界，避免极端压缩
    映射到 target_range 区间，避免触及 0 或 1
    """
    mean = series.mean()
    std = series.std()
    
    # 使用均值±2σ作为有效范围（覆盖95%数据）
    lower_bound = mean - 2 * std
    upper_bound = mean + 2 * std
    
    # 裁剪异常值
    clipped = series.clip(lower=lower_bound, upper=upper_bound)
    
    # 线性映射到 [0, 1]
    if upper_bound == lower_bound:
        normalized = pd.Series([0.5] * len(series))
    else:
        normalized = (clipped - lower_bound) / (upper_bound - lower_bound)
    
    # 反转（越小越好）
    if invert:
        normalized = 1 - normalized
    
    # 映射到目标区间 [0.2, 0.9]，避免边缘压缩
    min_target, max_target = target_range
    normalized = normalized * (max_target - min_target) + min_target
    
    return normalized

def plot_stacked_bar(df):
    """
    图1: 比率堆叠柱状图 (Success/Collision/Timeout)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    algorithms = df['Algorithm'].tolist()
    x_pos = np.arange(len(algorithms))
    
    success = df['Success_Rate'] * 100
    collision = df['Collision_Rate'] * 100
    timeout = df['Timeout_Rate'] * 100
    
    ax.bar(x_pos, success, label='Success', color='#2E7D32', alpha=0.9, edgecolor='white', linewidth=1)
    ax.bar(x_pos, collision, bottom=success, label='Collision', color='#C62828', alpha=0.9, edgecolor='white', linewidth=1)
    ax.bar(x_pos, timeout, bottom=success + collision, label='Timeout', color='#F9A825', alpha=0.9, edgecolor='white', linewidth=1)
    
    for i, (s, c, t) in enumerate(zip(success, collision, timeout)):
        if s > 5:
            ax.text(i, s/2, f'{s:.1f}%', ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        if c > 5:
            ax.text(i, s + c/2, f'{c:.1f}%', ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        if t > 5:
            ax.text(i, s + c + t/2, f'{t:.1f}%', ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Algorithm Performance: Success vs Failure Rates', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([a.upper() for a in algorithms], fontsize=11, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'stacked_bar_rates.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(OUTPUT_DIR / 'stacked_bar_rates.pdf', bbox_inches='tight', facecolor='white')
    print("✓ Saved: stacked_bar_rates.png/pdf")
    plt.close()

def plot_radar_chart(df):
    """
    图2: 多维度雷达图（改进归一化，避免极端压缩）
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='polar')
    
    categories = ['Success Rate', 'Time Efficiency', 'Path Efficiency', 'Smoothness', 'Compute Efficiency']
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    # 使用稳健归一化
    norm_data = pd.DataFrame()
    norm_data['Success_Rate'] = robust_normalize(df['Success_Rate'], invert=False)
    norm_data['Time_Eff'] = robust_normalize(df['Time_Mean'], invert=True)
    norm_data['Length_Eff'] = robust_normalize(df['Length_Mean'], invert=True)
    norm_data['Smooth_Eff'] = robust_normalize(df['Smoothness_Mean'], invert=True)
    norm_data['Compute_Eff'] = robust_normalize(df['ComputeTime_Mean'], invert=True)
    
    # 打印归一化后的值供检查
    print("\nNormalized values for radar chart:")
    print(norm_data.round(3))
    
    # 绘制每个算法
    for idx, row in df.iterrows():
        algo = row['Algorithm']
        values = [
            norm_data.iloc[idx]['Success_Rate'],
            norm_data.iloc[idx]['Time_Eff'],
            norm_data.iloc[idx]['Length_Eff'],
            norm_data.iloc[idx]['Smooth_Eff'],
            norm_data.iloc[idx]['Compute_Eff']
        ]
        values += values[:1]
        
        color = COLORS.get(algo, '#333333')
        ax.plot(angles, values, 'o-', linewidth=2.5, label=algo.upper(), color=color, markersize=8)
        ax.fill(angles, values, alpha=0.15, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=9, color='gray')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    
    plt.title('Multi-Dimensional Performance Radar\n(Robust Normalization: Higher is Better)', 
              fontsize=14, fontweight='bold', pad=30)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'radar_chart.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(OUTPUT_DIR / 'radar_chart.pdf', bbox_inches='tight', facecolor='white')
    print("✓ Saved: radar_chart.png/pdf")
    plt.close()

def plot_scatter_tradeoff(df):
    """
    图3: 成功率-效率权衡散点图
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for idx, row in df.iterrows():
        algo = row['Algorithm']
        x = row['Success_Rate'] * 100
        y = row['ComputeTime_Mean']
        color = COLORS.get(algo, '#333333')
        
        ax.scatter(x, y, s=400, c=color, alpha=0.8, edgecolors='white', linewidth=2, zorder=5)
        
        ax.annotate(algo.upper(), (x, y), xytext=(5, 5), 
                   textcoords='offset points', fontsize=11, fontweight='bold',
                   ha='left', va='bottom', color=color)
    
    ax.annotate('Ideal Region\n(High Success, Low Compute Time)', 
               xy=(85, df['ComputeTime_Mean'].min() * 1.1), 
               fontsize=10, style='italic', color='green', alpha=0.7,
               ha='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.2))
    
    ax.annotate('', xy=(90, df['ComputeTime_Mean'].min() * 1.2), 
               xytext=(60, df['ComputeTime_Mean'].max() * 0.8),
               arrowprops=dict(arrowstyle='->', color='green', lw=2, alpha=0.5))
    
    ax.set_xlabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Compute Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Success Rate vs Compute Time Trade-off', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    x_margin = (df['Success_Rate'].max() - df['Success_Rate'].min()) * 0.1 * 100
    y_margin = (df['ComputeTime_Mean'].max() - df['ComputeTime_Mean'].min()) * 0.1
    ax.set_xlim(df['Success_Rate'].min()*100 - x_margin, min(100, df['Success_Rate'].max()*100 + x_margin*2))
    ax.set_ylim(df['ComputeTime_Mean'].min() * 0.8, df['ComputeTime_Mean'].max() * 1.2)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'scatter_tradeoff.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(OUTPUT_DIR / 'scatter_tradeoff.pdf', bbox_inches='tight', facecolor='white')
    print("✓ Saved: scatter_tradeoff.png/pdf")
    plt.close()

def main():
    print(f"Loading data from: {INPUT_FILE}")
    
    if not INPUT_FILE.exists():
        print(f"Error: Input file not found: {INPUT_FILE}")
        return
    
    df = load_data()
    print(f"Loaded data for: {', '.join(df['Algorithm'].tolist())}")
    
    print("\nGenerating plots...")
    plot_stacked_bar(df)
    plot_radar_chart(df)
    plot_scatter_tradeoff(df)
    
    print(f"\nAll plots saved to: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()