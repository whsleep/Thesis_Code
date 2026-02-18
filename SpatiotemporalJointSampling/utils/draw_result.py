import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ================= 路径配置 =================
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
EXPERIMENT_ROOT = PROJECT_ROOT / "experiment_results_5"
INPUT_FILE = EXPERIMENT_ROOT / "performance" / "metrics_summary_5.csv"
OUTPUT_DIR = EXPERIMENT_ROOT / "performance" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# ===========================================

# 配色方案
COLORS = {
    'accdwa': '#E63946',   # 红色
    'dwa': '#F4A261',      # 橙色  
    'mppi': '#2A9D8F',     # 青色
    'teb': '#264653',      # 深蓝
    'rda': '#9B5DE5',      # 紫色
    'mpccbf': '#0077B6'     # 蓝色
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
    图1: 堆叠柱状图 - 展示各算法的成功率、碰撞率和超时率
    
    设计思路:
    - 使用堆叠柱状图直观展示三种结果的占比关系
    - 绿色=成功, 红色=碰撞, 黄色=超时
    - 添加数据标签和渐变效果增强可读性
    """
    # 创建画布，使用白色背景
    fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
    ax.set_facecolor('white')
    
    # 准备数据
    algorithms = df['Algorithm'].tolist()
    x_pos = np.arange(len(algorithms))
    bar_width = 0.6  # 柱状图宽度
    
    # 计算百分比（假设输入是0-1的小数）
    success = df['Success_Rate'] * 100
    collision = df['Collision_Rate'] * 100
    timeout = df['Timeout_Rate'] * 100
    
    # ==================== 绘制堆叠柱状图 ====================
    # 第一层：成功率（绿色，底部）
    bars1 = ax.bar(x_pos, success, 
                   width=bar_width,
                   label='Success', 
                   color='#2E7D32',      # 深绿色，代表成功
                   alpha=0.95, 
                   edgecolor='white',     # 白色边框分隔
                   linewidth=1.5,
                   zorder=3)              # 确保在网格线上方
    
    # 第二层：碰撞率（红色，中间）
    bars2 = ax.bar(x_pos, collision, 
                   width=bar_width,
                   bottom=success,        # 堆叠在成功率之上
                   label='Collision', 
                   color='#C62828',      # 深红色，代表危险/碰撞
                   alpha=0.95, 
                   edgecolor='white',
                   linewidth=1.5,
                   zorder=3)
    
    # 第三层：超时率（黄色/橙色，顶部）
    bars3 = ax.bar(x_pos, timeout, 
                   width=bar_width,
                   bottom=success + collision,  # 堆叠在前两层之上
                   label='Timeout', 
                   color='#F57F17',      # 深橙色，代表警告/超时
                   alpha=0.95, 
                   edgecolor='white',
                   linewidth=1.5,
                   zorder=3)
    
    # ==================== 添加数据标签 ====================
    # 在柱子内部显示百分比，只有当占比足够大时才显示（避免重叠）
    for i, (s, c, t) in enumerate(zip(success, collision, timeout)):
        # 成功率标签（在绿色区域中间）
        if s > 8:  # 只有当占比>8%时才显示，避免拥挤
            ax.text(i, s/2, f'{s:.1f}%', 
                   ha='center', va='center', 
                   fontsize=11, fontweight='bold', 
                   color='white',  # 白色文字在深色背景上更清晰
                   zorder=5)
        
        # 碰撞率标签（在红色区域中间）
        if c > 8:
            ax.text(i, s + c/2, f'{c:.1f}%', 
                   ha='center', va='center', 
                   fontsize=11, fontweight='bold', 
                   color='white',
                   zorder=5)
        
        # 超时率标签（在橙色区域中间）
        if t > 8:
            ax.text(i, s + c + t/2, f'{t:.1f}%', 
                   ha='center', va='center', 
                   fontsize=11, fontweight='bold', 
                   color='white',
                   zorder=5)
        
        # 对于很小的 segment，在柱子顶部显示数值
        total = s + c + t
        if s <= 8 and s > 0:
            ax.text(i, s/2, f'{s:.0f}', ha='center', va='center', fontsize=9, color='white', alpha=0.9)
        if c <= 8 and c > 0:
            ax.text(i, s + c/2, f'{c:.0f}', ha='center', va='center', fontsize=9, color='white', alpha=0.9)
        if t <= 8 and t > 0:
            ax.text(i, s + c + t/2, f'{t:.0f}', ha='center', va='center', fontsize=9, color='white', alpha=0.9)
    
    # ==================== 坐标轴和标题设置 ====================
    # Y轴：百分比
    ax.set_ylabel('Percentage (%)', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_ylim(0, 105)  # 稍微超过100，给顶部留空间
    ax.set_yticks(np.arange(0, 101, 20))
    ax.tick_params(axis='y', labelsize=11)
    
    # X轴：算法名称
    ax.set_xlabel('Planning Algorithm', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_xticks(x_pos)
    # 算法名大写显示，增加可读性
    ax.set_xticklabels([a.upper() for a in algorithms], 
                       fontsize=12, fontweight='bold', rotation=0)
    ax.tick_params(axis='x', length=0)  # 隐藏X轴刻度线，更简洁
    
    # 标题
    ax.set_title('Algorithm Performance Overview: Success vs Failure Breakdown', 
                 fontsize=15, fontweight='bold', pad=20, color='#1a1a1a')
    
    # 添加副标题说明
    ax.text(0.5, 1.02, 'Higher success rate indicates better navigation reliability', 
            transform=ax.transAxes, ha='center', va='bottom',
            fontsize=10, style='italic', color='gray')
    
    # ==================== 图例和网格 ====================
    # 图例：放在右上角，带半透明背景
    legend = ax.legend(loc='upper right', 
                      frameon=True, 
                      fancybox=True,
                      shadow=True,
                      fontsize=11,
                      framealpha=0.95)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('gray')
    
    # 水平网格线：虚线，浅色，帮助读数
    ax.grid(axis='y', alpha=0.4, linestyle='--', linewidth=0.8, color='gray', zorder=0)
    ax.set_axisbelow(True)  # 网格线在柱子下方
    
    # 添加100%参考线（理想情况）
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.3, linewidth=1, zorder=1)
    ax.text(len(algorithms)-0.5, 101, 'Target: 100%', ha='right', va='bottom', 
            fontsize=9, color='green', alpha=0.6, style='italic')
    
    # ==================== 美化边框 ====================
    # 移除顶部和右侧边框（spines），更现代
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('gray')
    ax.spines['bottom'].set_color('gray')
    ax.spines['left'].linewidth = 1.2
    ax.spines['bottom'].linewidth = 1.2
    
    # ==================== 保存和输出 ====================
    plt.tight_layout()
    
    # 保存为PNG（高分辨率）和PDF（矢量图）
    output_path_png = OUTPUT_DIR / 'stacked_bar_rates.png'
    output_path_pdf = OUTPUT_DIR / 'stacked_bar_rates.pdf'
    
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white', edgecolor='none')
    
    print(f"✓ Saved: {output_path_png.name}")
    print(f"✓ Saved: {output_path_pdf.name}")
    
    plt.close(fig)  # 关闭图形，释放内存

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