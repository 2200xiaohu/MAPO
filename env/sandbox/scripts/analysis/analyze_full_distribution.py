import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(".").resolve()
IEDR_FILE = PROJECT_ROOT / "Benchmark/topics/new_data/iedr_batch_results_new.json"
STATS_FILE = PROJECT_ROOT / "runner/benchmark_cases/difficulty_stats.txt"

def analyze_distribution():
    with open(IEDR_FILE, 'r') as f:
        data = json.load(f)
    
    distances = []
    for item in data:
        p0 = item.get('P_0', {})
        dist = p0.get('total', 0)
        if dist > 0:
            distances.append(dist)
            
    distances = np.array(distances)
    mu = np.mean(distances)
    sigma = np.std(distances)
    
    print(f"全量数据统计 (N={len(distances)}):")
    print(f"  均值 (μ): {mu:.2f}")
    print(f"  标准差 (σ): {sigma:.2f}")
    print(f"  最小值: {distances.min():.2f}")
    print(f"  最大值: {distances.max():.2f}")
    
    # Define Thresholds based on Sigma
    # Option 1: Quartiles? No, user asked for Sigma.
    # Let's try a standard spread:
    # Easy: < μ - σ
    # Medium: μ - σ to μ
    # Hard: μ to μ + σ
    # Extreme: > μ + σ
    
    th_easy = mu - sigma
    th_hard = mu + sigma
    th_extreme = mu + 1.5 * sigma # Let's see where this lands
    
    print("\n建议阈值 (基于标准差):")
    print(f"  较易 (< μ-σ): < {th_easy:.2f}")
    print(f"  中等 (μ-σ ~ μ): {th_easy:.2f} - {mu:.2f}")
    print(f"  困难 (μ ~ μ+σ): {mu:.2f} - {th_hard:.2f}")
    print(f"  极难 (> μ+σ): > {th_hard:.2f}")
    
    # Count in Full Dataset
    c_easy = np.sum(distances < th_easy)
    c_med = np.sum((distances >= th_easy) & (distances < mu))
    c_hard = np.sum((distances >= mu) & (distances < th_hard))
    c_extreme = np.sum(distances >= th_hard)
    
    print("\n全量数据分布预览:")
    print(f"  较易: {c_easy} ({c_easy/len(distances):.1%})")
    print(f"  中等: {c_med} ({c_med/len(distances):.1%})")
    print(f"  困难: {c_hard} ({c_hard/len(distances):.1%})")
    print(f"  极难: {c_extreme} ({c_extreme/len(distances):.1%})")
    
    # Generate Histogram for visual confirmation
    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(mu, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mu:.2f}')
    plt.axvline(mu-sigma, color='green', linestyle='dashed', linewidth=1, label=f'-1σ: {mu-sigma:.2f}')
    plt.axvline(mu+sigma, color='orange', linestyle='dashed', linewidth=1, label=f'+1σ: {mu+sigma:.2f}')
    plt.legend()
    plt.title('IEDR Distance Distribution')
    plt.xlabel('Total Deficit Distance')
    plt.ylabel('Count')
    plt.savefig('runner/benchmark_cases/full_distribution_hist.png')
    print("\n分布直方图已保存: runner/benchmark_cases/full_distribution_hist.png")
    
    # Return these thresholds for the next script to use
    return th_easy, mu, th_hard

if __name__ == "__main__":
    analyze_distribution()

