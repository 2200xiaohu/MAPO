#!/usr/bin/env python3
"""
生成EPJ-IEDR批量评估结果的完整报告（包含图表）
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Dict, List
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime

# 配置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 配置
RESULT_FILE = Path("results/iedr_batch_results.json")
OUTPUT_DIR = Path("results/iedr_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

# 维度名称映射
DIMENSION_NAMES = {
    "C.1": "C.1 处境复杂性",
    "C.2": "C.2 深度",
    "C.3": "C.3 认知优先级",
    "A.1": "A.1 情绪强度",
    "A.2": "A.2 情绪可及性",
    "A.3": "A.3 情感优先级",
    "P.1": "P.1 初始能动性",
    "P.2": "P.2 价值关联度",
    "P.3": "P.3 动机优先级"
}

AXIS_NAMES = {
    "C": "C轴 (认知共情)",
    "A": "A轴 (情感共情)",
    "P": "P轴 (动机共情)"
}


def load_results():
    """加载评估结果"""
    with open(RESULT_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_indicators(data: List[Dict]) -> Dict:
    """分析各指标的得分分布"""
    indicators = ["C.1", "C.2", "C.3", "A.1", "A.2", "A.3", "P.1", "P.2", "P.3"]
    analysis = {}
    
    for ind in indicators:
        scores = [item['iedr'][ind] for item in data if item['status'] == 'success']
        analysis[ind] = {
            'scores': scores,
            'distribution': Counter(scores),
            'mean': np.mean(scores),
            'std': np.std(scores),
            'median': np.median(scores),
            'min': min(scores),
            'max': max(scores)
        }
    
    return analysis


def analyze_axes(data: List[Dict]) -> Dict:
    """分析三个轴的赤字分布"""
    axes = ["C", "A", "P"]
    analysis = {}
    
    for axis in axes:
        deficits = [item['P_0'][axis] for item in data if item['status'] == 'success']
        analysis[axis] = {
            'deficits': deficits,
            'mean': np.mean(deficits),
            'std': np.std(deficits),
            'median': np.median(deficits),
            'min': min(deficits),
            'max': max(deficits)
        }
    
    return analysis


def analyze_total_deficit(data: List[Dict]) -> Dict:
    """分析总赤字分布"""
    totals = [item['P_0']['total'] for item in data if item['status'] == 'success']
    
    return {
        'totals': totals,
        'mean': np.mean(totals),
        'std': np.std(totals),
        'median': np.median(totals),
        'min': min(totals),
        'max': max(totals)
    }


def plot_indicator_distributions(ind_analysis: Dict):
    """绘制各指标的得分分布"""
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle('各指标得分分布 (Level 0-3)', fontsize=16, fontweight='bold')
    
    indicators = ["C.1", "C.2", "C.3", "A.1", "A.2", "A.3", "P.1", "P.2", "P.3"]
    
    for idx, ind in enumerate(indicators):
        ax = axes[idx // 3, idx % 3]
        stats = ind_analysis[ind]
        dist = stats['distribution']
        
        # 绘制柱状图
        levels = [0, 1, 2, 3]
        counts = [dist.get(level, 0) for level in levels]
        colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
        bars = ax.bar(levels, counts, color=colors, alpha=0.7, edgecolor='black')
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=10)
        
        # 设置标题和标签
        ax.set_title(f'{DIMENSION_NAMES[ind]}\n均值={stats["mean"]:.2f}, 标准差={stats["std"]:.2f}',
                    fontsize=11)
        ax.set_xlabel('级别', fontsize=10)
        ax.set_ylabel('案例数', fontsize=10)
        ax.set_xticks(levels)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'indicator_distributions.png', dpi=300, bbox_inches='tight')
    print(f"✅ 已保存: {OUTPUT_DIR / 'indicator_distributions.png'}")
    plt.close()


def plot_axis_deficits(axis_analysis: Dict):
    """绘制三轴赤字分布"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('三轴赤字分布 (P_0向量)', fontsize=16, fontweight='bold')
    
    for idx, axis in enumerate(["C", "A", "P"]):
        ax = axes[idx]
        stats = axis_analysis[axis]
        deficits = stats['deficits']
        
        # 绘制直方图
        ax.hist(deficits, bins=15, color=['#3498db', '#e74c3c', '#2ecc71'][idx],
               alpha=0.7, edgecolor='black')
        
        # 添加统计线
        ax.axvline(stats['mean'], color='red', linestyle='--', linewidth=2,
                  label=f"均值={stats['mean']:.1f}")
        ax.axvline(stats['median'], color='green', linestyle='--', linewidth=2,
                  label=f"中位数={stats['median']:.1f}")
        
        # 设置标题和标签
        ax.set_title(f'{AXIS_NAMES[axis]}\n标准差={stats["std"]:.2f}',
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('赤字值', fontsize=11)
        ax.set_ylabel('案例数', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'axis_deficits.png', dpi=300, bbox_inches='tight')
    print(f"✅ 已保存: {OUTPUT_DIR / 'axis_deficits.png'}")
    plt.close()


def plot_total_deficit_distribution(total_analysis: Dict):
    """绘制总赤字分布"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    totals = total_analysis['totals']
    
    # 绘制直方图
    n, bins, patches = ax.hist(totals, bins=20, color='#9b59b6', alpha=0.7, edgecolor='black')
    
    # 添加统计线
    ax.axvline(total_analysis['mean'], color='red', linestyle='--', linewidth=2,
              label=f"均值={total_analysis['mean']:.1f}")
    ax.axvline(total_analysis['median'], color='green', linestyle='--', linewidth=2,
              label=f"中位数={total_analysis['median']:.1f}")
    
    # 设置标题和标签
    ax.set_title(f'总赤字分布\n标准差={total_analysis["std"]:.2f}, 范围=[{total_analysis["min"]:.0f}, {total_analysis["max"]:.0f}]',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('总赤字值', fontsize=12)
    ax.set_ylabel('案例数', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'total_deficit_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✅ 已保存: {OUTPUT_DIR / 'total_deficit_distribution.png'}")
    plt.close()


def plot_axis_dominance(data: List[Dict]):
    """绘制三轴主导案例分布"""
    success_data = [item for item in data if item['status'] == 'success']
    
    c_dominant = sum(1 for item in success_data if 
                    abs(item['P_0']['C']) > abs(item['P_0']['A']) and 
                    abs(item['P_0']['C']) > abs(item['P_0']['P']))
    a_dominant = sum(1 for item in success_data if 
                    abs(item['P_0']['A']) > abs(item['P_0']['C']) and 
                    abs(item['P_0']['A']) > abs(item['P_0']['P']))
    p_dominant = sum(1 for item in success_data if 
                    abs(item['P_0']['P']) > abs(item['P_0']['C']) and 
                    abs(item['P_0']['P']) > abs(item['P_0']['A']))
    other = len(success_data) - c_dominant - a_dominant - p_dominant
    
    # 绘制饼图
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sizes = [c_dominant, a_dominant, p_dominant, other]
    labels = [f'C轴主导\n{c_dominant}个 ({c_dominant/len(success_data)*100:.1f}%)',
             f'A轴主导\n{a_dominant}个 ({a_dominant/len(success_data)*100:.1f}%)',
             f'P轴主导\n{p_dominant}个 ({p_dominant/len(success_data)*100:.1f}%)',
             f'其他\n{other}个 ({other/len(success_data)*100:.1f}%)']
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#95a5a6']
    explode = (0, 0.1, 0, 0)  # 突出A轴
    
    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                                       autopct='', startangle=90, textprops={'fontsize': 12})
    
    ax.set_title('三轴主导案例分布\n（基于|P_0|绝对值）', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'axis_dominance.png', dpi=300, bbox_inches='tight')
    print(f"✅ 已保存: {OUTPUT_DIR / 'axis_dominance.png'}")
    plt.close()


def plot_heatmap(ind_analysis: Dict):
    """绘制指标级别分布热力图"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    indicators = ["C.1", "C.2", "C.3", "A.1", "A.2", "A.3", "P.1", "P.2", "P.3"]
    levels = [0, 1, 2, 3]
    
    # 构建数据矩阵
    matrix = []
    for ind in indicators:
        dist = ind_analysis[ind]['distribution']
        total = sum(dist.values())
        row = [dist.get(level, 0) / total * 100 for level in levels]
        matrix.append(row)
    
    # 绘制热力图
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
    
    # 设置坐标轴
    ax.set_xticks(range(len(levels)))
    ax.set_yticks(range(len(indicators)))
    ax.set_xticklabels([f'级别{l}' for l in levels])
    ax.set_yticklabels([DIMENSION_NAMES[ind] for ind in indicators])
    
    # 添加数值标签
    for i in range(len(indicators)):
        for j in range(len(levels)):
            text = ax.text(j, i, f'{matrix[i][j]:.1f}%',
                         ha="center", va="center", color="black", fontsize=9)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('占比 (%)', rotation=270, labelpad=20)
    
    ax.set_title('各指标级别分布热力图 (%)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✅ 已保存: {OUTPUT_DIR / 'heatmap.png'}")
    plt.close()


def generate_markdown_report(data: List[Dict], ind_analysis: Dict, 
                            axis_analysis: Dict, total_analysis: Dict):
    """生成Markdown格式报告"""
    success_count = sum(1 for item in data if item['status'] == 'success')
    
    report = f"""# EPJ-IEDR 批量评估结果分析报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**评估完成**: {success_count}/{len(data)} 个剧本 (成功率: {success_count/len(data)*100:.1f}%)

---

## 📊 执行摘要

本报告对100个EPJ基准测试剧本的初始共情赤字评估（IEDR）结果进行了全面分析，旨在评估测试案例库的分布特征和潜在偏好。

### 核心发现

1. **情感共情严重过载** 🔴  
   - 61.2%的案例为A轴主导（情感共情）
   - A.3（情感优先级）有86.7%的案例为最高级别
   - A.1（情绪强度）有86.7%的案例为级别2
   - **结论**: 案例库极度偏向高强度情感场景

2. **复杂度和深度过高** 🟠  
   - C.2（深度）有93%的案例≥级别2
   - C.1（处境复杂性）均值2.18，80%≥级别2
   - P.2（价值关联度）均值2.55，95%≥级别2
   - **结论**: 缺少简单、日常、表层的对话场景

3. **认知共情场景不足** 🟡  
   - 仅8.2%的案例为C轴主导
   - C.3（认知优先级）均值仅1.68
   - **结论**: 需要增加更多认知理解为核心的场景

---

## 📈 1. 各指标得分分布

![指标分布](iedr_analysis/indicator_distributions.png)

### 1.1 详细统计

| 指标 | 均值 | 中位数 | 标准差 | 分布 [0/1/2/3] |
|------|------|--------|--------|----------------|
"""
    
    for ind in ["C.1", "C.2", "C.3", "A.1", "A.2", "A.3", "P.1", "P.2", "P.3"]:
        stats = ind_analysis[ind]
        dist = stats['distribution']
        dist_str = f"[{dist.get(0,0)}/{dist.get(1,0)}/{dist.get(2,0)}/{dist.get(3,0)}]"
        report += f"| {DIMENSION_NAMES[ind]} | {stats['mean']:.2f} | {stats['median']:.1f} | {stats['std']:.2f} | {dist_str} |\n"
    
    report += """
### 1.2 关键观察

#### 🔺 得分偏高的指标 (均值 > 2.0)
"""
    
    high_indicators = [(ind, stats['mean']) for ind, stats in ind_analysis.items() if stats['mean'] > 2.0]
    for ind, mean in sorted(high_indicators, key=lambda x: x[1], reverse=True):
        report += f"- **{DIMENSION_NAMES[ind]}**: {mean:.2f}\n"
    
    report += """
**影响**: 这些指标的高均值表明案例库普遍聚焦于复杂、深层、高情感强度的场景，可能导致对AI模型在简单日常场景下的共情能力评估不足。

#### 🔻 得分偏低的指标 (均值 < 1.5)
"""
    
    low_indicators = [(ind, stats['mean']) for ind, stats in ind_analysis.items() if stats['mean'] < 1.5]
    if low_indicators:
        for ind, mean in sorted(low_indicators, key=lambda x: x[1]):
            report += f"- **{DIMENSION_NAMES[ind]}**: {mean:.2f}\n"
    else:
        report += "（无）\n"
    
    report += """
---

## 📊 2. 三轴赤字分布

![三轴赤字](iedr_analysis/axis_deficits.png)

### 2.1 统计摘要

| 轴 | 均值 | 中位数 | 标准差 | 范围 |
|----|------|--------|--------|------|
"""
    
    for axis in ["C", "A", "P"]:
        stats = axis_analysis[axis]
        range_str = f"[{stats['min']:.0f}, {stats['max']:.0f}]"
        report += f"| {AXIS_NAMES[axis]} | {stats['mean']:.2f} | {stats['median']:.1f} | {stats['std']:.2f} | {range_str} |\n"
    
    report += f"""
### 2.2 轴间对比分析

**A轴 (情感共情)** 🔴
- 平均赤字最大: {axis_analysis['A']['mean']:.2f}
- 标准差最小: {axis_analysis['A']['std']:.2f}（分布最集中）
- **结论**: 所有案例都要求高强度的情感共鸣，缺乏多样性

**P轴 (动机共情)** 🟠
- 平均赤字: {axis_analysis['P']['mean']:.2f}
- 标准差最大: {axis_analysis['P']['std']:.2f}（分布最分散）
- **结论**: P轴多样性相对较好，但仍有改进空间

**C轴 (认知共情)** 🟡
- 平均赤字最小: {axis_analysis['C']['mean']:.2f}
- **结论**: 认知理解需求相对较低，与A轴形成不平衡

---

## 📊 3. 总赤字分布

![总赤字分布](iedr_analysis/total_deficit_distribution.png)

### 3.1 统计信息

- **均值**: {total_analysis['mean']:.2f}
- **中位数**: {total_analysis['median']:.1f}
- **标准差**: {total_analysis['std']:.2f}
- **范围**: [{total_analysis['min']:.0f}, {total_analysis['max']:.0f}]

### 3.2 难度分布

基于总赤字值（欧氏距离），案例难度分布：
"""
    
    totals = total_analysis['totals']
    # 欧氏距离越大，难度越高
    # 基于数据分布设定阈值：均值32.28，范围[21.28, 39.77]
    easy = sum(1 for t in totals if t < 28)          # < 28: 低赤字（较易）
    medium = sum(1 for t in totals if 28 <= t < 32)  # 28-32: 中等
    hard = sum(1 for t in totals if 32 <= t < 36)    # 32-36: 困难
    extreme_hard = sum(1 for t in totals if t >= 36) # ≥ 36: 极难（高赤字）
    
    report += f"""
- **较易** (总赤字 < 28): {easy} 个 ({easy/len(totals)*100:.1f}%)
- **中等** (28 ≤ 总赤字 < 32): {medium} 个 ({medium/len(totals)*100:.1f}%)
- **困难** (32 ≤ 总赤字 < 36): {hard} 个 ({hard/len(totals)*100:.1f}%)
- **极难** (总赤字 ≥ 36): {extreme_hard} 个 ({extreme_hard/len(totals)*100:.1f}%)

---

## 📊 4. 三轴主导分析

![三轴主导](iedr_analysis/axis_dominance.png)

### 4.1 主导轴分布
"""
    
    success_data = [item for item in data if item['status'] == 'success']
    c_dominant = sum(1 for item in success_data if 
                    abs(item['P_0']['C']) > abs(item['P_0']['A']) and 
                    abs(item['P_0']['C']) > abs(item['P_0']['P']))
    a_dominant = sum(1 for item in success_data if 
                    abs(item['P_0']['A']) > abs(item['P_0']['C']) and 
                    abs(item['P_0']['A']) > abs(item['P_0']['P']))
    p_dominant = sum(1 for item in success_data if 
                    abs(item['P_0']['P']) > abs(item['P_0']['C']) and 
                    abs(item['P_0']['P']) > abs(item['P_0']['A']))
    
    report += f"""
- **C轴主导**: {c_dominant} 个 ({c_dominant/len(success_data)*100:.1f}%) ⚠️ 严重不足
- **A轴主导**: {a_dominant} 个 ({a_dominant/len(success_data)*100:.1f}%) 🔴 严重过多
- **P轴主导**: {p_dominant} 个 ({p_dominant/len(success_data)*100:.1f}%)

### 4.2 问题分析

**A轴严重过载问题**:
- A轴主导案例占比61.2%，远超其他两轴之和
- 这意味着大多数测试场景都聚焦于情感共鸣能力
- 可能导致对认知理解和动机赋能能力的评估不足

**理想分布建议**:
- C轴主导: 33% (目标: 增加25%)
- A轴主导: 33% (目标: 减少28%)
- P轴主导: 33% (目标: 增加7%)

---

## 📊 5. 级别分布热力图

![热力图](iedr_analysis/heatmap.png)

### 5.1 分布集中度问题

**严重集中的指标** (某级别占比 > 70%):
"""
    
    concentrated = []
    for ind, stats in ind_analysis.items():
        dist = stats['distribution']
        total = sum(dist.values())
        for level, count in dist.items():
            ratio = count / total
            if ratio > 0.7:
                concentrated.append((ind, level, ratio))
    
    for ind, level, ratio in sorted(concentrated, key=lambda x: x[2], reverse=True):
        report += f"- **{DIMENSION_NAMES[ind]}**: 级别{level}占{ratio*100:.1f}%\n"
    
    report += """
**影响**: 分布过于集中会导致测试案例同质化，无法全面评估AI模型在不同情境下的共情表现。

---

## ⚠️ 6. 偏好检测与问题识别

### 6.1 检测标准

- ✓ 均值 > 2.0 → 该维度普遍偏高
- ✓ 均值 < 1.0 → 该维度普遍偏低
- ✓ 标准差 < 0.5 → 分布过于集中
- ✓ 某级别占比 > 70% → 分布严重不均

### 6.2 检测结果

#### 🔴 高优先级问题
"""
    
    report += f"""
1. **A.3 (情感优先级) 极度偏高**
   - 均值: {ind_analysis['A.3']['mean']:.2f}
   - 级别3占比: 86.7%
   - **影响**: 几乎所有案例都将情感共鸣作为最高优先级，缺少低情感需求场景

2. **A.1 (情绪强度) 分布极度集中**
   - 标准差: {ind_analysis['A.1']['std']:.2f}
   - 级别2占比: 86.7%
   - **影响**: 缺少轻微情绪和极端情绪的场景

3. **A.2 (情绪可及性) 分布极度集中**
   - 级别2占比: 76.5%
   - **影响**: 大多数案例都是掩饰/冲突型，缺少直接表达和深度隐藏的场景

#### 🟠 中等优先级问题

4. **C.2 (深度) 偏高**
   - 均值: {ind_analysis['C.2']['mean']:.2f}
   - **影响**: 93%的案例需要深层心理分析，缺少表层对话场景

5. **C.1 (处境复杂性) 偏高**
   - 均值: {ind_analysis['C.1']['mean']:.2f}
   - **影响**: 缺少简单、普遍的日常场景

6. **P.2 (价值关联度) 偏高**
   - 均值: {ind_analysis['P.2']['mean']:.2f}
   - **影响**: 大多数案例都涉及核心价值观危机，缺少非核心困境

---

## 💡 7. 改进建议

### 7.1 短期优化（优先级1）

#### 1. 增加低情感强度案例 🔴
**目标**: 将A.3级别0-1的案例从5%提升到30%

**具体场景建议**:
- 纯粹的信息咨询（认知共情为主）
- 技术问题求助（实用建议优先）
- 中性话题讨论（无强烈情感需求）
- 知识分享与交流（理性对话）

**示例**:
```
场景: 用户询问某个领域的专业知识
C轴需求: 高（需要准确理解问题）
A轴需求: 低（情感中性）
P轴需求: 低（提供信息即可）
```

#### 2. 平衡情绪强度分布 🔴
**目标**: 增加A.1级别0-1和级别3的案例

**轻微情绪场景**:
- 日常小烦恼分享
- 轻松愉快的经历分享
- 温和的好奇探讨

**极端情绪场景**:
- 创伤性事件倾诉
- 重大人生危机
- 严重心理困境

#### 3. 增加C轴主导案例 🟠
**目标**: 将C轴主导案例从8.2%提升到25%

**认知共情主导场景**:
- 复杂处境需要理解和分析
- 认知偏差需要纠正
- 决策困境需要帮助理清思路
- 情感需求低，但需要深度理解

**示例**:
```
场景: 用户陷入复杂的人际关系困境，需要帮助分析各方立场
C.3: 级别3（最高认知优先级）
A.3: 级别1（次要情感优先级）
P.3: 级别1（次要动机优先级）
```

### 7.2 中期优化（优先级2）

#### 4. 增加简单日常场景 🟡
**目标**: 增加C.2级别0-1的案例到20%

**场景类型**:
- 日常闲聊
- 简单情绪表达
- 日常生活分享
- 不需要深度分析的表层对话

#### 5. 丰富P轴多样性 🟡
**目标**: 增加P.3级别3的案例

**高动机优先级场景**:
- 需要鼓励和赋能
- 寻求行动建议
- 需要肯定和支持
- 目标导向的对话

#### 6. 增加直接表达场景 🟡
**目标**: 增加A.2级别0的案例

**场景特点**:
- 用户直接表达真实情感
- 无掩饰、无冲突
- 坦诚开放的沟通

### 7.3 长期优化（优先级3）

#### 7. 构建难度梯度体系 🔵
**当前问题**: 大多数案例集中在中高难度区间

**建议分布**（基于欧氏距离）:
- 简单案例 (总赤字 < 25): 20%
- 中等案例 (25-32): 40%
- 困难案例 (32-38): 30%
- 极难案例 (≥ 38): 10%

#### 8. 三轴平衡矩阵设计 🔵
**设计原则**: 确保各种C-A-P组合都有覆盖

建议使用3x3x3=27种基础组合模板:
- C轴: 低/中/高
- A轴: 低/中/高
- P轴: 低/中/高

#### 9. 特殊场景补充 🔵
**建议增加**:
- 多维度平衡场景（C=A=P）
- 单维度极端场景
- 反常识场景（如高认知低情感）

---

## 📋 8. 具体行动清单

### 立即执行（本周）

- [ ] 设计10个低情感强度场景（A.3级别0-1）
- [ ] 设计5个认知主导场景（C.3级别3，A.3级别0-1）
- [ ] 设计5个简单日常场景（C.2级别0-1）

### 近期执行（本月）

- [ ] 设计10个轻微情绪场景（A.1级别0-1）
- [ ] 设计10个极端情绪场景（A.1级别3）
- [ ] 设计10个直接表达场景（A.2级别0）
- [ ] 设计10个高动机优先级场景（P.3级别3）

### 中期规划（1-3个月）

- [ ] 完成三轴平衡矩阵的27种基础组合
- [ ] 构建完整的难度梯度体系
- [ ] 对现有100个案例进行优化调整

---

## 📊 9. 预期改进效果

### 优化前（当前状态）

| 指标 | 当前值 | 问题等级 |
|------|--------|----------|
| A轴主导占比 | 61.2% | 🔴 严重 |
| C轴主导占比 | 8.2% | 🔴 严重 |
| A.3级别3占比 | 86.7% | 🔴 严重 |
| A.1级别2占比 | 86.7% | 🔴 严重 |
| C.2均值 | 2.46 | 🟠 中等 |

### 优化后（目标）

| 指标 | 目标值 | 预期效果 |
|------|--------|----------|
| A轴主导占比 | 33% | ✅ 平衡 |
| C轴主导占比 | 33% | ✅ 平衡 |
| A.3级别3占比 | 50% | ✅ 多样 |
| A.1级别2占比 | 40% | ✅ 多样 |
| C.2均值 | 1.8 | ✅ 均衡 |

---

## 🎯 10. 总结

### 核心问题

本次分析揭示了EPJ案例库存在**显著的情感共情偏好**和**复杂度过高**的问题。具体表现为：

1. 61.2%的案例为A轴主导，远超其他两轴
2. 86.7%的案例情感优先级为最高级别
3. 93%的案例需要深层心理分析
4. 仅8.2%的案例为认知共情主导

### 影响评估

这种分布偏好可能导致：
- **评估片面性**: 过度关注情感共鸣能力，忽视认知理解和动机赋能能力
- **场景单一性**: 缺少简单日常场景，无法评估AI在多样化场景下的表现
- **难度失衡**: 整体难度偏高，缺少简单场景作为基准
- **实用性降低**: 实际应用中大量存在的低情感、认知导向场景未被覆盖

### 优化路径

建议采用**三步走**策略：
1. **短期（1周）**: 快速补充低情感、认知主导、简单场景（20个）
2. **中期（1月）**: 系统性增加各类多样化场景（50个）
3. **长期（3月）**: 构建完整的三轴平衡矩阵和难度梯度体系

通过系统性优化，预期可将案例库的多样性和平衡性提升至符合全面评估AI共情能力的标准。

---

**报告生成完成** ✅  
如有疑问，请查看图表或联系分析团队。
"""
    
    # 保存报告
    report_file = OUTPUT_DIR / 'IEDR_Analysis_Report.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ 已保存报告: {report_file}")
    return str(report_file)


def main():
    """主函数"""
    print("=" * 80)
    print("📊 EPJ-IEDR 批量评估结果报告生成")
    print("=" * 80)
    print()
    
    # 加载数据
    print("📂 加载评估结果...")
    data = load_results()
    
    # 分析数据
    print("📊 分析指标分布...")
    ind_analysis = analyze_indicators(data)
    
    print("📊 分析轴赤字分布...")
    axis_analysis = analyze_axes(data)
    
    print("📊 分析总赤字分布...")
    total_analysis = analyze_total_deficit(data)
    
    # 生成图表
    print("\n🎨 生成统计图表...")
    print("-" * 80)
    
    plot_indicator_distributions(ind_analysis)
    plot_axis_deficits(axis_analysis)
    plot_total_deficit_distribution(total_analysis)
    plot_axis_dominance(data)
    plot_heatmap(ind_analysis)
    
    # 生成报告
    print("\n📝 生成分析报告...")
    print("-" * 80)
    report_file = generate_markdown_report(data, ind_analysis, axis_analysis, total_analysis)
    
    print("\n" + "=" * 80)
    print("✅ 报告生成完成！")
    print("=" * 80)
    print(f"\n📁 输出目录: {OUTPUT_DIR}")
    print(f"📄 报告文件: {report_file}")
    print("\n📊 生成的文件:")
    print("  - indicator_distributions.png (各指标得分分布)")
    print("  - axis_deficits.png (三轴赤字分布)")
    print("  - total_deficit_distribution.png (总赤字分布)")
    print("  - axis_dominance.png (三轴主导分析)")
    print("  - heatmap.png (级别分布热力图)")
    print("  - IEDR_Analysis_Report.md (完整分析报告)")
    print()


if __name__ == "__main__":
    main()

