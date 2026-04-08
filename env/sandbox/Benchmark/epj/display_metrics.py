# Benchmark/epj/display_metrics.py
"""
EPJ显示指标（仅供参考，不用于决策）

重要说明：
- 这些指标仅用于UI显示、论文报告等
- ！！！绝不能用于STOP/CONTINUE的决策判断！！！
- 决策判断必须且只能使用"Epsilon解决区域"逻辑
"""

from typing import Tuple


def calculate_display_progress(P_t: Tuple[int, int, int], 
                               P_0: Tuple[int, int, int]) -> float:
    """
    计算等效进度分数（0-100）- 仅供显示
    
    使用曼哈顿距离（Manhattan Distance）而非欧几里得距离
    以避免"多轴惩罚"问题
    
    警告：
    - 此分数仅供参考显示
    - 不能用于STOP/CONTINUE决策
    - 仍然无法完全解决"过冲"问题
    
    Args:
        P_t: 当前位置向量 (C, A, P)
        P_0: 初始赤字向量 (C, A, P)
    
    Returns:
        float: 等效进度分数 (0-100)，仅供显示
    """
    # 计算初始总赤字（曼哈顿距离）
    initial_total_deficit = abs(P_0[0]) + abs(P_0[1]) + abs(P_0[2])
    
    if initial_total_deficit == 0:
        # 初始就没有赤字，视为100%
        return 100.0
    
    # 计算当前总赤字（曼哈顿距离）
    current_total_deficit = abs(P_t[0]) + abs(P_t[1]) + abs(P_t[2])
    
    # 计算赤字缩减比例
    deficit_reduced = initial_total_deficit - current_total_deficit
    progress_ratio = deficit_reduced / initial_total_deficit
    
    # 转换为0-100分数
    progress_score = progress_ratio * 100
    
    # 限制在0-100范围（但可能为负，说明情况恶化）
    progress_score = max(0, min(100, progress_score))
    
    return progress_score


def get_progress_description(progress_score: float) -> str:
    """
    根据进度分数给出描述性文本
    
    Args:
        progress_score: 进度分数 (0-100)
    
    Returns:
        str: 描述文本
    """
    if progress_score >= 90:
        return "接近完成"
    elif progress_score >= 60:
        return "对话深入"
    elif progress_score >= 30:
        return "对话中期"
    else:
        return "对话初期"


def calculate_dimensional_progress(P_t: Tuple[int, int, int], 
                                   P_0: Tuple[int, int, int]) -> dict:
    """
    计算各维度的进度（仅供显示）
    
    Args:
        P_t: 当前位置
        P_0: 初始赤字
    
    Returns:
        dict: {
            "C_progress": float (0-100),
            "A_progress": float (0-100),
            "P_progress": float (0-100)
        }
    """
    def calc_axis_progress(initial, current):
        if initial == 0:
            return 100.0
        # 计算该轴的改善比例
        improvement = abs(initial) - abs(current)
        ratio = improvement / abs(initial)
        return max(0, min(100, ratio * 100))
    
    return {
        "C_progress": calc_axis_progress(P_0[0], P_t[0]),
        "A_progress": calc_axis_progress(P_0[1], P_t[1]),
        "P_progress": calc_axis_progress(P_0[2], P_t[2])
    }


# ═══════════════════════════════════════════════════════════════
# 重要说明和警告
# ═══════════════════════════════════════════════════════════════

IMPORTANT_WARNINGS = """
⚠️⚠️⚠️ 重要警告 ⚠️⚠️⚠️

本模块提供的所有"进度分数"（0-100）都仅供显示参考，
绝不能用于STOP/CONTINUE的决策判断！

决策判断必须且只能使用"Epsilon解决区域"逻辑：
  is_in_zone = (abs(P_t.C) <= ε) AND (abs(P_t.A) <= ε) AND (abs(P_t.P) <= ε)

原因：
1. 任何基于距离的"进度分数"都无法正确处理"过冲"问题
2. 欧几里得距离会产生"多轴惩罚"
3. 即使曼哈顿距离也只是"较好"的显示指标，不是科学的决策标准

用途：
  ✅ 可以用于：UI显示、论文报告、可视化图表
  ❌ 不能用于：STOP/CONTINUE决策、进度阈值判断

唯一的科学决策标准：Epsilon解决区域检测
"""


# ═══════════════════════════════════════════════════════════════
# 测试示例
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("EPJ 显示指标测试")
    print("=" * 70)
    print(IMPORTANT_WARNINGS)
    
    # 测试案例1: 正常进展
    print("\n" + "=" * 70)
    print("案例1: 正常进展")
    print("=" * 70)
    
    P_0 = (-10, -17, -25)
    print(f"初始赤字 P_0 = {P_0}")
    print(f"初始总赤字（曼哈顿）= {abs(P_0[0]) + abs(P_0[1]) + abs(P_0[2])}")
    
    test_positions = [
        (-9, -16, -25),   # T=3
        (-6, -13, -24),   # T=6
        (-3, -10, -21),   # T=9
        (-2, -7, -18),    # T=12
        (-1, -4, -15),    # T=15
        (0, 0, 0),        # 完美达标
    ]
    
    print(f"\n进展轨迹（曼哈顿距离法）:")
    print(f"{'P_t':<20} {'总赤字':>8} {'进度分数':>10} {'描述':>10}")
    print("-" * 60)
    
    for P_t in test_positions:
        score = calculate_display_progress(P_t, P_0)
        desc = get_progress_description(score)
        deficit = abs(P_t[0]) + abs(P_t[1]) + abs(P_t[2])
        print(f"{str(P_t):<20} {deficit:>8} {score:>9.1f}% {desc:>10}")
    
    # 测试案例2: 过冲问题
    print("\n" + "=" * 70)
    print("案例2: 过冲问题演示")
    print("=" * 70)
    
    P_0_small = (-1, 0, 0)
    print(f"初始赤字 P_0 = {P_0_small}")
    print(f"模型表现极好，提供 v_t = (+2, 0, 0)")
    
    P_t_overshoot = (1, 0, 0)  # 过冲了
    score = calculate_display_progress(P_t_overshoot, P_0_small)
    
    print(f"当前位置 P_t = {P_t_overshoot}")
    print(f"等效进度分数 = {score:.1f}%")
    
    print(f"\n⚠️ 注意：尽管模型表现出色，但因为过冲，")
    print(f"   曼哈顿距离从1变成1，进度显示为0%")
    print(f"\n✅ 但EPJ的Epsilon检测会正确判断：")
    print(f"   abs(1) <= ε(1.0) → True → SUCCESS ✅")
    
    # 测试案例3: 多轴问题
    print("\n" + "=" * 70)
    print("案例3: 多轴惩罚（曼哈顿距离已解决）")
    print("=" * 70)
    
    P_0_single = (-10, 0, 0)
    P_t_cross = (-10, 10, 0)  # C轴没改善，A轴增加了
    
    # 曼哈顿距离
    d_0_manhattan = abs(P_0_single[0]) + abs(P_0_single[1]) + abs(P_0_single[2])
    d_t_manhattan = abs(P_t_cross[0]) + abs(P_t_cross[1]) + abs(P_t_cross[2])
    
    score_manhattan = calculate_display_progress(P_t_cross, P_0_single)
    
    print(f"初始: P_0 = {P_0_single}, 曼哈顿距离 = {d_0_manhattan}")
    print(f"当前: P_t = {P_t_cross}, 曼哈顿距离 = {d_t_manhattan}")
    print(f"曼哈顿距离法进度 = {score_manhattan:.1f}%")
    print(f"\n✅ 曼哈顿距离：总赤字从10变成20，进度为0%（合理）")
    print(f"   因为虽然A轴有共情，但C轴的赤字完全没解决")
    
    # 对比欧几里得距离
    from Benchmark.epj.scoring import calculate_distance_to_zone
    d_0_euclidean = calculate_distance_to_zone(P_0_single)
    d_t_euclidean = calculate_distance_to_zone(P_t_cross)
    score_euclidean = (d_0_euclidean - d_t_euclidean) / d_0_euclidean * 100
    
    print(f"\n❌ 如果用欧几里得距离：")
    print(f"   d_0 = {d_0_euclidean:.2f}, d_t = {d_t_euclidean:.2f}")
    print(f"   进度 = {score_euclidean:.1f}% （负数！模型被惩罚了）")
    
    # 测试案例4: 分维度进度
    print("\n" + "=" * 70)
    print("案例4: 各维度进度（仅供显示）")
    print("=" * 70)
    
    P_0 = (-10, -17, -25)
    P_t = (-2, -7, -18)
    
    dim_progress = calculate_dimensional_progress(P_t, P_0)
    
    print(f"P_0 = {P_0}")
    print(f"P_t = {P_t}")
    print(f"\n各维度改善：")
    print(f"  C轴: {dim_progress['C_progress']:.1f}% (从-10到-2)")
    print(f"  A轴: {dim_progress['A_progress']:.1f}% (从-17到-7)")
    print(f"  P轴: {dim_progress['P_progress']:.1f}% (从-25到-18)")

