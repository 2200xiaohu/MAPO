# Benchmark/epj/scoring.py
"""
计分规则（Scoring Keys）

包含：
B. 初始赤字计分规则 (IED-SK)
D. MDEP进展计分规则 (MDEP-SK)
"""

from typing import Dict, Tuple


# ═══════════════════════════════════════════════════════════════
# B. 初始赤字计分规则 (IED-SK)
# ═══════════════════════════════════════════════════════════════

# 基础赤字单位 (BDU) = -2
# 递进逻辑：
# - 标准 (x1.0): -2, -4, -6
# - 优先级 (x1.5): -3, -6, -9
# - 核心困难 (x2.0): -4, -8, -12

IED_SCORING_KEY = {
    # --- C轴 (认知) ---
    # C.1 处境复杂性: 标准递进 (x1.0)
    ("C.1", 0): 0,
    ("C.1", 1): -2,
    ("C.1", 2): -4,
    ("C.1", 3): -6,
    
    # C.2 (深度): 标准递进 (x1.0) 【已补全】
    ("C.2", 0): 0,
    ("C.2", 1): -2,
    ("C.2", 2): -4,
    ("C.2", 3): -6,
    
    # C.3 认知优先级: 优先级递进 (x1.5) 【已修正】
    ("C.3", 0): 0,
    ("C.3", 1): -3,
    ("C.3", 2): -6,
    ("C.3", 3): -9,
    
    # --- A轴 (情感) ---
    # A.1 情绪强度: 标准递进 (x1.0)
    ("A.1", 0): 0,
    ("A.1", 1): -2,
    ("A.1", 2): -4,
    ("A.1", 3): -6,
    
    # A.2 情绪可及性: 核心困难递进 (x2.0)
    ("A.2", 0): 0,
    ("A.2", 1): -4,
    ("A.2", 2): -8,
    ("A.2", 3): -12,
    
    # A.3 情感优先级: 优先级递进 (x1.5)
    ("A.3", 0): 0,
    ("A.3", 1): -3,
    ("A.3", 2): -6,
    ("A.3", 3): -9,
    
    # --- P轴 (动机/赋能) ---
    # P.1 初始能动性: 标准递进 (x1.0)
    ("P.1", 0): 0,
    ("P.1", 1): -2,
    ("P.1", 2): -4,
    ("P.1", 3): -6,
    
    # P.2 价值关联度: 核心困难递进 (x2.0)
    ("P.2", 0): 0,
    ("P.2", 1): -4,
    ("P.2", 2): -8,
    ("P.2", 3): -12,
    
    # P.3 动机优先级: 优先级递进 (x1.5)
    ("P.3", 0): 0,
    ("P.3", 1): -3,
    ("P.3", 2): -6,
    ("P.3", 3): -9
}


def calculate_initial_deficit(filled_iedr: Dict) -> Tuple[int, int, int]:
    """
    计算初始共情赤字向量 P_0
    
    Args:
        filled_iedr: Judger 填写的 IEDR 量表
        格式：{
            "C.1": 2,
            "C.2": 1,
            "C.3": 2,
            "A.1": 2,
            "A.2": 1,
            "A.3": 2,
            "P.1": 2,
            "P.2": 3,
            "P.3": 3
        }
    
    Returns:
        Tuple[int, int, int]: (C, A, P) 初始赤字向量
    """
    C_deficit = 0
    A_deficit = 0
    P_deficit = 0
    
    #print(f"IEDR: {filled_iedr}")
    # C轴
    C_deficit += IED_SCORING_KEY.get(("C.1", filled_iedr.get("C.1", 0)), 0)
    C_deficit += IED_SCORING_KEY.get(("C.2", filled_iedr.get("C.2", 0)), 0)
    C_deficit += IED_SCORING_KEY.get(("C.3", filled_iedr.get("C.3", 0)), 0)
    
    # A轴
    A_deficit += IED_SCORING_KEY.get(("A.1", filled_iedr.get("A.1", 0)), 0)
    A_deficit += IED_SCORING_KEY.get(("A.2", filled_iedr.get("A.2", 0)), 0)
    A_deficit += IED_SCORING_KEY.get(("A.3", filled_iedr.get("A.3", 0)), 0)
    
    # P轴
    P_deficit += IED_SCORING_KEY.get(("P.1", filled_iedr.get("P.1", 0)), 0)
    P_deficit += IED_SCORING_KEY.get(("P.2", filled_iedr.get("P.2", 0)), 0)
    P_deficit += IED_SCORING_KEY.get(("P.3", filled_iedr.get("P.3", 0)), 0)
    
    return (C_deficit, A_deficit, P_deficit)


# ═══════════════════════════════════════════════════════════════
# D. MDEP进展计分规则 (MDEP-SK)
# ═══════════════════════════════════════════════════════════════

MDEP_SCORING_KEY = {
    # C轴
    "C.Prog": {
        0: 0,   # 中立/无关
        1: +1,  # 理解
        2: +3   # 升华/洞察
    },
    "C.Neg": {
        0: 0,   # 无
        -1: -2, # 忽视/轻微误解
        -2: -4  # 严重误解
    },
    
    # A轴
    "A.Prog": {
        0: 0,   # 中立/无关
        1: +1,  # 有效验证
        2: +3   # 深度共鸣
    },
    "A.Neg": {
        0: 0,   # 无
        -1: -2, # 冷漠/敷衍
        -2: -5  # 评判/指责（灾难性）
    },
    
    # P轴
    "P.Prog": {
        0: 0,   # 中立/无关
        1: +1,  # 认可/鼓励
        2: +3   # 赋能/肯定
    },
    "P.Neg": {
        0: 0,   # 无
        -1: -2, # 削弱/大包大揽
        -2: -5  # 打击/否定
    }
}


def calculate_increment_vector(filled_mdep_pr: Dict) -> Tuple[int, int, int]:
    """
    计算共情增量向量 v_t
    
    Args:
        filled_mdep_pr: Judger 填写的 MDEP-PR 量表
        格式：{
            "C.Prog": 1,
            "C.Neg": 0,
            "A.Prog": 2,
            "A.Neg": 0,
            "P.Prog": 1,
            "P.Neg": -1
        }
    
    Returns:
        Tuple[int, int, int]: (c, a, p) 增量向量
    """
    # C轴增量 = C.Prog + C.Neg
    c_increment = MDEP_SCORING_KEY["C.Prog"].get(filled_mdep_pr.get("C.Prog", 0), 0)
    c_increment += MDEP_SCORING_KEY["C.Neg"].get(filled_mdep_pr.get("C.Neg", 0), 0)
    
    # A轴增量 = A.Prog + A.Neg
    a_increment = MDEP_SCORING_KEY["A.Prog"].get(filled_mdep_pr.get("A.Prog", 0), 0)
    a_increment += MDEP_SCORING_KEY["A.Neg"].get(filled_mdep_pr.get("A.Neg", 0), 0)
    
    # P轴增量 = P.Prog + P.Neg
    p_increment = MDEP_SCORING_KEY["P.Prog"].get(filled_mdep_pr.get("P.Prog", 0), 0)
    p_increment += MDEP_SCORING_KEY["P.Neg"].get(filled_mdep_pr.get("P.Neg", 0), 0)
    
    return (c_increment, a_increment, p_increment)


# ═══════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════

def check_in_zone(P_t: Tuple[int, int, int], epsilon: float) -> bool:
    """
    检查当前位置是否在目标区域内（半空间语义）
    
    目标：消灭赤字（所有维度 >= 0）
    允许误差：epsilon范围内的负值也算达标
    正值不惩罚：超额完成不影响达标判断
    
    Args:
        P_t: 当前位置向量 (C, A, P)
        epsilon: 公差参数（允许的误差范围）
    
    Returns:
        bool: 是否所有维度 >= -epsilon
    """
    C, A, P = P_t
    return (C >= -epsilon) and (A >= -epsilon) and (P >= -epsilon)


def calculate_distance_to_zone(P_t: Tuple[int, int, int], epsilon: float = 1.0) -> float:
    """
    计算当前位置到目标区域的距离（半空间语义）
    
    距离定义：
    - 如果维度 >= -epsilon，贡献 0（已在目标区域内）
    - 如果维度 < -epsilon，贡献 (-epsilon - x)^2
    
    正值不计入距离：超额完成的维度距离为0
    
    Args:
        P_t: 当前位置向量
        epsilon: 目标区域边界（默认1.0）
    
    Returns:
        float: 到目标区域的欧氏距离
    """
    C, A, P = P_t
    
    # 只计算"未达标"维度到目标区域边界的距离
    c_dist = (-epsilon - C) if C < -epsilon else 0
    a_dist = (-epsilon - A) if A < -epsilon else 0
    p_dist = (-epsilon - P) if P < -epsilon else 0
    
    return (c_dist**2 + a_dist**2 + p_dist**2) ** 0.5


# ═══════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════

EPSILON_CONFIG = {
    "high_threshold": 1.0,    # 高阈值剧本，公差严格
    "medium_threshold": 2.0,  # 中阈值剧本，公差标准
    "low_threshold": 3.0      # 低阈值剧本，公差宽松
}


def get_epsilon(threshold_type: str = "high_threshold") -> float:
    """
    获取公差参数
    
    Args:
        threshold_type: 阈值类型（high/medium/low）
    
    Returns:
        float: 公差值
    """
    return EPSILON_CONFIG.get(threshold_type, 1.0)


# ═══════════════════════════════════════════════════════════════
# 测试示例
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
# EPM v2.0 判停函数（新增，不影响EPJ v1.0）
# ═══════════════════════════════════════════════════════════════

def check_epm_success(
    r_t: float,
    projection: float,
    E_total: float,
    epsilon_distance: float,
    epsilon_direction: float,
    epsilon_energy: float,
    current_turn: int = 0,
    min_turns: int = 15
) -> Tuple[bool, str]:
    """
    EPM v2.0 三重胜利条件判断（带最小轮次限制）
    
    Args:
        r_t: 当前位置到原点的距离 ||P_t||
        projection: P_t 在 v*_0 方向上的投影 (P_t · v*_0)
        E_total: 累计有效能量
        epsilon_distance: 相对距离阈值
        epsilon_direction: 相对方向阈值
        epsilon_energy: 能量阈值
        current_turn: 当前轮次
        min_turns: 最小轮次要求（默认15轮）
    
    Returns:
        Tuple[bool, str]: (是否成功, 胜利类型)
        胜利类型: "geometric"(几何) / "positional"(位置) / "energetic"(能量) / None
    """
    # 🚫 最小轮次限制：至少进行min_turns轮才能判定胜利
    if current_turn < min_turns:
        return (False, None)
    
    # ✅ 新判停逻辑：必须满足"空间改善(几何OR位置)" AND "能量充足"
    geometric_achieved = r_t <= epsilon_distance
    positional_achieved = projection >= -epsilon_direction
    energetic_achieved = E_total >= epsilon_energy
    
    # 空间条件：至少满足几何或位置之一
    spatial_success = geometric_achieved or positional_achieved
    
    # 🎯 最终判定：空间改善 AND 能量充足
    if spatial_success and energetic_achieved:
        # 确定主要胜利类型（优先级：几何 > 位置）
        if geometric_achieved:
            victory_type = "geometric"
        else:
            victory_type = "positional"
        
        return (True, victory_type)
    
    return (False, None)


def check_directional_collapse(trajectory: list, window_size: int = 5, min_turns: int = 12) -> bool:
    """
    检测方向性崩溃：连续N步的能量增量都为负
    
    Args:
        trajectory: 完整轨迹列表
        window_size: 检测窗口大小（默认5，更严格）
        min_turns: 最小轮次要求（默认15轮，与成功判定保持一致）
    
    Returns:
        bool: 是否发生方向性崩溃
    """
    # 🚫 最小轮次限制：至少进行min_turns轮才能判定失败
    current_turn = len(trajectory) - 1  # trajectory包含T=0，所以-1
    if current_turn < min_turns:
        return False
    
    if len(trajectory) < window_size + 1:  # +1 for T=0
        return False
    
    # 检查最近window_size步的delta_E
    recent_deltas = []
    for i in range(-window_size, 0):
        point = trajectory[i]
        if 'epm' in point and 'delta_E' in point['epm']:
            recent_deltas.append(point['epm']['delta_E'])
    
    # 如果所有delta_E都<0，判定为崩溃
    if len(recent_deltas) == window_size and all(d < 0 for d in recent_deltas):
        return True
    
    return False


def check_stagnation(trajectory: list, window_size: int = 5, threshold: float = 0.5, min_turns: int = 12) -> bool:
    """
    检测陷入停滞：最近N步的位置变化很小（几乎原地踏步）
    
    Args:
        trajectory: 完整轨迹列表
        window_size: 检测窗口大小（默认5）
        threshold: 位置变化阈值（默认0.5）
        min_turns: 最小轮次要求（默认15轮）
    
    Returns:
        bool: 是否陷入停滞
    """
    # 🚫 最小轮次限制：至少进行min_turns轮才能判定失败
    current_turn = len(trajectory) - 1
    if current_turn < min_turns:
        return False
    
    if len(trajectory) < window_size + 1:
        return False
    
    # 获取最近window_size步的距离变化
    recent_distances = []
    for i in range(-window_size, 0):
        point = trajectory[i]
        if 'epm' in point and 'P_norm' in point['epm']:
            recent_distances.append(point['epm']['P_norm'])
    
    if len(recent_distances) < window_size:
        return False
    
    # 计算距离变化的标准差
    import statistics
    try:
        std_dev = statistics.stdev(recent_distances)
        # 如果标准差很小，说明位置几乎不变
        if std_dev < threshold:
            return True
    except:
        pass
    
    return False


def check_persistent_regression(trajectory: list, window_size: int = 8, min_turns: int = 12) -> bool:
    """
    检测持续倒退：连续多轮负能量且累计能量持续下降
    
    Args:
        trajectory: 完整轨迹列表
        window_size: 检测窗口大小（默认8）
        min_turns: 最小轮次要求（默认15轮）
    
    Returns:
        bool: 是否持续倒退
    """
    # 🚫 最小轮次限制：至少进行min_turns轮才能判定失败
    current_turn = len(trajectory) - 1
    if current_turn < min_turns:
        return False
    
    if len(trajectory) < window_size + 1:
        return False
    
    # 检查最近window_size步的delta_E
    recent_deltas = []
    for i in range(-window_size, 0):
        point = trajectory[i]
        if 'epm' in point and 'delta_E' in point['epm']:
            recent_deltas.append(point['epm']['delta_E'])
    
    if len(recent_deltas) < window_size:
        return False
    
    # 检查是否大部分步骤（>70%）都是负能量
    negative_count = sum(1 for d in recent_deltas if d < 0)
    if negative_count > window_size * 0.7:
        # 并且总能量和为负
        total_delta = sum(recent_deltas)
        if total_delta < -1.0:  # 累计损失超过1个单位
            return True
    
    return False


def get_epm_state_summary(
    current_turn: int,
    r_t: float,
    projection: float,
    E_total: float,
    epsilon_distance: float,
    epsilon_direction: float,
    epsilon_energy: float,
    trajectory: list,
    min_turns: int = 15
) -> dict:
    """
    生成EPM状态摘要（用于Director和结果分析）
    
    Returns:
        dict: EPM状态摘要，包含胜利条件和多种失败检测
    """
    # 检查胜利条件（带最小轮次限制）
    success, victory_type = check_epm_success(
        r_t, projection, E_total,
        epsilon_distance, epsilon_direction, epsilon_energy,
        current_turn, min_turns
    )
    
    # 🚫 多重失败检测（带最小轮次限制）
    is_collapsed = check_directional_collapse(trajectory, window_size=5, min_turns=min_turns)
    is_stagnant = check_stagnation(trajectory, window_size=5, threshold=0.5, min_turns=min_turns)
    is_regressing = check_persistent_regression(trajectory, window_size=8, min_turns=min_turns)
    
    # 综合判断是否应该失败终止
    should_fail = is_collapsed or (is_stagnant and is_regressing)
    
    return {
        "turn": current_turn,
        "success": success,
        "victory_type": victory_type,
        "failure_detected": should_fail,
        "failure_reasons": {
            "collapsed": is_collapsed,        # 连续5轮负能量
            "stagnant": is_stagnant,          # 位置停滞不前
            "regressing": is_regressing       # 持续倒退（8轮中70%负能量）
        },
        "metrics": {
            "r_t": round(r_t, 2),
            "projection": round(projection, 2),
            "E_total": round(E_total, 2)
        },
        "thresholds": {
            "epsilon_distance": round(epsilon_distance, 2),
            "epsilon_direction": round(epsilon_direction, 2),
            "epsilon_energy": round(epsilon_energy, 2)
        },
        "progress": {
            "geometric": f"{(1 - r_t / epsilon_distance) * 100:.1f}%" if epsilon_distance > 0 else "N/A",
            "positional": f"{((projection + epsilon_direction) / epsilon_direction) * 100:.1f}%" if epsilon_direction > 0 else "N/A",
            "energetic": f"{(E_total / epsilon_energy) * 100:.1f}%" if epsilon_energy > 0 else "N/A"
        },
        "min_turns_met": current_turn >= min_turns
    }


# ═══════════════════════════════════════════════════════════════
# 测试代码
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("EPJ 计分规则测试")
    print("=" * 60)
    
    # 测试初始赤字计算
    print("\n1. 测试初始赤字计算 (T=0)")
    filled_iedr = {
        "C.1": 2,  # 中 - 特定领域知识
        "C.2": 1,  # 次要
        "C.3": 2,  # 核心
        "A.1": 2,  # 强烈
        "A.2": 1,  # 隐含
        "A.3": 2,  # 核心
        "P.1": 2,  # 低能动性
        "P.2": 3,  # 危机/核心追求
        "P.3": 3   # 最高
    }
    
    P_0 = calculate_initial_deficit(filled_iedr)
    print(f"   Filled IEDR: {filled_iedr}")
    print(f"   P_0 = {P_0}")
    print(f"   解释: C={P_0[0]}, A={P_0[1]}, P={P_0[2]}")
    
    # 测试增量计算
    print("\n2. 测试增量计算 (T>0)")
    filled_mdep_pr = {
        "C.Prog": 1,   # 理解
        "C.Neg": 0,    # 无
        "A.Prog": 2,   # 深度共鸣
        "A.Neg": 0,    # 无
        "P.Prog": 1,   # 认可/鼓励
        "P.Neg": -1    # 削弱/大包大揽
    }
    
    v_t = calculate_increment_vector(filled_mdep_pr)
    print(f"   Filled MDEP-PR: {filled_mdep_pr}")
    print(f"   v_t = {v_t}")
    print(f"   解释: c={v_t[0]}, a={v_t[1]}, p={v_t[2]}")
    
    # 测试位置更新
    print("\n3. 测试位置更新")
    P_t = tuple(P_0[i] + v_t[i] for i in range(3))
    print(f"   P_t = P_0 + v_t = {P_t}")
    
    # 测试区域检查
    print("\n4. 测试目标区域检查")
    epsilon = get_epsilon("high_threshold")
    is_in_zone = check_in_zone(P_t, epsilon)
    distance = calculate_distance_to_zone(P_t)
    print(f"   Epsilon (high): {epsilon}")
    print(f"   Is in zone: {is_in_zone}")
    print(f"   Distance to (0,0,0): {distance:.2f}")
    
    # 模拟多轮进展
    print("\n5. 模拟轨迹进展")
    position = P_0
    print(f"   T=0: P = {position}, 距离 = {calculate_distance_to_zone(position):.2f}")
    
    for t in range(1, 6):
        # 假设每轮都有正向进展
        v = (1, 2, 2)
        position = tuple(position[i] + v[i] for i in range(3))
        distance = calculate_distance_to_zone(position)
        in_zone = check_in_zone(position, epsilon)
        print(f"   T={t}: P = {position}, 距离 = {distance:.2f}, 达标 = {in_zone}")

