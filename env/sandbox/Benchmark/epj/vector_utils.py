# Benchmark/epj/vector_utils.py
"""
EPJ向量工具函数

提供向量解析和格式化的共享工具，避免代码重复
"""

import re
from typing import Tuple


def parse_vector_string(vector_str: str) -> Tuple[int, int, int]:
    """
    解析向量字符串为三元组
    
    Args:
        vector_str: 向量字符串，如 "(-3, -5, -12)" 或 "(+1, +3, +1)"
    
    Returns:
        Tuple[int, int, int]: (c, a, p) 三个维度的值
    
    Examples:
        >>> parse_vector_string("(-10, -17, -25)")
        (-10, -17, -25)
        >>> parse_vector_string("(+2, +3, +1)")
        (2, 3, 1)
        >>> parse_vector_string("(0, 0, 0)")
        (0, 0, 0)
    """
    numbers = re.findall(r'[+-]?\d+', str(vector_str))
    if len(numbers) >= 3:
        return (int(numbers[0]), int(numbers[1]), int(numbers[2]))
    return (0, 0, 0)


def format_vector(vec: Tuple[int, int, int], with_sign: bool = False) -> str:
    """
    格式化向量为字符串
    
    Args:
        vec: 三维向量 (c, a, p)
        with_sign: 是否显示正号
    
    Returns:
        str: 格式化的向量字符串
    
    Examples:
        >>> format_vector((-10, -17, -25))
        "(-10, -17, -25)"
        >>> format_vector((2, 3, 1), with_sign=True)
        "(+2, +3, +1)"
    """
    if with_sign:
        return f"({vec[0]:+d}, {vec[1]:+d}, {vec[2]:+d})"
    else:
        return f"({vec[0]}, {vec[1]}, {vec[2]})"


def vector_magnitude(vec: Tuple[int, int, int]) -> float:
    """
    计算向量的模（欧几里得距离）
    
    Args:
        vec: 三维向量 (c, a, p)
    
    Returns:
        float: 向量的模
    """
    import math
    return math.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)


def vector_manhattan_distance(vec: Tuple[int, int, int]) -> int:
    """
    计算向量的曼哈顿距离（各维度绝对值之和）
    
    Args:
        vec: 三维向量 (c, a, p)
    
    Returns:
        int: 曼哈顿距离
    """
    return abs(vec[0]) + abs(vec[1]) + abs(vec[2])


# ═══════════════════════════════════════════════════════════════
# 测试
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("EPJ向量工具测试")
    print("=" * 70)
    
    # 测试解析
    print("\n1. 测试向量解析")
    test_cases = [
        "(-10, -17, -25)",
        "(+2, +3, +1)",
        "(0, 0, 0)",
        "(-3, +5, -12)"
    ]
    
    for case in test_cases:
        parsed = parse_vector_string(case)
        print(f"   {case} → {parsed}")
    
    # 测试格式化
    print("\n2. 测试向量格式化")
    vectors = [
        (-10, -17, -25),
        (2, 3, 1),
        (0, 0, 0)
    ]
    
    for vec in vectors:
        formatted = format_vector(vec)
        formatted_with_sign = format_vector(vec, with_sign=True)
        print(f"   {vec}")
        print(f"      无符号: {formatted}")
        print(f"      带符号: {formatted_with_sign}")
    
    # 测试距离计算
    print("\n3. 测试距离计算")
    for vec in vectors:
        euclidean = vector_magnitude(vec)
        manhattan = vector_manhattan_distance(vec)
        print(f"   {vec}")
        print(f"      欧几里得距离: {euclidean:.2f}")
        print(f"      曼哈顿距离: {manhattan}")
    
    print("\n" + "=" * 70)
    print("✅ 所有测试完成")
    print("=" * 70)

