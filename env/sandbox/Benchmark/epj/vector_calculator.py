# Benchmark/epj/vector_calculator.py
"""
向量计算器（Orchestrator的EPJ组件）

职责：
- 计算初始赤字向量 P_0
- 计算增量向量 v_t
- 更新当前位置 P_t
- 生成状态数据包
"""

from typing import Dict, Tuple, List
from .scoring import (
    calculate_initial_deficit,
    calculate_increment_vector,
    check_in_zone,
    calculate_distance_to_zone,
    get_epsilon
)


class VectorCalculator:
    """
    向量计算器 - EPJ系统的计算核心
    
    职责：
    - 纯粹的数学计算，不做质性判断
    - 不做最终决策（由Director负责）
    - 透明、可重现的计算过程
    """
    
    def __init__(self, threshold_type: str = "high_threshold", K: int = 3, max_turns: int = 30, enable_epm: bool = True):
        """
        初始化向量计算器
        
        Args:
            threshold_type: 阈值类型（high/medium/low）
            K: 评估周期（每K轮评估一次）
            max_turns: 最大轮次
            enable_epm: 是否启用EPM v2.0计算（默认True，向后兼容）
        """
        self.epsilon = get_epsilon(threshold_type)
        self.K = K
        self.max_turns = max_turns
        
        # 轨迹追踪（EPJ v1.0 - 保留）
        self.P_0 = None  # 初始赤字向量
        self.trajectory = []  # 完整轨迹：[(turn, P_t, v_t), ...]
        self.current_P = None  # 当前位置
        
        # EPM v2.0 新增属性（不影响EPJ v1.0）
        self.enable_epm = enable_epm
        self.v_star_0 = None  # 全局理想方向（归一化）
        self.E_total = 0.0  # 累计有效能量
        self.epsilon_distance_relative = None  # 相对距离阈值
        self.epsilon_direction_relative = None  # 相对方向阈值
        self.epsilon_energy = None  # 能量阈值
        
        # print(f"✅ [VectorCalculator/计算器] 初始化完成")
        # if enable_epm:
        #     print(f"   🆕 EPM v2.0 模式（能量动力学）")
        #     print(f"   - 相对阈值将在P_0计算后动态设置")
        #     print(f"   - EPJ v1.0 兼容参数 ε = {self.epsilon}（向后兼容）")
        # else:
        #     print(f"   EPJ v1.0 模式")
        #     print(f"   公差参数 ε = {self.epsilon}")
        # print(f"   评估周期 K = {self.K}")
        # print(f"   最大轮次 = {self.max_turns}")
    
    def _calculate_vector_norm(self, vec: Tuple) -> float:
        """计算向量的欧氏模长（EPM辅助函数）"""
        import math
        return math.sqrt(sum(x**2 for x in vec))
    
    def _calculate_dot_product(self, vec1: Tuple, vec2: Tuple) -> float:
        """计算两个向量的内积（EPM辅助函数）"""
        return sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
    
    def _normalize_vector(self, vec: Tuple) -> Tuple[float, float, float]:
        """归一化向量（EPM辅助函数）"""
        norm = self._calculate_vector_norm(vec)
        if norm == 0:
            return (0.0, 0.0, 0.0)
        return tuple(x / norm for x in vec)
    
    def _initialize_epm_from_P0(self, P_0: Tuple) -> dict:
        """
        从P_0初始化EPM v2.0参数（独立方法，确保所有路径都能正确初始化）
        
        Args:
            P_0: 初始赤字向量
        
        Returns:
            dict: EPM初始化数据（用于轨迹点）
        """
        if not self.enable_epm:
            return {}
        
        # 1. 计算初始赤字模长
        P_0_norm = self._calculate_vector_norm(P_0)
        
        # 2. 计算全局理想方向 v*_0 = -P_0 / ||P_0||
        if P_0_norm > 0:
            self.v_star_0 = tuple(-x / P_0_norm for x in P_0)
        else:
            self.v_star_0 = (0.0, 0.0, 0.0)
        
        # 3. 计算EPM阈值（α = 0.05，更严格：需消除95%赤字）
        alpha = 0.05
        self.epsilon_distance_relative = alpha * P_0_norm
        self.epsilon_direction_relative = alpha * P_0_norm
        self.epsilon_energy = P_0_norm
        
        # 4. 初始化累计能量
        self.E_total = 0.0
        
        # 5. 打印初始化信息
        # print(f"\n🆕 [EPM v2.0] 初始化完成")
        # print(f"   ||P_0|| = {P_0_norm:.2f}")
        # print(f"   v*_0（理想方向）= {tuple(round(x, 3) for x in self.v_star_0)}")
        # print(f"   ε_distance（距离阈值）= {self.epsilon_distance_relative:.2f} (α=0.05，需消除95%赤字)")
        # print(f"   ε_direction（方向阈值）= {self.epsilon_direction_relative:.2f}")
        # print(f"   ε_energy（能量阈值）= {self.epsilon_energy:.2f}")
        
        # 6. 返回EPM轨迹数据
        return {
            "P_norm": round(P_0_norm, 2),
            "v_star_0": tuple(round(x, 4) for x in self.v_star_0),
            "E_total": 0.0,
            "projection": -P_0_norm,  # P_0 · v*_0 = -||P_0||
            "alignment": 0.0  # T=0时无移动
        }
    
    def calculate_P_0(self, filled_iedr: Dict) -> Tuple[int, int, int]:
        """
        计算初始赤字向量 P_0 (T=0)
        
        Args:
            filled_iedr: Judger填写的IEDR量表
        
        Returns:
            Tuple[int, int, int]: (C, A, P) 初始赤字向量
        """
        print(f"═══ [VectorCalculator/计算器] T=0: 计算 P_0 ═══")
        
        # EPJ v1.0 原有逻辑（保留）
        P_0 = calculate_initial_deficit(filled_iedr)
        
        self.P_0 = P_0
        self.current_P = P_0
        
        # 构建轨迹点（基础字段）
        trajectory_point = {
            "turn": 0,
            "P_t": P_0,
            "v_t": (0, 0, 0),
            "distance": calculate_distance_to_zone(P_0, self.epsilon),
            "in_zone": check_in_zone(P_0, self.epsilon)
        }
        
        # EPM v2.0 初始化（调用统一方法）
        epm_data = self._initialize_epm_from_P0(P_0)
        if epm_data:
            trajectory_point['epm'] = epm_data
        
        self.trajectory.append(trajectory_point)
        
        print(f"✅ [VectorCalculator] P_0 计算完成")
        print(f"   P_0 = {P_0}")
        if self.enable_epm:
            print(f"   初始距离（欧氏）= {self._calculate_vector_norm(P_0):.2f}")
        else:
            print(f"   初始距离 = {calculate_distance_to_zone(P_0, self.epsilon):.2f}")
        
        return P_0
    
    def calculate_v_t_and_update(self, filled_mdep_pr: Dict, current_turn: int) -> Tuple[int, int, int]:
        """
        计算增量向量 v_t 并更新当前位置 (T>0)
        
        Args:
            filled_mdep_pr: Judger填写的MDEP-PR量表
            current_turn: 当前轮次
        
        Returns:
            Tuple[int, int, int]: (c, a, p) 增量向量
        """
        print(f"═══ [VectorCalculator/计算器] T={current_turn}: 计算 v_t 并更新 P_t ═══")
        
        # EPJ v1.0 原有逻辑（保留）
        # 计算增量
        v_t = calculate_increment_vector(filled_mdep_pr)
        
        # 更新位置
        P_prev = self.current_P if self.current_P else (0, 0, 0)
        P_t = tuple(P_prev[i] + v_t[i] for i in range(3))
        
        self.current_P = P_t
        
        # 💾 提取详细分析信息（如果存在）
        detailed_analysis = filled_mdep_pr.get('detailed_analysis', {})
        
        # 记录轨迹（基础字段）
        trajectory_point = {
            "turn": current_turn,
            "P_t": P_t,
            "v_t": v_t,
            "distance": calculate_distance_to_zone(P_t, self.epsilon),
            "in_zone": check_in_zone(P_t, self.epsilon)
        }
        
        # 如果有详细分析，添加到轨迹点中
        if detailed_analysis:
            trajectory_point['mdep_analysis'] = detailed_analysis
        
        # EPM v2.0 能量动力学计算（新增）
        if self.enable_epm and self.v_star_0 is not None:
            # 1. 计算向量模长
            v_t_norm = self._calculate_vector_norm(v_t)
            
            # 2. 计算对齐度（夹角余弦）
            if v_t_norm > 0:
                alignment = self._calculate_dot_product(v_t, self.v_star_0) / v_t_norm
            else:
                alignment = 0.0
            
            # 3. 计算有效能量增量（投影功）
            delta_E = v_t_norm * alignment
            
            # 4. 累计能量
            self.E_total += delta_E
            
            # 5. 计算当前位置的投影
            P_t_norm = self._calculate_vector_norm(P_t)
            projection = self._calculate_dot_product(P_t, self.v_star_0)
            
            # 6. 添加EPM字段到轨迹点
            trajectory_point['epm'] = {
                "v_t_norm": round(v_t_norm, 2),
                "alignment": round(alignment, 4),
                "delta_E": round(delta_E, 2),
                "E_total": round(self.E_total, 2),
                "P_norm": round(P_t_norm, 2),
                "projection": round(projection, 2)
            }
            
            # print(f"\n🆕 [EPM v2.0] 能量动力学分析")
            # print(f"   移动模长 ||v_t|| = {v_t_norm:.2f}")
            # print(f"   对齐度 cos(θ) = {alignment:.3f} ({'正向' if alignment > 0 else '反向' if alignment < 0 else '垂直'})")
            # print(f"   有效能量增量 ΔE = {delta_E:+.2f}")
            # print(f"   累计能量 E_total = {self.E_total:.2f} / {self.epsilon_energy:.2f} ({self.E_total/self.epsilon_energy*100:.1f}%)")
            # print(f"   当前距离 ||P_t|| = {P_t_norm:.2f}")
            # print(f"   位置投影 P_t·v*_0 = {projection:+.2f} ({'已穿越' if projection >= 0 else '未穿越'})")
        
        self.trajectory.append(trajectory_point)
        
        print(f"\n✅ [VectorCalculator] 计算完成")
        print(f"   v_t = {v_t}")
        print(f"   P_t = {P_t}")
        if self.enable_epm:
            print(f"   距离（欧氏）= {P_t_norm:.2f}")
        else:
            print(f"   距离 = {calculate_distance_to_zone(P_t, self.epsilon):.2f}")
        
        return v_t
    
    def generate_state_packet(self, current_turn: int) -> dict:
        """
        生成状态数据包（发送给Director）
        
        Args:
            current_turn: 当前轮次
        
        Returns:
            dict: 状态数据包
        """
        if not self.current_P:
            raise ValueError("尚未计算P_t，请先调用 calculate_v_t_and_update")
        
        P_t = self.current_P
        v_t = self.trajectory[-1]['v_t'] if self.trajectory else (0, 0, 0)
        
        is_in_zone = check_in_zone(P_t, self.epsilon)
        is_timeout = (current_turn >= self.max_turns)
        distance = calculate_distance_to_zone(P_t, self.epsilon)
        
        # 计算等效进度分数（仅供显示）
        from .display_metrics import calculate_display_progress
        display_progress = calculate_display_progress(P_t, self.P_0) if self.P_0 else 0
        
        # 🔧 停滞检测：动态调整window_size以适应不同的K值
        # 目标：无论K是多少，都覆盖约12-15轮对话
        adaptive_window_size = max(10, int(12 / self.K))
        stagnation_result = self.detect_stagnation(
            window_size=adaptive_window_size, 
            min_total_change=3  # 放宽阈值：从2增加到3
        )
        
        state_packet = {
            "current_turn": current_turn,
            "max_turns": self.max_turns,
            "K": self.K,
            "epsilon": self.epsilon,
            
            # 向量信息（科学决策用）
            "P_0_start_deficit": f"({self.P_0[0]}, {self.P_0[1]}, {self.P_0[2]})" if self.P_0 else None,
            "P_t_current_position": f"({P_t[0]}, {P_t[1]}, {P_t[2]})",
            "v_t_last_increment": f"({v_t[0]}, {v_t[1]}, {v_t[2]})",
            
            # 判断信息（科学决策用）
            "is_in_zone": is_in_zone,
            "is_timeout": is_timeout,
            "is_stagnant": stagnation_result['is_stagnant'],  # 🔧 新增
            "stagnation_info": stagnation_result,  # 🔧 新增：完整的停滞信息
            "distance_to_goal": round(distance, 2),
            
            # 显示信息（仅供参考）
            "display_progress": round(display_progress, 1),  # 0-100的等效分数
            
            # 轨迹摘要
            "trajectory_length": len(self.trajectory),
            "total_evaluations": len(self.trajectory) - 1  # 减去T=0
        }
        
        # print(f"📦 [VectorCalculator] 状态数据包生成完成")
        # print(f"   P_t = {state_packet['P_t_current_position']}")
        # print(f"   距离目标 = {distance:.2f}（欧氏距离）")
        # print(f"   等效进度 = {display_progress:.1f}%（仅供显示）")
        # print(f"   在区域内 = {is_in_zone}（科学决策标准）")
        # print(f"   超时 = {is_timeout}")
        
        # 🔧 显示停滞检测结果
        if stagnation_result['is_stagnant']:
            print(f"   ⚠️  停滞检测 = True ({stagnation_result['stagnation_type']})")
            print(f"      {stagnation_result['reason']}")
        else:
            print(f"   ✅ 停滞检测 = False（进展正常）")
        
        return state_packet
    
    def get_trajectory(self) -> List[dict]:
        """获取完整轨迹"""
        return self.trajectory.copy()
    
    def get_current_position(self) -> Tuple[int, int, int]:
        """获取当前位置"""
        return self.current_P if self.current_P else (0, 0, 0)
    
    # ═══════════════════════════════════════════════════════════════
    # 🔧 问题4修复: 停滞检测
    # ═══════════════════════════════════════════════════════════════
    
    def detect_stagnation(self, window_size: int = 4, min_total_change: int = 2) -> dict:
        """
        检测最近N次评估是否陷入停滞
        
        停滞定义：
        1. 向量总变化量太小（如连续3次评估总共才变化<3个单位）
        2. 距离几乎不变（波动<2.0）
        3. 在目标区域外徘徊，无实质进展
        
        Args:
            window_size: 检测窗口大小（建议3-4）
            min_total_change: 最小总变化量阈值
        
        Returns:
            dict: {
                "is_stagnant": bool,
                "reason": str,
                "stagnation_type": str,
                "window_data": dict
            }
        """
        # 需要至少window_size+1个数据点（包括T=0）
        if len(self.trajectory) < window_size + 1:
            return {
                "is_stagnant": False,
                "reason": "评估次数不足，无法判断停滞"
            }
        
        # 提取最近window_size次评估的数据（不包括T=0）
        recent_points = self.trajectory[-(window_size):]
        
        # 检测1: 向量总变化量
        total_vector_change = 0
        for point in recent_points:
            v_t = point.get('v_t', (0, 0, 0))
            # 计算向量的曼哈顿范数（总变化量）
            total_vector_change += abs(v_t[0]) + abs(v_t[1]) + abs(v_t[2])
        
        if total_vector_change < min_total_change:
            return {
                "is_stagnant": True,
                "reason": f"最近{window_size}次评估向量总变化量过小(仅{total_vector_change}个单位)",
                "stagnation_type": "MINIMAL_CHANGE",
                "window_data": {
                    "total_change": total_vector_change,
                    "threshold": min_total_change,
                    "window_size": window_size
                }
            }
        
        # 检测2: 距离波动太小（说明在某个位置徘徊）
        recent_distances = [point['distance'] for point in recent_points]
        distance_variance = max(recent_distances) - min(recent_distances)
        
        # 🔧 优化：放宽距离波动阈值（1.5 → 2.0）
        if distance_variance < 2.0:
            # 额外检查：是否在目标区域外徘徊
            avg_distance = sum(recent_distances) / len(recent_distances)
            if avg_distance > self.epsilon:  # 不在目标区域内
                return {
                    "is_stagnant": True,
                    "reason": f"最近{window_size}次评估距离几乎不变(波动{distance_variance:.1f}，平均距离{avg_distance:.1f})",
                    "stagnation_type": "OSCILLATION",
                    "window_data": {
                        "distance_variance": distance_variance,
                        "avg_distance": avg_distance,
                        "recent_distances": recent_distances
                    }
                }
        
        # 检测3: 反复震荡（正负交替但净变化小）
        # 计算净变化（从window开始到结束的实际位移）
        if len(recent_points) >= 2:
            start_P = recent_points[0]['P_t']
            end_P = recent_points[-1]['P_t']
            net_change = sum(abs(end_P[i] - start_P[i]) for i in range(3))
            
            # 🔧 优化：放宽净变化阈值（2 → 3）
            # 如果总变化量足够，但净变化<3，说明在震荡
            if total_vector_change >= min_total_change and net_change < 3:
                return {
                    "is_stagnant": True,
                    "reason": f"向量反复震荡：总变化{total_vector_change}，但净变化仅{net_change}",
                    "stagnation_type": "OSCILLATION",
                    "window_data": {
                        "total_change": total_vector_change,
                        "net_change": net_change,
                        "start_P": start_P,
                        "end_P": end_P
                    }
                }
        
        # 没有检测到停滞
        return {
            "is_stagnant": False,
            "reason": "对话正常进展中"
        }


# ═══════════════════════════════════════════════════════════════
# 测试
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("VectorCalculator 测试")
    print("=" * 70)
    
    # 初始化计算器
    calculator = VectorCalculator(threshold_type="high_threshold", K=3)
    
    # T=0: 计算初始赤字
    print("\n1. T=0: 计算初始赤字")
    filled_iedr = {
        "C.1": 2, "C.2": 1,
        "A.1": 2, "A.2": 1, "A.3": 2,
        "P.1": 2, "P.2": 3, "P.3": 3
    }
    P_0 = calculator.calculate_P_0(filled_iedr)
    
    # 模拟几轮评估
    print("\n2. 模拟对话进展")
    
    for cycle in range(1, 6):
        turn = cycle * 3
        print(f"\n--- 第{cycle}次评估（第{turn}轮） ---")
        
        # 模拟Judger填写的MDEP-PR
        filled_mdep_pr = {
            "C.Prog": 1, "C.Neg": 0,
            "A.Prog": 2, "A.Neg": 0,
            "P.Prog": 1, "P.Neg": 0
        }
        
        v_t = calculator.calculate_v_t_and_update(filled_mdep_pr, turn)
        state_packet = calculator.generate_state_packet(turn)
        
        print(f"\n📦 状态数据包:")
        print(f"   当前轮次: {state_packet['current_turn']}")
        print(f"   当前位置: {state_packet['P_t_current_position']}")
        print(f"   本轮增量: {state_packet['v_t_last_increment']}")
        print(f"   距离目标: {state_packet['distance_to_goal']}")
        print(f"   在区域内: {state_packet['is_in_zone']}")
        
        if state_packet['is_in_zone']:
            print(f"\n🎉 达到目标区域！")
            break
    
    # 显示完整轨迹
    print(f"\n3. 完整轨迹")
    trajectory = calculator.get_trajectory()
    for point in trajectory:
        print(f"   T={point['turn']}: P={point['P_t']}, v={point['v_t']}, 距离={point['distance']:.2f}")

