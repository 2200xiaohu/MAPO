# Benchmark/orchestrator/epj_orchestrator.py
"""
EPJ Orchestrator - 向量计算与状态管理

职责：
- 在T=0时调用Judger填写IEDR，计算P_0
- 每K轮调用Judger填写MDEP-PR，计算v_t和P_t
- 生成状态数据包发送给Director
- 纯粹的计算和状态管理，不做质性判断和最终决策
"""

from typing import Dict, Tuple
from Benchmark.epj.vector_calculator import VectorCalculator


class EPJOrchestrator:
    """
    EPJ编排器 - 计算核心
    
    三层架构中的"计算器"角色：
    - 接收Judger的量表
    - 执行透明的数学计算
    - 生成状态数据包
    - 不做最终决策
    """
    
    def __init__(self, judger, threshold_type: str = "high_threshold", K: int = 3, max_turns: int = 30, enable_epm: bool = True):
        """
        初始化EPJ编排器
        
        Args:
            judger: Judger实例
            threshold_type: 阈值类型
            K: 评估周期
            max_turns: 最大轮次
            enable_epm: 是否启用EPM v2.0（默认True，向后兼容）
        """
        self.judger = judger
        self.calculator = VectorCalculator(threshold_type, K, max_turns, enable_epm)
        self.K = K
        self.enable_epm = enable_epm
        
        # 标记是否已初始化P_0
        self.initialized = False
        
        # print(f"✅ [EPJOrchestrator/编排器] 初始化完成")
        # print(f"   评估周期 K = {K}")
        # if enable_epm:
        #     print(f"   🆕 EPM v2.0 已启用")
    
    def initialize_at_T0(self, script_content: dict) -> dict:
        """
        T=0时的初始化：计算初始赤字向量
        
        流程：
        1. Judger填写IEDR量表
        2. Calculator计算P_0
        
        Args:
            script_content: 剧本内容（包含actor_prompt和scenario）
        
        Returns:
            dict: 初始化结果
            {
                "P_0": (C, A, P),
                "filled_iedr": {...},
                "initial_distance": float
            }
        """
        print(f"{'='*70}")
        print(f"EPJ 初始化 (T=0) - 重新初始化")
        print(f"{'='*70}")
        
        # 步骤1: Judger填写IEDR
        filled_iedr = self.judger.fill_iedr(script_content)
        
        # 步骤2: Calculator计算P_0
        P_0 = self.calculator.calculate_P_0(filled_iedr)
        
        distance = self.calculator.trajectory[0]['distance']
        
        self.initialized = True
        
        # 💾 保存IEDR结果供最终结果使用
        self.iedr_result = {
            "filled_iedr": filled_iedr,
            "P_0": P_0,
            "initial_distance": distance,
            "source": "runtime_evaluation"
        }
        
        print(f"\n✅ [EPJOrchestrator] T=0 初始化完成")
        print(f"   P_0 = {P_0}")
        print(f"   初始赤字距离 = {distance:.2f}")
        
        return {
            "P_0": P_0,
            "filled_iedr": filled_iedr,
            "initial_distance": distance
        }
    
    def initialize_with_precomputed_iedr(self, filled_iedr: dict, P_0: Tuple[int, int, int]) -> dict:
        """
        使用预先计算的IEDR初始化（避免重复调用Judger）
        
        当已经有批量评估的IEDR结果时使用此方法，跳过Judger调用
        
        Args:
            filled_iedr: 预先填写的IEDR量表
            P_0: 预先计算的初始赤字向量 (C, A, P)
        
        Returns:
            dict: 初始化结果
            {
                "P_0": (C, A, P),
                "filled_iedr": {...},
                "initial_distance": float
            }
        """
        print(f"{'='*70}")
        print(f"EPJ 初始化 (T=0) - 使用预先计算的IEDR")
        print(f"{'='*70}")
        
        # 直接使用预先计算的P_0初始化Calculator
        self.calculator.P_0 = P_0
        self.calculator.current_P = P_0
        
        # 初始化轨迹（EPJ v1.0基础）
        from Benchmark.epj.scoring import calculate_distance_to_zone, check_in_zone
        distance = calculate_distance_to_zone(P_0, self.calculator.epsilon)
        
        trajectory_point = {
            "turn": 0,
            "P_t": P_0,
            "v_t": (0, 0, 0),
            "distance": distance,
            "in_zone": check_in_zone(P_0, self.calculator.epsilon)
        }
        
        # 🔧 EPM v2.0: 直接使用预计算的EPM参数（如果存在）
        # 优先使用预计算参数，避免重复计算
        if self.calculator.enable_epm:
            # 检查filled_iedr是否包含预计算的EPM参数
            if isinstance(filled_iedr, dict) and 'epm' in filled_iedr:
                # 使用预计算参数
                epm_precomputed = filled_iedr['epm']
                self.calculator.v_star_0 = tuple(epm_precomputed['v_star_0'])
                self.calculator.epsilon_distance_relative = epm_precomputed['epsilon_distance']
                self.calculator.epsilon_direction_relative = epm_precomputed['epsilon_direction']
                self.calculator.epsilon_energy = epm_precomputed['epsilon_energy']
                self.calculator.E_total = 0.0
                
                # 添加到轨迹点
                trajectory_point['epm'] = {
                    "P_norm": epm_precomputed['P_0_norm'],
                    "v_star_0": epm_precomputed['v_star_0'],
                    "E_total": 0.0,
                    "projection": -epm_precomputed['P_0_norm'],
                    "alignment": 0.0
                }
                
                print(f"\n🆕 [EPM v2.0] 使用预计算参数")
                print(f"   ||P_0|| = {epm_precomputed['P_0_norm']:.2f}")
                print(f"   v*_0（理想方向）= {tuple(round(x, 3) for x in epm_precomputed['v_star_0'])}")
                print(f"   ε_distance（距离阈值）= {epm_precomputed['epsilon_distance']:.2f}")
                print(f"   ε_direction（方向阈值）= {epm_precomputed['epsilon_direction']:.2f}")
                print(f"   ε_energy（能量阈值）= {epm_precomputed['epsilon_energy']:.2f}")
            else:
                # 回退：运行时计算
                epm_data = self.calculator._initialize_epm_from_P0(P_0)
                if epm_data:
                    trajectory_point['epm'] = epm_data
        
        self.calculator.trajectory = [trajectory_point]
        
        self.initialized = True
        
        # 💾 保存IEDR结果供最终结果使用
        self.iedr_result = {
            "filled_iedr": filled_iedr,
            "P_0": P_0,
            "initial_distance": distance,
            "source": "precomputed"
        }
        
        print(f"\n✅ [EPJOrchestrator] T=0 初始化完成（使用预计算IEDR）")
        print(f"   P_0 = {P_0}")
        print(f"   初始赤字距离 = {distance:.2f}")
        print(f"   📝 跳过Judger调用，节省API消耗")
        
        return {
            "P_0": P_0,
            "filled_iedr": filled_iedr,
            "initial_distance": distance
        }
    
    def evaluate_at_turn_K(self, recent_turns: list, current_turn: int, script_content: dict = None, full_history: list = None) -> dict:
        """
        每K轮的评估：计算增量和新位置
        
        流程：
        1. Judger填写MDEP-PR量表
        2. Calculator计算v_t
        3. Calculator更新P_t
        4. Calculator生成状态数据包
        
        Args:
            recent_turns: 最近K轮的对话（评估范围）
            current_turn: 当前轮次
            script_content: 剧本内容（包含actor_prompt，用于Judger代入视角）
            full_history: 完整对话历史（供Judger参考上下文）
        
        Returns:
            dict: 状态数据包（发送给Director）
        """
        if not self.initialized:
            raise RuntimeError("EPJ未初始化，请先调用 initialize_at_T0")
        
        print(f"\n{'='*70}")
        print(f"EPJ 评估 (T={current_turn})")
        print(f"{'='*70}")
        
        # 步骤1: Judger填写MDEP-PR（传递script_content以便代入Actor视角）
        # 🆕 同时传递完整历史供Judger参考上下文
        filled_mdep_pr = self.judger.fill_mdep_pr(
            recent_turns, 
            script_context=script_content,
            full_history=full_history
        )
        
        # 步骤2 & 3: Calculator计算v_t并更新P_t
        v_t = self.calculator.calculate_v_t_and_update(filled_mdep_pr, current_turn)
        
        # 步骤4: 生成状态数据包（EPJ v1.0 基础）
        state_packet = self.calculator.generate_state_packet(current_turn)
        
        # 附加量表信息（供Director参考）
        state_packet['filled_mdep_pr'] = filled_mdep_pr
        
        # EPM v2.0: 添加能量动力学摘要（如果启用）
        if self.enable_epm and self.calculator.v_star_0 is not None:
            from Benchmark.epj.scoring import get_epm_state_summary
            
            # 获取最新的EPM指标
            latest_trajectory = self.calculator.trajectory[-1]
            if 'epm' in latest_trajectory:
                epm_data = latest_trajectory['epm']
                
                epm_summary = get_epm_state_summary(
                    current_turn=current_turn,
                    r_t=epm_data['P_norm'],
                    projection=epm_data['projection'],
                    E_total=epm_data['E_total'],
                    epsilon_distance=self.calculator.epsilon_distance_relative,
                    epsilon_direction=self.calculator.epsilon_direction_relative,
            epsilon_energy=self.calculator.epsilon_energy,
            trajectory=self.calculator.trajectory,
            min_turns=12  # 🔧 显式传递最小轮次限制，确保EPM v2.0不会过早判定成功
        )
                
                state_packet['epm_summary'] = epm_summary
                
                # 如果EPM判定成功，添加标记（Director可用于决策）
                if epm_summary['success']:
                    print(f"🎉 [EPM] 检测到{epm_summary['victory_type']}胜利!")
        
        return state_packet
    
    def should_evaluate(self, current_turn: int) -> bool:
        """
        判断是否应该在本轮进行评估
        
        Args:
            current_turn: 当前轮次
        
        Returns:
            bool: 是否应该评估
        """
        # 只在K的倍数轮次评估
        return (current_turn > 0) and (current_turn % self.K == 0)
    
    def get_current_position(self) -> Tuple[int, int, int]:
        """获取当前位置向量"""
        return self.calculator.get_current_position()
    
    def get_trajectory(self) -> list:
        """获取完整轨迹"""
        return self.calculator.get_trajectory()
    
    def get_initial_deficit(self) -> Tuple[int, int, int]:
        """获取初始赤字向量"""
        return self.calculator.P_0 if self.calculator.P_0 else (0, 0, 0)


# ═══════════════════════════════════════════════════════════════
# 测试
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("EPJOrchestrator 集成测试")
    print("=" * 70)
    
    # 模拟Judger
    class MockJudger:
        def fill_iedr(self, script_content):
            return {
                "C.1": 2, "C.2": 1,
                "A.1": 2, "A.2": 1, "A.3": 2,
                "P.1": 2, "P.2": 3, "P.3": 3,
                "reasoning": "测试数据"
            }
        
        def fill_mdep_pr(self, recent_turns):
            return {
                "C.Prog": 1, "C.Neg": 0,
                "A.Prog": 2, "A.Neg": 0,
                "P.Prog": 1, "P.Neg": 0,
                "reasoning": "测试数据"
            }
    
    # 初始化
    judger = MockJudger()
    epj_orch = EPJOrchestrator(judger, K=3, max_turns=30)
    
    # T=0: 初始化
    print("\n1. T=0 初始化")
    script_content = {
        "actor_prompt": "刘静，26岁，设计师...",
        "scenario": {"剧本编号": "script_001"}
    }
    init_result = epj_orch.initialize_at_T0(script_content)
    
    # 模拟对话循环
    print("\n2. 模拟对话循环")
    for turn in range(1, 16):
        if epj_orch.should_evaluate(turn):
            print(f"\n--- 第{turn}轮（应该评估） ---")
            
            # 模拟最近K轮对话
            recent_turns = [
                {"actor": f"Actor消息{i}", "test_model": f"AI回复{i}"}
                for i in range(turn - epj_orch.K + 1, turn + 1)
            ]
            
            # 执行评估
            state_packet = epj_orch.evaluate_at_turn_K(recent_turns, turn)
            
            print(f"\n📦 状态数据包摘要:")
            print(f"   位置: {state_packet['P_t_current_position']}")
            print(f"   距离: {state_packet['distance_to_goal']}")
            print(f"   达标: {state_packet['is_in_zone']}")
        else:
            print(f"  第{turn}轮（不评估）")
    
    # 显示完整轨迹
    print(f"\n3. 完整评估轨迹")
    trajectory = epj_orch.get_trajectory()
    for point in trajectory:
        print(f"   T={point['turn']}: P={point['P_t']}, 距离={point['distance']:.2f}, 达标={point['in_zone']}")

