# Benchmark/epj/rubrics.py
"""
量表元数据（Rubrics Metadata）

⚠️ 重要说明：
- 本文件只包含量表的元数据（名称、代码、用途等）
- 量表的详细定义（指标、级别、描述）请参考：Benchmark/epj/RUBRICS_DEFINITION.md
- Prompt中的量表说明是权威来源，与RUBRICS_DEFINITION.md保持同步

职责：
- 为代码提供量表的基础信息
- 避免在多处重复定义量表详情
"""

# ═══════════════════════════════════════════════════════════════
# A. 初始共情赤字量表 (IEDR) - 元数据
# ═══════════════════════════════════════════════════════════════

IEDR_RUBRIC = {
    "name": "初始共情赤字量表 (Initial Empathy Deficit Rubric)",
    "code": "IEDR",
    "purpose": "量化剧本的初始共情赤字",
    "timing": "T=0 (对话开始前)",
    "filled_by": "Judger (LLM)",
    "definition_source": "Benchmark/epj/RUBRICS_DEFINITION.md",
    
    # 基本结构信息
    "dimensions": ["C", "A", "P"],
    "indicators": {
        "C": ["C.1", "C.2", "C.3"],  # 认知共情：3个指标
        "A": ["A.1", "A.2", "A.3"],  # 情感共情：3个指标
        "P": ["P.1", "P.2", "P.3"]   # 动机共情：3个指标
    },
    "total_indicators": 9,
    "level_range": (0, 3),  # 所有指标都是0-3级
    
    # 输出格式（用于验证）
    "output_format": {
        "C.1": "int (0-3)",
        "C.2": "int (0-3)",
        "C.3": "int (0-3)",
        "A.1": "int (0-3)",
        "A.2": "int (0-3)",
        "A.3": "int (0-3)",
        "P.1": "int (0-3)",
        "P.2": "int (0-3)",
        "P.3": "int (0-3)",
        "reasoning": "str (简要判断依据)"
    }
}


# ═══════════════════════════════════════════════════════════════
# C. MDEP进展量表 (MDEP-PR) - 元数据
# ═══════════════════════════════════════════════════════════════

MDEP_PR_RUBRIC = {
    "name": "MDEP进展量表 (MDEP Progress Rubric)",
    "code": "MDEP-PR",
    "purpose": "量化每K轮对话中的共情进展/倒退",
    "timing": "T>0 (每K轮评估一次)",
    "filled_by": "Judger (LLM)",
    "definition_source": "Benchmark/epj/RUBRICS_DEFINITION.md",
    "K": 3,  # 默认每3轮评估一次（可在运行时调整）
    
    # 基本结构信息
    "dimensions": ["C", "A", "P"],
    "indicators": {
        "C": ["C_Prog", "C_Neg"],    # 认知：进展+倒退
        "A": ["A_Prog", "A_Neg"],    # 情感：进展+倒退
        "P": ["P_Prog", "P_Neg"]     # 动机：进展+倒退
    },
    "total_indicators": 6,
    "level_range": {
        "Prog": (0, 2),   # 进展：0/1/2
        "Neg": (-2, 0)    # 倒退：-2/-1/0
    },
    
    # 输出格式（新格式 - 带evidence和reasoning）
    "output_format": {
        "C_Prog_level": "int (0/1/2)",
        "C_Prog_evidence": "str (引用关键回应，0级别时为空)",
        "C_Prog_reasoning": "str (必填！说明判断依据)",
        "C_Neg_level": "int (0/-1/-2)",
        "C_Neg_evidence": "str (引用关键回应，0级别时为空)",
        "C_Neg_reasoning": "str (必填！说明判断依据)",
        # ... A和P维度同理
    },
    
    # 旧格式（用于向后兼容，scoring.py仍然接受这个格式）
    "legacy_output_format": {
        "C.Prog": "int (0/1/2)",
        "C.Neg": "int (0/-1/-2)",
        "A.Prog": "int (0/1/2)",
        "A.Neg": "int (0/-1/-2)",
        "P.Prog": "int (0/1/2)",
        "P.Neg": "int (0/-1/-2)",
        "reasoning": "str (判断依据)"
    }
}


def get_iedr_rubric():
    """
    获取初始共情赤字量表元数据
    
    Returns:
        dict: IEDR量表的元数据
    
    注意：详细的指标定义请参考 Benchmark/epj/RUBRICS_DEFINITION.md
    """
    return IEDR_RUBRIC


def get_mdep_pr_rubric():
    """
    获取MDEP进展量表元数据
    
    Returns:
        dict: MDEP-PR量表的元数据
    
    注意：详细的指标定义请参考 Benchmark/epj/RUBRICS_DEFINITION.md
    """
    return MDEP_PR_RUBRIC

