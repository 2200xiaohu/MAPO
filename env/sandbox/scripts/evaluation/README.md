# Evaluation Scripts - IEDR评估脚本

本目录包含IEDR（Initial Empathy Deficit Rubric）评估相关的脚本。

---

## 📊 脚本列表（3个）

### 1. `batch_evaluate_iedr.py`
**功能**: 批量评估初始共情赤字（IEDR）

**评估维度**:
- **C轴（认知共情）**: C.1处境复杂性、C.2深度、C.3认知优先级
- **A轴（情感共情）**: A.1情绪强度、A.2情绪可及性、A.3情感优先级
- **P轴（动机共情）**: P.1初始能动性、P.2价值关联度、P.3动机优先级

**输入**: 
- `Benchmark/topics/new_data/character_setting/script_*.md`
- `Benchmark/topics/new_data/scenarios/character_stories.json`

**输出**: 
- `scripts/results/iedr_batch_results_new.json`

**配置**:
```python
START_SCRIPT = 1          # 起始剧本编号
END_SCRIPT = 395          # 结束剧本编号
MAX_WORKERS = 5           # 并发数
JUDGER_MODEL = "anthropic/claude-3.5-sonnet"
```

**使用方法**:
```bash
python3 scripts/evaluation/batch_evaluate_iedr.py
```

---

### 2. `analyze_iedr_distribution.py`
**功能**: 分析IEDR分布统计

**分析内容**:
- 各指标的均值、中位数、标准差
- 三轴赤字分布（C/A/P）
- 总赤字（欧氏距离）
- 轴主导类型统计

**输入**: 
- `scripts/results/iedr_batch_results_new.json`

**输出**: 
- `scripts/results/iedr_distribution_analysis.txt`

**使用方法**:
```bash
python3 scripts/evaluation/analyze_iedr_distribution.py
```

**输出示例**:
```
=== IEDR分布分析 ===

1. 各指标得分分布
   C.1 处境复杂性: 均值=2.54, 中位数=3.0, 标准差=0.59
   C.2 深度: 均值=2.86, 中位数=3.0, 标准差=0.35
   ...

2. 三轴赤字分布
   C轴: 均值=-17.17, 范围=[-21, -7]
   A轴: 均值=-19.11, 范围=[-27, -3]
   P轴: 均值=-19.02, 范围=[-27, -7]
```

---

### 3. `generate_iedr_report.py`
**功能**: 生成IEDR综合分析报告

**报告内容**:
1. 执行摘要和核心发现
2. 各指标详细统计
3. 三轴赤字分布分析
4. 总赤字和难度分布
5. 轴主导分析
6. 维度相关性
7. 偏好检测和问题识别
8. 改进建议和行动计划
9. 预期效果

**输入**: 
- `scripts/results/iedr_batch_results_new.json`

**输出**: 
- `scripts/results/iedr_analysis_new/IEDR_Analysis_Report_New.md`
- `scripts/results/iedr_analysis_new/Risk_Assessment_Summary.md`
- 各类可视化图表（PNG）

**使用方法**:
```bash
python3 scripts/evaluation/generate_iedr_report.py
```

---

## 🔄 完整工作流程

```bash
# 步骤1: 批量评估IEDR（需要数小时）
python3 scripts/evaluation/batch_evaluate_iedr.py

# 步骤2: 分析分布统计
python3 scripts/evaluation/analyze_iedr_distribution.py

# 步骤3: 生成可视化图表（需要先运行可视化脚本）
python3 scripts/visualization/visualize_iedr_new.py

# 步骤4: 生成综合报告
python3 scripts/evaluation/generate_iedr_report.py
```

---

## 📊 评估标准

### IEDR评分规则

每个维度评分范围: 0-3

- **级别0**: 无赤字/最优状态
- **级别1**: 轻微赤字
- **级别2**: 中等赤字
- **级别3**: 严重赤字

### 三轴赤字计算

```python
# 每个轴的赤字 = -(各维度得分之和)
C_deficit = -(C.1 + C.2 + C.3)
A_deficit = -(A.1 + A.2 + A.3)
P_deficit = -(P.1 + P.2 + P.3)

# 总赤字 = 欧氏距离
total_deficit = sqrt(C_deficit^2 + A_deficit^2 + P_deficit^2)
```

---

## 📈 数据集统计

### 新数据集（395个剧本）

- **总剧本数**: 395
- **评估成功率**: 100%
- **平均总赤字**: 32.32
- **轴主导分布**:
  - C轴主导: 12.4%
  - A轴主导: 45.1%
  - P轴主导: 35.4%
  - 平局: 7.1%

---

## 🔧 配置说明

### API配置

在`config/api_config.py`中配置：

```python
OPENROUTER_API_KEY = "your_api_key_here"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
```

### 模型选择

推荐模型：
- `anthropic/claude-3.5-sonnet` (默认，最准确)
- `anthropic/claude-3-opus`
- `openai/gpt-4-turbo`

---

**最后更新**: 2025-11-14  
**维护者**: EPJ项目组

