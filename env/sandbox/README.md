# EPJ Benchmark - 共情进度判断基准测试系统

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**EPJ (Empathy Progress Judge)** 是一个基于三维向量空间（C-A-P）的科学量化评估系统，用于评估AI模型在共情对话中的表现。

---

## 📋 目录

- [系统概述](#系统概述)
- [核心特性](#核心特性)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [使用指南](#使用指南)
- [配置说明](#配置说明)
- [结果分析](#结果分析)
- [常见问题](#常见问题)
- [贡献指南](#贡献指南)

---

## 🎯 系统概述

EPJ系统采用**三层架构**设计：

1. **Judger (传感器)** - 填写心理量表（IEDR、MDEP-PR）
2. **Orchestrator (计算器)** - 计算向量和生成状态数据包
3. **Director (决策者)** - 基于状态数据包做决策

### 评估维度

系统基于**三维共情空间**进行评估：

- **C轴（认知共情）**: 理解、澄清、认知重构
- **A轴（情感共情）**: 情感验证、情感共鸣
- **P轴（动机共情）**: 价值肯定、动机赋能

### 核心指标

- **IEDR (Initial Empathy Deficit Rubric)**: 初始共情赤字量表
- **MDEP-PR (Multi-Dimensional Empathy Progress Rubric)**: 多维度共情进展量表
- **EPM v2.0**: 能量动力学模型（Energy Progress Model）

---

## ✨ 核心特性

- ✅ **科学量化**: 基于向量空间的三维量化评估
- ✅ **自动化评估**: 支持批量评估和自动化流程
- ✅ **多模型支持**: 支持OpenAI、Anthropic、自定义API等
- ✅ **可视化分析**: 提供轨迹可视化和统计分析
- ✅ **可扩展性**: 模块化设计，易于扩展和定制

---

## 🚀 快速开始

### 1. 环境要求

- Python 3.8+
- 8GB+ RAM（推荐16GB）
- API密钥（OpenRouter/OpenAI/Anthropic等）

### 2. 安装依赖

```bash
# 克隆仓库
git clone <repository-url>
cd sandbox

# 安装依赖
pip install -r requirements.txt
```

### 3. 配置API密钥

创建 `.env` 文件（参考 `.env.example`）：

```bash
# Benchmark/topics/.env
OPENROUTER_API_KEY=your_api_key_here
```

或编辑 `config/api_config.py`：

```python
OPENROUTER_API_KEY = "your_api_key_here"
```

### 4. 运行基准测试

```bash
# 使用自定义模型API
python runner/run_benchmark_custom_model.py

# 使用SOTA模型（OpenRouter）
python runner/run_benchmark_sota_model.py

# 使用种子模型
python runner/run_benchmark_seed_model.py
```

---

## 📁 项目结构

```
sandbox/
├── Benchmark/              # 核心代码库
│   ├── agents/            # 四个核心Agent
│   │   ├── actor.py       # Actor（倾诉者）
│   │   ├── director.py   # Director（导演）
│   │   ├── judger.py     # Judger（评估员）
│   │   └── test_model.py # TestModel（被测模型）
│   ├── epj/              # EPJ系统核心
│   │   ├── vector_calculator.py  # 向量计算器
│   │   ├── scoring.py            # 评分系统
│   │   ├── judger_prompts.py     # Judger Prompt模板
│   │   └── RUBRICS_DEFINITION.md # 量表定义文档
│   ├── orchestrator/     # 编排器
│   │   └── chat_loop_epj.py      # EPJ对话循环
│   ├── prompts/          # Prompt模板
│   ├── topics/           # 剧本数据（395个剧本）
│   └── llms/             # LLM API封装
├── runner/                # 运行脚本
│   ├── benchmark_cases/   # 基准案例配置
│   └── run_benchmark_*.py # 运行脚本
├── scripts/               # 分析和评估脚本
│   ├── analysis/         # 数据分析
│   ├── evaluation/       # IEDR评估
│   └── utils/            # 工具脚本
├── results/              # 结果数据
│   └── benchmark_runs/   # 运行结果
├── config/               # 配置文件
└── README.md             # 本文件
```

---

## 📖 使用指南

### 运行单个案例

```python
from Benchmark.agents import Actor, Director, Judger
from Benchmark.orchestrator.chat_loop_epj import run_chat_loop_epj

# 初始化Agents
actor = Actor(model_name="your_actor_model")
director = Director(scenario=scenario, actor_prompt=actor_prompt)
judger = Judger(model_name="your_judger_model")
test_model = YourTestModel()

# 运行对话循环
result = run_chat_loop_epj(
    actor=actor,
    director=director,
    judger=judger,
    test_model=test_model,
    script_id="001",
    max_turns=30,
    K=3
)
```

### 批量评估IEDR

```bash
python scripts/evaluation/batch_evaluate_iedr.py
```

### 分析结果

```bash
# 生成统计分析
python scripts/analysis/generate_iedr_report.py

# 生成案例对比表
python scripts/analysis/generate_case_comparison_table.py
```

---

## ⚙️ 配置说明

### API配置

支持多种API提供商：

1. **OpenRouter** (推荐)
   ```python
   OPENROUTER_API_KEY=your_key
   ```

2. **OpenAI**
   ```python
   OPENAI_API_KEY=your_key
   ```

3. **自定义API**
   ```python
   # 在runner脚本中配置
   CUSTOM_API_CONFIG = {
       "api_key": "your_key",
       "base_url": "https://your-api.com/v1",
       "model_name": "your_model"
   }
   ```

### 模型配置

在运行脚本中配置使用的模型：

```python
ACTOR_MODEL = "anthropic/claude-3.5-sonnet"
DIRECTOR_MODEL = "anthropic/claude-3.5-sonnet"
JUDGER_MODEL = "anthropic/claude-3.5-sonnet"
```

### 评估参数

```python
MAX_TURNS = 30      # 最大对话轮次
K = 3              # 评估周期（每K轮评估一次）
THRESHOLD_TYPE = "high_threshold"  # 阈值类型
```

---

## 📊 结果分析

### 结果文件结构

```
results/benchmark_runs/{model_name}_{timestamp}/
├── script_001_result.json    # 单个案例结果
├── script_002_result.json
├── ...
├── summary.json              # 汇总统计
└── descriptive_statistics.md # 描述性统计
```

### 结果字段说明

```json
{
  "script_id": "001",
  "total_turns": 15,
  "termination_reason": "EPM_SUCCESS",
  "epj": {
    "P_0_initial_deficit": [-15, -18, -16],
    "P_final_position": [-2, -1, 0],
    "trajectory": [...],
    "epm_victory_analysis": {...}
  }
}
```

### 可视化

```bash
# 生成轨迹可视化
python scripts/visualization/visualize_trajectories_plotly.py

# 生成能力雷达图
python scripts/analysis/visualize_capability_radar.py
```

---

## ❓ 常见问题

### Q1: 如何添加新的测试模型？

A: 实现 `TestModel` 接口，或参考 `runner/run_benchmark_custom_model.py` 中的 `CustomTestModel` 类。

### Q2: 如何自定义评估标准？

A: 修改 `Benchmark/epj/rubrics.py` 和 `Benchmark/epj/judger_prompts.py` 中的量表定义。

### Q3: 评估需要多长时间？

A: 单个案例约5-15分钟（取决于模型响应速度）。批量评估395个案例需要数小时。

### Q4: 如何理解EPJ评分？

A: 参考 `Benchmark/epj/RUBRICS_DEFINITION.md` 了解详细的评分标准。

### Q5: 支持哪些模型？

A: 支持所有兼容OpenAI API格式的模型，包括：
- OpenAI GPT系列
- Anthropic Claude系列
- 自部署模型（通过OpenAI兼容API）

---

## 🤝 贡献指南

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

---

## 📧 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 Issue
- 发送邮件至 [your-email@example.com]

---

## 🙏 致谢

感谢所有为本项目做出贡献的开发者和研究者。

---

**最后更新**: 2025-01-XX  
**版本**: 1.0.0

