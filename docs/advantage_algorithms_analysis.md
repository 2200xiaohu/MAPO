# 多轮对话 RL 训练 Advantage 算法对比分析

## 概述

本文档详细对比分析了四种多轮对话场景下的 Advantage 估计算法：

1. **GRPO Multiturn** (grpo_multiturn)
2. **REINFORCE++ Multiturn V2** (reinforce_plus_plus_multiturn_v2)
3. **REINFORCE++ Multiturn V3** (reinforce_plus_plus_multiturn_v3)
4. **REINFORCE++ Multiturn Baseline** (reinforce_plus_plus_multiturn_baseline)

代码位置: /nas/naifan/verl_reinforce_pp/verl/trainer/ppo/core_algos.py

---

## 算法对比表

| 特性 | GRPO Multiturn | V2 | V3 | Baseline |
|------|---------------|-----|-----|----------|
| Return 计算 | 直接 reward sum | Gamma 累积 | Gamma 累积 | Gamma 累积 |
| 分组策略 | (prompt_id, turn_idx) | prompt_id (混合所有轮次) | 不分组 | (prompt_id, turn_idx) |
| 归一化方式 | (R - mean) / std | (R - mean) / std | 无归一化 | 减 baseline 后 whitening |
| 单样本处理 | adv = -1000 (mask掉) | mean=0, std=1 | N/A | baseline=0 |
| 轮次对齐 | 严格对齐 | 不对齐 | 不关心 | 严格对齐 |

---

## 详细算法分析

### 1. GRPO Multiturn

**核心思想**: 按 (prompt_id, turn_idx) 严格分组,同一prompt的相同轮次位置互相比较。

**算法步骤**:

```
Step 1: 数据收集
for each sample i:
    for each turn (start, end) in sample:
        score = reward[start:end].sum()
        group_scores[(uid, turn_idx)].append(score)
        
Step 2: 统计计算
for each (uid, turn_idx) group:
    if len(group) < 2:
        stats = {valid: False}  # 单样本组标记为无效
    else:
        stats = {mean: mean(scores), std: std(scores)}

Step 3: 优势计算
for each turn:
    if not valid:
        adv = -1000.0  # 特殊标记,后续会被 mask 掉
    else:
        adv = (score - mean) / (std + epsilon)
```

**特点**:
- 轮次级别严格对齐比较
- 无效组 (adv=-1000) 会被后续处理 mask 掉
- 如果某轮次只有一个有效样本,该轮次数据会被丢弃

---

### 2. REINFORCE++ Multiturn V2

**核心思想**: 按 prompt_id 分组,但所有轮次的 return 混合在一起归一化。

**算法步骤**:

```
Step 1: 计算 Returns (gamma 累积,轮次边界重置)
for t in reversed(range(seq_len)):
    running_return = reward[:, t] + gamma * running_return
    returns[:, t] = running_return
    running_return = running_return * response_mask[:, t]  # 边界重置!

Step 2: 提取轮次 return
for each sample:
    for each turn:
        turn_return = returns[sample_idx, turn_start]
        prompt_groups[prompt_id].append(turn_return)  # 不区分轮次位置

Step 3: 按 prompt 归一化
for each prompt_id:
    mean = mean(all_turn_returns)
    std = std(all_turn_returns)
    
for each turn:
    adv = (turn_return - mean) / (std + epsilon)
```

**特点**:
- Gamma 折扣累积 return
- 每个轮次的 return 是独立的(边界重置)
- 不同轮次的 return 混合归一化
- 单样本 prompt: mean=0, std=1,导致 advantage = turn_return

---

### 3. REINFORCE++ Multiturn V3

**核心思想**: 最简单的方案,直接使用 return 作为 advantage,不做任何归一化。

**算法步骤**:

```
Step 1: 计算 Returns (与 V2 相同)
for t in reversed(range(seq_len)):
    running_return = reward[:, t] + gamma * running_return
    returns[:, t] = running_return
    running_return = running_return * response_mask[:, t]

Step 2: 直接使用 return 作为 advantage
advantages = returns * response_mask
```

**特点**:
- 最简单,计算开销最小
- 不依赖分组,避免单样本问题
- 没有归一化,advantage 尺度可能很大或很小
- 可能导致训练不稳定

---

### 4. REINFORCE++ Multiturn Baseline

**核心思想**: 结合 GRPO 的轮次对齐和 V2 的 gamma 累积,引入 baseline 机制。

**算法步骤**:

```
Step 1: 计算 Returns (与 V2 相同)
for t in reversed(range(seq_len)):
    running_return = reward[:, t] + gamma * running_return
    returns[:, t] = running_return
    running_return = running_return * response_mask[:, t]

Step 2: 按 (prompt_id, turn_idx) 分组
grouped[prompt_id][turn_idx].append(turn_return)

Step 3: 计算 baseline
for each (prompt_id, turn_idx):
    if len(group) > 1:
        baseline = mean(returns)
    else:
        baseline = 0.0  # 回退策略

Step 4: 减去 baseline 并全局 whitening
for each turn:
    adv = turn_return - baseline
    
all_advantages = whitening(all_advantages)
```

**特点**:
- Gamma 折扣累积
- 轮次级别 baseline,降低方差
- 全局 whitening 保持尺度一致
- 单样本组 baseline=0,可能不是最优选择

---

## 数值示例

假设有 2 个 prompt,每个 prompt 有 2 个 rollout,每个 rollout 有 3 轮对话:

```
Prompt 0:
  Rollout 0: Turn 0 reward=1.0, Turn 1 reward=0.5, Turn 2 reward=0.8
  Rollout 1: Turn 0 reward=0.3, Turn 1 reward=0.6, Turn 2 reward=0.4

Prompt 1:
  Rollout 2: Turn 0 reward=0.7, Turn 1 reward=0.2, Turn 2 reward=0.9
  Rollout 3: Turn 0 reward=0.5, Turn 1 reward=0.8, Turn 2 reward=0.1
```

### 各算法计算结果

#### GRPO Multiturn (直接用 reward)
| (Prompt, Turn) | Group Scores | Mean | Std | Sample 0 Adv | Sample 1 Adv |
|----------------|--------------|------|-----|--------------|--------------|
| (0, 0) | [1.0, 0.3] | 0.65 | 0.35 | +1.0 | -1.0 |
| (0, 1) | [0.5, 0.6] | 0.55 | 0.05 | -1.0 | +1.0 |
| (0, 2) | [0.8, 0.4] | 0.60 | 0.20 | +1.0 | -1.0 |
| (1, 0) | [0.7, 0.5] | 0.60 | 0.10 | +1.0 | -1.0 |
| ... | ... | ... | ... | ... | ... |

#### V2 (gamma=1.0,按 prompt 混合)
Prompt 0 的所有 returns: [1.0, 0.5, 0.8, 0.3, 0.6, 0.4]
- 全部 mean = 0.6, std = 0.235
- 归一化后各轮 advantage 不同

#### V3 (无归一化)
- 直接使用 return: Turn 0 adv = 1.0, Turn 1 adv = 0.5, Turn 2 adv = 0.8, ...

#### Baseline
- Turn 0 baseline (prompt 0) = mean([1.0, 0.3]) = 0.65
- Sample 0, Turn 0: adv_raw = 1.0 - 0.65 = 0.35
- 最后全局 whitening

---

## 关键问题与建议

### 1. Return 计算的轮次独立性

**重要澄清**: V2/V3/Baseline 中,每个轮次的 return 是**独立**的:

```python
running_return = running_return * response_mask[:, t]  # 关键!
```

当 response_mask[t] = 0 (user token),running_return 被重置为 0,确保:
- Turn 2 的 return = 只累积 Turn 2 的 rewards
- Turn 1 的 return = 只累积 Turn 1 的 rewards
- Turn 0 的 return = 只累积 Turn 0 的 rewards

### 2. 单样本组处理差异

| 算法 | 单样本组处理 | 潜在问题 |
|------|-------------|---------|
| GRPO | adv = -1000,后续 mask | 数据损失 |
| V2 | mean=0, std=1,adv=return | 未归一化,尺度不一致 |
| V3 | 不存在此问题 | N/A |
| Baseline | baseline=0 | 与其他组不一致 |

**建议**: 单样本情况下,可考虑使用全局统计量作为回退。

### 3. 算法选择建议

- **数据充足、轮次对齐重要**: 使用 GRPO Multiturn 或 Baseline
- **数据稀疏、避免样本损失**: 使用 V2 或 V3
- **训练不稳定**: 避免 V3,使用有归一化的变体
- **计算效率优先**: V3 最快

---

## 代码位置参考

```
文件: /nas/naifan/verl_reinforce_pp/verl/trainer/ppo/core_algos.py

GRPO Multiturn: 行 332-426
@register_adv_est(AdvantageEstimator.GRPO_MULTITURN)
def compute_grpo_mutliturn_outcome_advantage(...)

V2: 行 1598-1690
@register_adv_est(AdvantageEstimator.REINFORCE_PLUS_PLUS_MULTITURN_V2)
def compute_reinforce_plus_plus_multiturn_v2_advantage(...)

V3: 行 1693-1831
@register_adv_est(AdvantageEstimator.REINFORCE_PLUS_PLUS_MULTITURN_V3)
def compute_reinforce_plus_plus_multiturn_v3_advantage(...)

Baseline: 行 1834-1962
@register_adv_est(AdvantageEstimator.REINFORCE_PLUS_PLUS_MULTITURN_BASELINE)
def compute_reinforce_plus_plus_multiturn_baseline_advantage(...)
```

---

## 更新日志

- **2025-01-11**: 初始版本,完成四种算法的对比分析
