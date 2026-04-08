#!/usr/bin/env python3
"""
批量评估所有剧本的初始共情赤字（IEDR）

用途：
- 离线批量评估 character_setting/ 中所有剧本的 IEDR
- 使用详细的prompt（包含evidence和reasoning）
- 保存结果到JSON文件
"""

import json
import glob
import math
from pathlib import Path
from typing import Dict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 复用现有的Judger类和配置加载
from Benchmark.agents.judger import Judger
from Benchmark.topics.config_loader import ConfigLoader


# ═══════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).parent
OUTPUT_FILE = BASE_DIR / "results/iedr_batch_results.json"  # 正式输出文件（使用完整prompt）

# Judger模型配置
JUDGER_MODEL = "google/gemini-2.5-pro"

# 评估范围配置
START_SCRIPT = 101  # 从哪个编号开始
END_SCRIPT = 281    # 到哪个编号结束（包含）
TEST_LIMIT = None   # 测试模式：限制处理的剧本数量（None表示处理全部）

# 并发配置
MAX_WORKERS = 10    # 最大并发线程数（建议5-15，太高可能触发API限流）


# ═══════════════════════════════════════════════════════════════
# 详细版 IEDR Prompt（用于批量评估）
# ═══════════════════════════════════════════════════════════════

def generate_detailed_iedr_prompt(script_content: dict) -> str:
    """
    生成详细的IEDR评估prompt（带evidence和reasoning）
    
    输入格式与 judger_prompts.py 的 generate_iedr_prompt 一致
    """
    # 与 judger_prompts.py 保持一致的输入处理方式
    actor_prompt = script_content.get('actor_prompt', '')
    scenario = script_content.get('scenario', {})
    
    # 格式化故事的经过
    story_progress = scenario.get('故事的经过', {})
    story_progress_text = ""
    if story_progress:
        # 按阶段编号排序
        sorted_stages = sorted(story_progress.items(), key=lambda x: int(x[0].replace('阶段', '')))
        for stage_key, stage_data in sorted_stages:
            title = stage_data.get('标题', '')
            content = stage_data.get('内容', '')
            story_progress_text += f"\n**{stage_key}: {title}**\n{content}\n"
    else:
        story_progress_text = "\n(无详细故事经过)\n"
    
    # 获取故事的结果和插曲
    story_result = scenario.get('故事的结果', 'N/A')
    story_episode = scenario.get('故事的插曲', '')
    
    prompt = f"""# ROLE: 资深共情心理分析师 & 角色画像解构专家

你是一位资深的共情心理分析师，拥有深厚的心理学知识和丰富的案例分析经验。你的核心专长是：
1. **深度解构 (Deconstruction)**：能够从复杂的角色背景、经历和需求描述中，精准提炼出其在认知 (C)、情感 (A)、动机 (P) 三个维度上的核心心理状态和潜在"赤字"。
2. **循证推理 (Evidence-Based Reasoning)**：你的所有分析**必须**严格基于提供的文本证据，并清晰阐述从证据到结论的逻辑链条。
3. **客观校准 (Objective Calibration)**：你的任务是将角色的初始状态**校准**到《初始共情赤字量表 (IEDR)》的相应级别上，为后续的量化计算提供**可靠的、非主观的**基础。

你的工作是**绝对客观、中立且基于证据**的。你**不**做预测，**不**做决策，**不**输出任何最终的数字分数。你的**唯一**任务是：
1. **深度分析**下方提供的 `Actor Profile` (包含所有子部分)。
2. **严格按照**《IEDR 量表》的定义和级别描述。
3. 为量表中的**每一项**指标，选择**一个最符合**的定性级别 (`[0]`-`[3]`)。
4. 为每一个**非零**级别的选择，提供**明确的文本证据 (`evidence`)** 和**详细的心理分析理由 (`reasoning`)**。

# CONTEXT: 角色完整画像与剧本信息

## Actor Profile (角色画像)
```
{actor_prompt}
```

## Scenario (场景信息 - 完整故事)

**剧本编号**: {scenario.get('剧本编号', 'N/A')}

### 故事的经过
{story_progress_text}

### 故事的结果
{story_result}

{f"### 故事的插曲{chr(10)}{story_episode}" if story_episode else ""}

---

# TASK: 科学填写 IEDR 量表并提供详细理由 (核心任务)

你的**唯一**任务是：基于对上述 `Actor Profile` 的**全面、深度分析**，**确定**该角色在**对话开始前 (T=0)** 的**初始共情赤字状态**，并将其**映射**到《IEDR 量表》的相应级别上。

* **禁止臆断**: 你的判断**必须**有据可循。如果剧本没有提供足够信息来支持某个指标的非零判断，则**必须**选择 `[0]` 级别。
* **证据优先**: **直接引用**剧本中的关键词句作为 `evidence`。
* **理由是关键**: `reasoning` **必须**清晰地解释：
    * 你是如何**解读** `evidence` 的？
    * 这段证据**如何**与该指标项的**量表定义**（例如 C.1 的"特定领域知识"）相关联？
    * 这个关联**为什么**支持你选择的**特定级别**（例如 `[2]` 而不是 `[1]` 或 `[3]`）
    * (如果适用) 这个判断如何与其他 `Actor Profile` 信息（如经历、优先级）相呼应？
* **零级**: 如果选择 `[0]` 级别，`evidence` 也需要填写 `"0"`,并且 `reasoning` 字段也需要填写**理由**。

# IEDR 量表定义 (必须严格遵守)

## C轴 (认知共情 - 被理解)

### C.1 处境复杂性
理解此处境所需的背景知识复杂度

- `[0]` **(无)** 普遍的日常经验
- `[1]` **(低)** 基础的社交或生活常识
- `[2]` **(中)** 特定的领域知识（如"甲方/乙方"）
- `[3]` **(高)** 角色独特的、复杂的个人经历

### C.2 深度
情感或认知的深度层次

- `[0]` **(表层)**
- `[1]` **(浅层)**
- `[2]` **(中层)**
- `[3]` **(深层)**

### C.3 认知优先级
角色对"被理解"的需求优先级

- `[0]` **(无关)**
- `[1]` **(次要)**
- `[2]` **(核心)**
- `[3]` **(最高)**

---

## A轴 (情感共情 - 被共鸣)

### A.1 情绪强度
初始情绪的强烈程度

- `[0]` **(平静)**
- `[1]` **(轻微)**
- `[2]` **(强烈)**
- `[3]` **(极端)**

### A.2 情绪可及性
情绪的表达难度

- `[0]` **(清晰)**
- `[1]` **(隐含)**
- `[2]` **(掩饰/冲突)**
- `[3]` **(深度)**

### A.3 情感优先级
角色对"被共鸣"的需求优先级

- `[0]` **(无关)**
- `[1]` **(次要)**
- `[2]` **(核心)**
- `[3]` **(最高)**

---

## P轴 (动机共情 - 被肯定/赋能)

### P.1 初始能动性
角色的初始行动能力

- `[0]` **(高能动性)**
- `[1]` **(中能动性)**
- `[2]` **(低能动性)**
- `[3]` **(无能动性)**

### P.2 价值关联度
困境对核心价值观的冲击程度

- `[0]` **(无关)**
- `[1]` **(轻微)**
- `[2]` **(核心)**
- `[3]` **(危机/核心追求)**

### P.3 动机优先级
角色对"被肯定/赋能"的需求优先级

- `[0]` **(无关)**
- `[1]` **(次要)**
- `[2]` **(核心)**
- `[3]` **(最高)**

---

# 输出格式 (Strict JSON Output with Reasoning)

你的**唯一**输出必须是一个**严格符合以下结构**的 JSON 对象。**禁止**包含任何 JSON 之外的内容。

```json
{{{{
  "C.1_level": <Selected level 0-3>,
  "C.1_evidence": "<Direct quote if level != 0, else '0'>",
  "C.1_reasoning": "<Detailed justification if level != 0, else brief explanation why it's 0>",
  "C.2_level": <Selected level 0-3>,
  "C.2_evidence": "<Direct quote if level != 0, else '0'>",
  "C.2_reasoning": "<Detailed justification if level != 0, else brief explanation why it's 0>",
  "C.3_level": <Selected level 0-3>,
  "C.3_evidence": "<Direct quote if level != 0, else '0'>",
  "C.3_reasoning": "<Detailed justification if level != 0, else brief explanation why it's 0>",
  "A.1_level": <Selected level 0-3>,
  "A.1_evidence": "<Direct quote if level != 0, else '0'>",
  "A.1_reasoning": "<Detailed justification if level != 0, else brief explanation why it's 0>",
  "A.2_level": <Selected level 0-3>,
  "A.2_evidence": "<Direct quote if level != 0, else '0'>",
  "A.2_reasoning": "<Detailed justification if level != 0, else brief explanation why it's 0>",
  "A.3_level": <Selected level 0-3>,
  "A.3_evidence": "<Direct quote if level != 0, else '0'>",
  "A.3_reasoning": "<Detailed justification if level != 0, else brief explanation why it's 0>",
  "P.1_level": <Selected level 0-3>,
  "P.1_evidence": "<Direct quote if level != 0, else '0'>",
  "P.1_reasoning": "<Detailed justification if level != 0, else brief explanation why it's 0>",
  "P.2_level": <Selected level 0-3>,
  "P.2_evidence": "<Direct quote if level != 0, else '0'>",
  "P.2_reasoning": "<Detailed justification if level != 0, else brief explanation why it's 0>",
  "P.3_level": <Selected level 0-3>,
  "P.3_evidence": "<Direct quote if level != 0, else '0'>",
  "P.3_reasoning": "<Detailed justification if level != 0, else brief explanation why it's 0>"
}}}}
```

**🔧 极其重要的要求**：
- **只输出JSON，不要其他任何内容**（不要输出任何markdown标记、说明文字、代码块标记，只输出纯JSON对象）
- **evidence字段**: 
  * level=0时填写 `"0"`
  * level!=0时必须包含直接引用的关键词句
- **reasoning字段**: 
  * level=0时简要说明为什么是0级（例如："处境为普遍日常经验"）
  * level!=0时必须包含详细的心理分析和证据解读
- **确保JSON完整，避免被截断**（所有27个字段都必须完整输出）

**⚠️ 严禁输出的内容**：
- 不要输出 ```json 这样的代码块标记
- 不要输出任何解释性文字
- 只输出一个合法的JSON对象，从 {{ 开始，到 }} 结束
"""
    
    return prompt


def convert_to_standard_format(detailed_response: dict) -> dict:
    """将详细格式转换为标准IEDR格式"""
    return {
        "C.1": detailed_response.get('C.1_level', 0),
        "C.2": detailed_response.get('C.2_level', 0),
        "C.3": detailed_response.get('C.3_level', 0),
        "A.1": detailed_response.get('A.1_level', 0),
        "A.2": detailed_response.get('A.2_level', 0),
        "A.3": detailed_response.get('A.3_level', 0),
        "P.1": detailed_response.get('P.1_level', 0),
        "P.2": detailed_response.get('P.2_level', 0),
        "P.3": detailed_response.get('P.3_level', 0),
        "_detailed": detailed_response  # 保留详细信息
    }


def evaluate_single_script(script_id: str, config_loader: ConfigLoader, judger: Judger) -> dict:
    """
    评估单个剧本的IEDR（并发安全版本）
    
    复用现有的配置加载逻辑，只是使用更详细的prompt
    注意：在并发环境中，详细日志被移除以避免输出混乱
    """
    try:
        # 提取数字ID（ConfigLoader期望"001"格式，而不是"script_001"）
        numeric_id = script_id.replace("script_", "")
        
        # 复用ConfigLoader加载剧本数据
        script_config = config_loader.load_config(numeric_id)
        
        # 构造script_content（与judger.fill_iedr的输入格式一致）
        script_content = {
            'actor_prompt': script_config['actor_prompt'],
            'scenario': script_config['scenario']
        }
        
        # 生成详细prompt（替代judger_prompts.py中的简化版本）
        from Benchmark.llms.api import get_llm_response
        prompt = generate_detailed_iedr_prompt(script_content)
        
        # 调用LLM（复用judger的API调用逻辑）
        response = get_llm_response(
            messages=[{"role": "user", "content": prompt}],
            model_name=JUDGER_MODEL,
            json_mode=True,
            max_tokens=8000  # 增加到8000确保完整输出27个字段
        )
        
        # 解析响应（复用judger的解析逻辑）
        detailed_result = judger._parse_rubric_response(response)
        
        # 转换为标准格式
        standard_result = convert_to_standard_format(detailed_result)
        
        # 🔧 使用 scoring.py 计算初始赤字向量 P_0
        from Benchmark.epj.scoring import calculate_initial_deficit
        P_0 = calculate_initial_deficit(standard_result)
        
        # 🔧 使用欧氏距离计算总赤字
        total_deficit = math.sqrt(P_0[0]**2 + P_0[1]**2 + P_0[2]**2)
        
        # 🆕 EPM v2.0: 预计算EPM参数
        epm_params = None
        if total_deficit > 0:
            # 计算理想方向向量 v*_0 = -P_0 / ||P_0||
            v_star_0 = tuple(-x / total_deficit for x in P_0)
            
            # 计算相对阈值（α = 0.10）
            alpha = 0.10
            epsilon_distance = alpha * total_deficit
            epsilon_direction = alpha * total_deficit
            epsilon_energy = total_deficit
            
            epm_params = {
                "P_0_norm": round(total_deficit, 2),
                "v_star_0": [round(x, 4) for x in v_star_0],  # 理想方向（归一化）
                "epsilon_distance": round(epsilon_distance, 2),
                "epsilon_direction": round(epsilon_direction, 2),
                "epsilon_energy": round(epsilon_energy, 2),
                "alpha": alpha
            }
        
        result = {
            "script_id": script_id,
            "status": "success",
            "iedr": standard_result,
            "P_0": {
                "C": P_0[0],
                "A": P_0[1],
                "P": P_0[2],
                "total": round(total_deficit, 2)
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # 添加EPM参数（如果计算成功）
        if epm_params:
            result["epm"] = epm_params
        
        return result
        
    except Exception as e:
        return {
            "script_id": script_id,
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


# ═══════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════

def main():
    """批量评估所有剧本的IEDR"""
    
    print("="*70)
    print("🚀 EPJ批量IEDR评估系统")
    print("="*70)
    
    # 初始化ConfigLoader和Judger（复用现有组件）
    config_loader = ConfigLoader()
    judger = Judger(model_name=JUDGER_MODEL)
    
    # 🔧 读取现有结果（如果存在）
    existing_results = []
    evaluated_ids = set()
    if OUTPUT_FILE.exists():
        print(f"\n📂 读取现有结果: {OUTPUT_FILE}")
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            existing_results = json.load(f)
        evaluated_ids = {r['script_id'] for r in existing_results}
        print(f"✅ 已有 {len(existing_results)} 个剧本的评估结果")
    
    # 🔧 生成待评估的剧本ID列表（从START_SCRIPT到END_SCRIPT）
    all_script_ids = [f"script_{i:03d}" for i in range(START_SCRIPT, END_SCRIPT + 1)]
    
    # 过滤掉已经评估过的脚本
    new_script_ids = [sid for sid in all_script_ids if sid not in evaluated_ids]
    
    print(f"\n📁 总共需要评估: script_{START_SCRIPT:03d} 到 script_{END_SCRIPT:03d} (共 {len(all_script_ids)} 个)")
    print(f"✅ 已评估: {len(all_script_ids) - len(new_script_ids)} 个")
    print(f"🆕 待评估: {len(new_script_ids)} 个")
    
    # 应用测试限制
    if TEST_LIMIT is not None:
        script_ids = new_script_ids[:TEST_LIMIT]
        print(f"⚠️ 测试模式：只处理前 {TEST_LIMIT} 个剧本")
    else:
        script_ids = new_script_ids
    
    # 批量评估（并发执行）
    new_results = []
    success_count = 0
    error_count = 0
    progress_lock = threading.Lock()
    completed = 0
    
    if len(script_ids) == 0:
        print(f"\n✅ 所有脚本都已评估完成，无需新评估！")
    else:
        print(f"\n🚀 使用 {MAX_WORKERS} 个并发线程进行评估...")
        print(f"{'='*70}")
        
        # 确保输出目录存在
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        # 使用线程池并发执行
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 提交所有任务
            future_to_script = {
                executor.submit(evaluate_single_script, script_id, config_loader, judger): script_id
                for script_id in script_ids
            }
            
            # 处理完成的任务
            for future in as_completed(future_to_script):
                script_id = future_to_script[future]
                
                try:
                    result = future.result()
                    
                    with progress_lock:
                        new_results.append(result)
                        completed += 1
                        
                        if result['status'] == 'success':
                            success_count += 1
                            print(f"✅ [{completed}/{len(script_ids)}] {script_id} - 成功")
                        else:
                            error_count += 1
                            print(f"❌ [{completed}/{len(script_ids)}] {script_id} - 失败: {result.get('error', 'Unknown')}")
                        
                        # 每完成20个任务就保存一次进度（防止中途失败丢失所有数据）
                        if completed % 20 == 0:
                            temp_all_results = existing_results + new_results
                            temp_all_results.sort(key=lambda x: x['script_id'])
                            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                                json.dump(temp_all_results, f, ensure_ascii=False, indent=2)
                            print(f"💾 [进度保存] 已完成 {completed}/{len(script_ids)}，结果已保存")
                
                except Exception as e:
                    with progress_lock:
                        completed += 1
                        error_count += 1
                        print(f"💥 [{completed}/{len(script_ids)}] {script_id} - 异常: {e}")
    
    # 🔧 合并结果：现有结果 + 新结果
    all_results = existing_results + new_results
    
    # 按 script_id 排序（确保顺序一致）
    all_results.sort(key=lambda x: x['script_id'])
    
    # 最终保存结果
    print(f"\n{'='*70}")
    print(f"💾 最终保存结果...")
    print(f"{'='*70}")
    
    # 保存JSON（合并后的完整结果）
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 结果已保存: {OUTPUT_FILE}")
    print(f"   总计: {len(all_results)} 个剧本（{len(existing_results)} 个现有 + {len(new_results)} 个新增）")
    
    # 打印统计
    print(f"\n{'='*70}")
    print(f"📊 本次评估统计")
    print(f"{'='*70}")
    print(f"本次评估: {len(script_ids)} 个剧本")
    print(f"成功: {success_count} 个")
    print(f"失败: {error_count} 个")
    if len(script_ids) > 0:
        print(f"成功率: {success_count/len(script_ids)*100:.1f}%")
    
    print(f"\n{'='*70}")
    print(f"📊 累计统计")
    print(f"{'='*70}")
    total_success = sum(1 for r in all_results if r['status'] == 'success')
    total_error = sum(1 for r in all_results if r['status'] == 'error')
    print(f"总计: {len(all_results)} 个剧本")
    print(f"成功: {total_success} 个")
    print(f"失败: {total_error} 个")
    print(f"成功率: {total_success/len(all_results)*100:.1f}%")
    
    print(f"\n{'='*70}")
    print(f"🎉 批量评估完成！")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

