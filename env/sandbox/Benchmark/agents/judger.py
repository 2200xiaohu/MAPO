# Benchmark/agents/judger.py

from Benchmark.prompts.judger_prompts import (
    generate_progress_prompt,
    generate_quality_prompt,
    generate_overall_prompt
)
from Benchmark.llms.api import get_llm_response
import re
import json

class Judger:
    """
    共情评估员（传感器），负责填写心理量表。

    EPJ系统中的职责：
    - T=0: 填写初始共情赤字量表 (IEDR)
    - T>0: 每K轮填写MDEP进展量表 (MDEP-PR)
    - 禁止计算分数和做最终决策（由Orchestrator和Director负责）
    """
    def __init__(self, model_name: str = "google/gemini-2.5-pro"):
        self.model_name = model_name

    def evaluate_empathy_progress(self, recent_turns: list, current_progress: int) -> int:
        """
        评估最近3轮的共情进度改善情况

        Args:
            recent_turns: 最近3轮对话记录
            current_progress: 当前累计进度分数

        Returns:
            int: 进度分数（可为负数）
        """
        prompt = generate_progress_prompt(recent_turns, current_progress)

        try:
            #print(f"--- [Judger] 正在评估共情进度（模型: {self.model_name}）... ---")

            response = get_llm_response(
                messages=[{"role": "user", "content": prompt}],
                model_name=self.model_name,
                max_tokens=20000,
                client_key="judger"
            )

            # 提取分数
            progress_score = self._extract_score(response)
            #print(f"--- [Judger] 进度评估完成，分数: {progress_score} ---")

            return progress_score

        except Exception as e:
            #print(f"!!! [Judger] 进度评估失败: {e} !!!")
            return 0  # 默认返回0分

    def evaluate_empathy_quality(self, recent_turns: list) -> int:
        """
        评估最近3轮的共情质量

        Args:
            recent_turns: 最近3轮对话记录

        Returns:
            int: 过程质量分数 (0-100)
        """
        prompt = generate_quality_prompt(recent_turns)

        try:
            #print(f"--- [Judger] 正在评估共情质量（模型: {self.model_name}）... ---")

            response = get_llm_response(
                messages=[{"role": "user", "content": prompt}],
                model_name=self.model_name,
                max_tokens=20000,
                client_key="judger"
            )

            # 提取分数
            quality_score = self._extract_score(response, min_score=0, max_score=100)
            #print(f"--- [Judger] 质量评估完成，分数: {quality_score} ---")

            return quality_score

        except Exception as e:
            #print(f"!!! [Judger] 质量评估失败: {e} !!!")
            return 50  # 默认返回中等分数

    def evaluate_overall_quality(self, full_history: list) -> int:
        """
        评估整体共情质量

        Args:
            full_history: 完整对话历史记录

        Returns:
            int: 总质量分数 (0-100)
        """
        prompt = generate_overall_prompt(full_history)

        try:
            #print(f"--- [Judger] 正在评估整体共情质量（模型: {self.model_name}）... ---")

            response = get_llm_response(
                messages=[{"role": "user", "content": prompt}],
                model_name=self.model_name,
                max_tokens=20000,
                client_key="judger"
            )

            # 提取分数
            overall_score = self._extract_score(response, min_score=0, max_score=100)
            #print(f"--- [Judger] 整体评估完成，分数: {overall_score} ---")

            return overall_score

        except Exception as e:
            #print(f"!!! [Judger] 整体评估失败: {e} !!!")
            return 50  # 默认返回中等分数

    def _extract_score(self, response: str, min_score=None, max_score=None) -> int:
        """
        从LLM响应中提取分数
        """
        try:
            # 查找数字
            numbers = re.findall(r'-?\d+', response)
            if numbers:
                score = int(numbers[-1])  # 取最后一个数字

                # 应用范围限制
                if min_score is not None:
                    score = max(score, min_score)
                if max_score is not None:
                    score = min(score, max_score)

                return score
            else:
                raise ValueError("未找到有效分数")

        except Exception as e:
            #print(f"⚠️ [Judger] 分数提取失败: {e}, 响应: {response}")
            return 0 if min_score is None else min_score

    # ═══════════════════════════════════════════════════════════════
    # EPJ 系统 - 量表填写功能
    # ═══════════════════════════════════════════════════════════════

    def fill_iedr(self, script_content: dict) -> dict:
        """
        填写初始共情赤字量表 (IEDR) - T=0时调用

        这是EPJ系统的第一步：量化剧本的初始共情需求

        Args:
            script_content: 剧本内容（包含actor_prompt和scenario）

        Returns:
            dict: 填写完成的IEDR量表
            {
                "C.1": 0-3,
                "C.2": 0-3,
                "A.1": 0-3,
                "A.2": 0-3,
                "A.3": 0-3,
                "P.1": 0-3,
                "P.2": 0-3,
                "P.3": 0-3,
                "reasoning": "判断依据"
            }
        """
        from Benchmark.epj.judger_prompts import generate_iedr_prompt

        prompt = generate_iedr_prompt(script_content)

        max_retries = 5
        last_error = None

        for attempt in range(max_retries):
            try:
                #print(f"═══ [Judger/传感器] T=0: 填写 IEDR 量表 ═══")

                # 🆕 IEDR 使用 OpenRouter + Gemini 2.5 Pro（始终走外网 API）
                response = get_llm_response(
                    messages=[{"role": "user", "content": prompt}],
                    model_name="google/gemini-2.5-pro",  # 🆕 显式指定 Gemini 模型
                    json_mode=True,
                    max_tokens=20000,  # IEDR：确保有足够空间输出完整JSON（包含详细evidence和reasoning）
                    client_key="iedr_evaluator"  # 🆕 使用专用客户端（始终 OpenRouter）
                )

                if not response or "（错误：API返回空响应" in response:
                    last_error = "API返回空响应"
                    continue  # 进入下一次重试

                # 解析JSON响应
                raw_response = self._parse_rubric_response(response)

                if 'error' in raw_response:
                    last_error = raw_response['reason']
                    continue  # 进入下一次重试

                # 转换格式：提取level字段
                filled_iedr = self._convert_iedr_format(raw_response)

                #print(f"✅ [Judger] IEDR 量表填写完成")
                #print(f"   C.1={filled_iedr.get('C.1')}, C.2={filled_iedr.get('C.2')}, C.3={filled_iedr.get('C.3')}")
                #print(f"   A.1={filled_iedr.get('A.1')}, A.2={filled_iedr.get('A.2')}, A.3={filled_iedr.get('A.3')}")
                #print(f"   P.1={filled_iedr.get('P.1')}, P.2={filled_iedr.get('P.2')}, P.3={filled_iedr.get('P.3')}")

                return filled_iedr

            except Exception as e:
                #print(f"!!! [Judger] IEDR 填写失败: {e} !!!")
                # 返回默认值（中等难度）
                last_error = str(e)
                #print(f"⚠️ [Judger] 尝试 {attempt + 1}/{max_retries} 失败: {e}")
                if attempt == max_retries - 1:  # 最后一次尝试
                    break
        print(f"!!! [Judger] IEDR 填写失败（已重试{max_retries}次）: {last_error} !!!")
        return {
            "C.1": 1, "C.2": 1, "C.3": 1,
            "A.1": 1, "A.2": 1, "A.3": 1,
            "P.1": 1, "P.2": 1, "P.3": 1,
            "reasoning": "填写失败，使用默认值。原因：{last_error}"
        }

    def fill_mdep_pr(self, recent_turns: list, script_context: dict = None, actor_prompt: str = None, full_history: list = None) -> dict:
        """
        填写MDEP进展量表 (MDEP-PR) - T>0时每K轮调用

        这是EPJ系统的核心评估步骤：量化每K轮的共情进展

        Args:
            recent_turns: 最近K轮的对话记录（评估范围）
            script_context: 剧本上下文（可选）
            full_history: 完整对话历史（供参考上下文，可选）

        Returns:
            dict: 填写完成的MDEP-PR量表
            {
                "C.Prog": 0-2,
                "C.Neg": 0 or -1 or -2,
                "A.Prog": 0-2,
                "A.Neg": 0 or -1 or -2,
                "P.Prog": 0-2,
                "P.Neg": 0 or -1 or -2,
                "reasoning": "判断依据"
            }
        """
        from Benchmark.epj.judger_prompts import generate_mdep_pr_prompt

        # 🆕 传递完整历史供Judger参考上下文
        prompt = generate_mdep_pr_prompt(recent_turns, script_context, full_history, model_name=self.model_name)

        # 🔧 添加重试机制（最多5次）
        max_retries = 5
        last_error = None
        last_raw_response = None      # 🆕 保存最后一次成功解析的 raw_response
        last_validation_result = None  # 🆕 保存最后一次验证结果

        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    import time
                    import random
                    wait_time = 3  # 固定3秒等待
                    print(f"🔄 [Judger] 等待 {wait_time:.1f}秒后重试第 {attempt} 次...")
                    time.sleep(wait_time)

                # print(f"═══ [Judger/传感器] T>0: 填写 MDEP-PR 量表 ═══")
                # print(f"   评估轮次: 最近 {len(recent_turns)} 轮")

                #print(f"prompt: {prompt}")
                response = get_llm_response(
                    messages=[{"role": "user", "content": prompt}],
                    model_name=self.model_name,
                    json_mode=True,
                    max_tokens=20000,  # 🔧 问题3修复：增加到6000，确保完整JSON不被截断
                    client_key="judger"
                )

                # 🔧 检测空响应或错误标记
                if not response or "（错误：API返回空响应" in response:
                    last_error = "API返回空响应"
                    #print(f"⚠️ [Judger] 检测到空响应，准备重试...")
                    continue  # 进入下一次重试

                # 解析JSON响应
                raw_response = self._parse_rubric_response(response)

                if 'error' in raw_response:
                    last_error = raw_response['reason']
                    continue  # 进入下一次重试

                # 🆕 保存最后一次成功解析的 raw_response（即使后续验证失败）
                last_raw_response = raw_response

                # 🆕 新增：严格字段验证
                validation_result = self._validate_mdep_pr_response(raw_response)
                last_validation_result = validation_result  # 🆕 保存验证结果

                if not validation_result["valid"]:
                    last_error = f"格式验证失败: {validation_result['error']}"
                    print(f"⚠️ [Judger] {last_error}")
                    continue  # 进入下一次重试

                # 🔧 问题3修复：转换新格式并显示reasoning
                filled_mdep_pr = self._convert_mdep_format(raw_response)

                #print(f"✅ [Judger] MDEP-PR 量表填写完成")
                #print(f"   C: Prog={filled_mdep_pr.get('C.Prog')}, Neg={filled_mdep_pr.get('C.Neg')}")
                #print(f"   A: Prog={filled_mdep_pr.get('A.Prog')}, Neg={filled_mdep_pr.get('A.Neg')}")
                #print(f"   P: Prog={filled_mdep_pr.get('P.Prog')}, Neg={filled_mdep_pr.get('P.Neg')}")

                # 🔧 问题3修复：显示reasoning（如果有）
                self._print_mdep_reasoning(raw_response)

                # 💾 附加详细分析信息（供保存使用）
                filled_mdep_pr['detailed_analysis'] = raw_response

                return filled_mdep_pr

            except Exception as e:
                last_error = str(e)
                #print(f"⚠️ [Judger] 尝试 {attempt + 1}/{max_retries} 失败: {e}")
                if attempt == max_retries - 1:  # 最后一次尝试
                    break

        # 🆕 所有重试都失败后，尝试构建部分成功的返回值
        print(f"⚠️ [Judger] MDEP-PR 填写失败（已重试{max_retries}次）: {last_error}")

        # 如果有最后一次解析成功的响应，尝试提取有效字段
        if last_raw_response is not None:
            return self._build_partial_mdep_result(last_raw_response, last_validation_result, last_error)

        # 完全没有解析成功，返回全默认值
        print(f"⚠️ [Judger] 完全失败。使用错误的默认值")
        return {
            "C.Prog": 0, "C.Neg": 0,
            "A.Prog": 0, "A.Neg": 0,
            "P.Prog": 1, "P.Neg": -1,  # 🆕 P轴使用新默认值
            "reasoning": f"填写失败（重试{max_retries}次后），使用默认值。 {last_error}"
        }

    def _parse_rubric_response(self, response: str) -> dict:
        """
        解析量表填写响应（提取JSON）并清洗List值
        """
        if not response or not isinstance(response, str):
            raise ValueError(f"响应类型错误: {type(response)}")
        parsed = None
        try:
            # 1. 尝试直接解析
            parsed = json.loads(response.strip())
        except json.JSONDecodeError:
            # 2. 尝试提取代码块中的JSON
            json_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
            json_match = re.search(json_pattern, response, re.DOTALL)

            if json_match:
                json_str = json_match.group(1).strip()
                try:
                    parsed = json.loads(json_str)
                except json.JSONDecodeError as e2:
                    raise ValueError(f"无法解析提取的JSON: {e2}")
            else:
                raise ValueError(f"无法从响应中提取JSON")

        if isinstance(parsed, dict):
            for key in parsed: # 直接遍历 key
                val = parsed[key]

                # 处理 List
                if isinstance(val, list):
                    val = val[0] if len(val) > 0 else None
                    parsed[key] = val # 更新字典

                # 处理后再次检查现在的 val 是否为 Dict
                if isinstance(val, dict):
                    print(f"⚠️ [Judger] 检测到非法嵌套字典 (Key: {key})，请求重试...")
                    return {'error': True, 'reason': f'nested_dict_detected_in_{key}'}

        #print(f"提取的JSON (处理后): {parsed}")
        return parsed

    # ═══════════════════════════════════════════════════════════════
    # 🆕 新增：MDEP-PR 响应格式严格验证
    # ═══════════════════════════════════════════════════════════════

    def _validate_mdep_pr_response(self, raw_response: dict) -> dict:
        """
        验证 MDEP-PR 响应格式是否符合要求

        期望格式：
        {
          "C_Prog_level": <0/1/2>,
          "C_Prog_evidence": "<string>",
          "C_Prog_reasoning": "<string>",
          "C_Neg_level": <0/-1/-2>,
          "C_Neg_evidence": "<string>",
          "C_Neg_reasoning": "<string>",
          "A_Prog_level": <0/1/2>,
          "A_Prog_evidence": "<string>",
          "A_Prog_reasoning": "<string>",
          "A_Neg_level": <0/-1/-2>,
          "A_Neg_evidence": "<string>",
          "A_Neg_reasoning": "<string>",
          "P_Prog_level": <0/1/2>,
          "P_Prog_evidence": "<string>",
          "P_Prog_reasoning": "<string>",
          "P_Neg_level": <0/-1/-2>,
          "P_Neg_evidence": "<string>",
          "P_Neg_reasoning": "<string>"
        }

        Args:
            raw_response: 解析后的JSON响应

        Returns:
            dict: 验证结果
                - 成功: {"valid": True}
                - 失败: {"valid": False, "error": "错误描述", "missing_fields": [...], "invalid_fields": [...]}
        """
        # 定义所有必需字段及其验证规则
        FIELD_SPECS = {
            # Prog 字段: level 范围 0-2
            "C_Prog_level": {"type": "int", "valid_values": [0, 1, 2]},
            "C_Prog_evidence": {"type": "str", "allow_empty_for_zero": True, "level_field": "C_Prog_level"},
            "C_Prog_reasoning": {"type": "str", "min_length": 1},

            "A_Prog_level": {"type": "int", "valid_values": [0, 1, 2]},
            "A_Prog_evidence": {"type": "str", "allow_empty_for_zero": True, "level_field": "A_Prog_level"},
            "A_Prog_reasoning": {"type": "str", "min_length": 1},

            "P_Prog_level": {"type": "int", "valid_values": [0, 1, 2]},
            #"P_Prog_evidence": {"type": "str", "allow_empty_for_zero": True, "level_field": "P_Prog_level"},
            # "P_Prog_reasoning": {"type": "str", "min_length": 1},

            # Neg 字段: level 范围 0/-1/-2
            "C_Neg_level": {"type": "int", "valid_values": [0, -1, -2]},
            "C_Neg_evidence": {"type": "str", "allow_empty_for_zero": True, "level_field": "C_Neg_level"},
            "C_Neg_reasoning": {"type": "str", "min_length": 1},

            "A_Neg_level": {"type": "int", "valid_values": [0, -1, -2]},
            "A_Neg_evidence": {"type": "str", "allow_empty_for_zero": True, "level_field": "A_Neg_level"},
            "A_Neg_reasoning": {"type": "str", "min_length": 1},

            "P_Neg_level": {"type": "int", "valid_values": [0, -1, -2]},
            # "P_Neg_evidence": {"type": "str", "allow_empty_for_zero": True, "level_field": "P_Neg_level"},
            # "P_Neg_reasoning": {"type": "str", "min_length": 1},
        }

        missing_fields = []
        invalid_fields = []
        error_details = []

        for field_name, spec in FIELD_SPECS.items():
            # 检查字段是否存在
            if field_name not in raw_response:
                missing_fields.append(field_name)
                continue

            value = raw_response[field_name]

            # 类型检查
            if spec["type"] == "int":
                # 允许字符串形式的数字
                if isinstance(value, str):
                    try:
                        value = int(value)
                    except ValueError:
                        invalid_fields.append(field_name)
                        error_details.append(f"{field_name}: 无法转换为整数 (值: {repr(value)})")
                        continue

                if not isinstance(value, int):
                    invalid_fields.append(field_name)
                    error_details.append(f"{field_name}: 期望整数，得到 {type(value).__name__} (值: {repr(value)})")
                    continue

                # 值范围检查
                if "valid_values" in spec and value not in spec["valid_values"]:
                    invalid_fields.append(field_name)
                    error_details.append(f"{field_name}: 值 {value} 不在有效范围 {spec['valid_values']} 内")
                    continue

            elif spec["type"] == "str":
                if not isinstance(value, str):
                    invalid_fields.append(field_name)
                    error_details.append(f"{field_name}: 期望字符串，得到 {type(value).__name__}")
                    continue

                # evidence 字段：0级别时可以为 "0" 或空
                if spec.get("allow_empty_for_zero") and spec.get("level_field"):
                    level_field = spec["level_field"]
                    level_value = raw_response.get(level_field, None)

                    # 如果 level 是 0，evidence 可以是 "0" 或较短的内容
                    if level_value == 0 or level_value == "0":
                        # 0级别时 evidence 可以是 "0" 或空字符串
                        pass
                    else:
                        # 非0级别时，evidence 必须有实质内容
                        if len(value.strip()) < 2:
                            invalid_fields.append(field_name)
                            error_details.append(f"{field_name}: 非零级别时 evidence 必须有实质内容 (当前: {repr(value)})")
                            continue

                # reasoning 字段：必须非空
                elif spec.get("min_length"):
                    if len(value.strip()) < spec["min_length"]:
                        invalid_fields.append(field_name)
                        error_details.append(f"{field_name}: reasoning 不能为空")
                        continue

        # 汇总结果
        if missing_fields or invalid_fields:
            error_msg_parts = []
            if missing_fields:
                error_msg_parts.append(f"缺失字段: {missing_fields}")
            if invalid_fields:
                error_msg_parts.append(f"无效字段: {invalid_fields}")
            if error_details:
                error_msg_parts.append(f"详情: {'; '.join(error_details[:3])}")  # 只显示前3个错误详情

            return {
                "valid": False,
                "error": " | ".join(error_msg_parts),
                "missing_fields": missing_fields,
                "invalid_fields": invalid_fields,
                "error_details": error_details
            }

        return {"valid": True}

    # ═══════════════════════════════════════════════════════════════
    # 🆕 新增：从部分有效响应构建 MDEP-PR 结果
    # ═══════════════════════════════════════════════════════════════

    def _build_partial_mdep_result(self, raw_response: dict, validation_result: dict, error_msg: str) -> dict:
        """
        从部分有效的响应中构建 MDEP-PR 结果

        对于有效字段：使用解析值
        对于无效/缺失字段：使用默认值
            - P_Prog_level: 默认 1
            - P_Neg_level: 默认 -1
            - 其他: 默认 0

        Args:
            raw_response: 最后一次成功解析的 JSON 响应
            validation_result: 验证结果（包含 missing_fields 和 invalid_fields）
            error_msg: 错误信息

        Returns:
            dict: 构建的 MDEP-PR 结果
        """
        # 字段默认值映射
        DEFAULT_VALUES = {
            "C_Prog_level": 0,
            "C_Neg_level": 0,
            "A_Prog_level": 0,
            "A_Neg_level": 0,
            "P_Prog_level": 1,   # 🆕 特殊默认值
            "P_Neg_level": -1,   # 🆕 特殊默认值
        }

        # 获取无效和缺失的字段列表
        invalid_fields = set()
        if validation_result and not validation_result.get("valid", True):
            invalid_fields.update(validation_result.get("missing_fields", []))
            invalid_fields.update(validation_result.get("invalid_fields", []))

        # 构建结果
        result = {}
        field_mapping = {
            "C_Prog_level": "C.Prog",
            "C_Neg_level": "C.Neg",
            "A_Prog_level": "A.Prog",
            "A_Neg_level": "A.Neg",
            "P_Prog_level": "P.Prog",
            "P_Neg_level": "P.Neg",
        }

        used_defaults = []

        for raw_key, result_key in field_mapping.items():
            if raw_key in invalid_fields or raw_key not in raw_response:
                # 使用默认值
                result[result_key] = DEFAULT_VALUES[raw_key]
                used_defaults.append(f"{result_key}={DEFAULT_VALUES[raw_key]}")
            else:
                # 使用解析值
                value = raw_response[raw_key]
                # 确保类型正确
                if isinstance(value, str):
                    try:
                        value = int(value)
                    except ValueError:
                        value = DEFAULT_VALUES[raw_key]
                        used_defaults.append(f"{result_key}={value}")
                result[result_key] = value

        # 保存原始响应供分析
        result["_raw_response"] = raw_response
        result["detailed_analysis"] = raw_response

        # 记录部分成功信息
        if used_defaults:
            result["reasoning"] = f"部分字段使用默认值: {', '.join(used_defaults)}。原因: {error_msg}"
            print(f"⚠️ [Judger] 部分成功。使用默认值: {', '.join(used_defaults)}。\n最后返回结果: {result}")
        else:
            result["reasoning"] = "所有字段解析成功"

        return result

    def _convert_iedr_format(self, raw_response: dict) -> dict:
        """
        将IEDR的详细响应格式转换为简化格式

        输入格式（详细）:
        {
          "C.1_level": 2,
          "C.1_evidence": "...",
          "C.1_reasoning": "...",
          ...
        }

        输出格式（简化）:
        {
          "C.1": 2,
          "C.2": 1,
          ...
        }
        """
        indicators = [
            "C.1", "C.2", "C.3",
            "A.1", "A.2", "A.3",
            "P.1", "P.2", "P.3"
        ]

        simplified = {}
        for indicator in indicators:
            level_key = f"{indicator}_level"
            if level_key in raw_response:
                value = raw_response[level_key]
            else:
                # 如果没有_level后缀，尝试直接使用indicator键
                value = raw_response.get(indicator, 1)

            # 🆕 确保类型为整数（修复字符串类型导致的 KeyError）
            if isinstance(value, str):
                try:
                    value = int(value)
                except ValueError:
                    value = 1  # 默认中等难度
            elif not isinstance(value, int):
                value = 1  # 非整数类型，使用默认值

            simplified[indicator] = value
        return simplified

    def _convert_mdep_format(self, raw_response: dict) -> dict:
        """
        提取level字段用于向量计算

        说明：
        - Prompt要求输出完整格式（level + evidence + reasoning），这是为了科学性和可追溯性
        - scoring.py只需要level数字来计算向量 v_t
        - 原始的evidence和reasoning保存在 _raw_response 中，供后续分析使用

        新格式（Prompt输出）：
        {
          "C_Prog_level": 2,
          "C_Prog_evidence": "AI说...",
          "C_Prog_reasoning": "因为...",
          ...
        }

        计算格式（scoring.py输入）：
        {
          "C.Prog": 2,
          "C.Neg": 0,
          ...
        }

        Args:
            raw_response: Judger LLM的原始JSON响应

        Returns:
            dict: 提取level后的数据（用于scoring.calculate_increment_vector）
        """
        # 检查是否是新格式（带_level后缀）
        if 'C_Prog_level' in raw_response:
            # 新格式：提取level字段，转换为scoring期望的格式
            scoring_format = {
                "C.Prog": raw_response.get('C_Prog_level', 0),
                "C.Neg": raw_response.get('C_Neg_level', 0),
                "A.Prog": raw_response.get('A_Prog_level', 0),
                "A.Neg": raw_response.get('A_Neg_level', 0),
                "P.Prog": raw_response.get('P_Prog_level', 0),
                "P.Neg": raw_response.get('P_Neg_level', 0),
                "_raw_response": raw_response  # 保存完整响应（包含evidence和reasoning）
            }
            #print(f"提取的MDEP-PR: {scoring_format}")
            return scoring_format
        else:
            # 兼容旧格式（如果Prompt没有按预期输出）
            #print(f"提取的MDEP-PR: {raw_response}")
            return raw_response

    def _print_mdep_reasoning(self, raw_response: dict):
        """
        打印MDEP-PR的evidence和reasoning（问题3修复）

        Args:
            raw_response: 原始JSON响应
        """
        # 检查是否是新格式
        if 'C_Prog_level' not in raw_response:
            # 旧格式：只显示reasoning字段
            #if 'reasoning' in raw_response:
                #print(f"\n📝 [Judger推理]: {raw_response['reasoning']}")
            return

        # 新格式：显示详细的evidence和reasoning
        #print(f"\n📝 [Judger分析详情]:")

        dimensions = [
            ('C', '认知'),
            ('A', '情感'),
            ('P', '动机')
        ]

        for dim_code, dim_name in dimensions:
            #print(f"\n   【{dim_name}轴】:")

            # Prog
            prog_level = raw_response.get(f'{dim_code}_Prog_level', 0)
            prog_evidence = raw_response.get(f'{dim_code}_Prog_evidence', '')
            prog_reasoning = raw_response.get(f'{dim_code}_Prog_reasoning', '')

            # if prog_level != 0:
            #     #print(f"     进展[{prog_level:+d}]:")
            #     if prog_evidence:
            #         #print(f"       证据: {prog_evidence[:60]}...")
            #     if prog_reasoning:
            #         #print(f"       理由: {prog_reasoning}")
            # else:
            #     # 🔧 问题2修复：0级别也显示reasoning
            #     if prog_reasoning:
            #         #print(f"     进展[0]: {prog_reasoning}")

            # Neg
            neg_level = raw_response.get(f'{dim_code}_Neg_level', 0)
            neg_evidence = raw_response.get(f'{dim_code}_Neg_evidence', '')
            neg_reasoning = raw_response.get(f'{dim_code}_Neg_reasoning', '')

            # if neg_level != 0:
            #     #print(f"     倒退[{neg_level:+d}]:")
            #     if neg_evidence:
            #         #print(f"       证据: {neg_evidence[:60]}...")
            #     if neg_reasoning:
            #         #print(f"       理由: {neg_reasoning}")
            # else:
            #     # 🔧 问题2修复：0级别也显示reasoning
            #     if neg_reasoning:
            #         #print(f"     倒退[0]: {neg_reasoning}")
