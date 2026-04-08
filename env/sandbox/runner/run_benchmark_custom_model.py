#!/usr/bin/env python3
"""
使用自定义模型API运行Benchmark评测

支持SGLang等自部署模型的评测
"""

import json
import sys
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# 添加项目根路径，确保可以正确导入 Benchmark 模块
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Benchmark.topics.config_loader import load_config
from Benchmark.agents.actor import Actor
from Benchmark.agents.director import Director
from Benchmark.agents.judger import Judger
from Benchmark.orchestrator.chat_loop_epj import run_chat_loop_epj
from Benchmark.prompts.test_model_prompts import generate_test_model_prompts

# ═══════════════════════════════════════════════════════════════
# 自定义模型API配置
# ═══════════════════════════════════════════════════════════════

# 🔧 在这里配置你的自定义模型API
CUSTOM_API_CONFIG = {
    "api_key": "EMPTY",
    "base_url": "http://115.190.74.46:58101/v1",
    "model_name": "default",
    "max_tokens": 8192,
    "temperature": 0.7,
    "top_p": 0.8,
    "presence_penalty": 1.5,
    "extra_body": {
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": False},
    }
}

# 评测配置
# 使用新数据集的 30 个三维分层抽样案例 ID 列表
BENCHMARK_CASES_FILE = (
    PROJECT_ROOT / "runner/benchmark_cases/sampled_benchmark_30.txt"
)
OUTPUT_DIR = PROJECT_ROOT / "results/benchmark_runs"
MAX_TURNS = 45  # 每个对话最多轮次
K = 1  # 每K轮评估一次EPJ

# 并发配置
MAX_WORKERS = 1  # 并发线程数（设为1按顺序运行，避免API限流）

# 测试模式配置
TEST_MODE = False  # 🚀 运行全部30个案例
TEST_LIMIT = 1    # 测试模式下运行的案例数量（当前未使用）

# Judger、Actor、Director使用的模型（用于评估、角色扮演和剧情控制）
JUDGER_MODEL = "google/gemini-2.5-pro"  # 评估模型
ACTOR_MODEL = "google/gemini-2.5-pro"  # 角色扮演模型
DIRECTOR_MODEL = "google/gemini-2.5-pro"  # 剧情控制模型


# ═══════════════════════════════════════════════════════════════
# 自定义测试模型类（包装用户的API）
# ═══════════════════════════════════════════════════════════════

class CustomTestModel:
    """
    自定义测试模型类
    
    包装用户自己部署的模型API（如SGLang），使其符合TestModel接口
    """
    
    def __init__(self, config: Dict):
        """
        初始化自定义测试模型
        
        Args:
            config: API配置字典，包含api_key, base_url, model_name等
        """
        self.config = config
        self.client = OpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"],
        )
        self.model_name = config["model_name"]
        print(f"✅ 自定义测试模型API已连接: {config['base_url']}")
        print(f"   模型: {config['model_name']}")
    
    def generate_reply(self, history: List[Dict]) -> str:
        """
        根据对话历史生成回复（符合TestModel接口）
        
        Args:
            history: 对话历史列表，每个元素包含role和content
        
        Returns:
            str: 模型生成的回复
        """
        import time
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # 🔧 使用test_model_prompts.py生成规范的prompt
                system_prompt, user_prompt = generate_test_model_prompts(history)
                
                # 构建OpenAI API格式的消息
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                
                # 准备API调用参数
                api_params = {
                    "model": self.config["model_name"],
                    "messages": messages,
                    "max_tokens": self.config.get("max_tokens", 8192),
                    "temperature": self.config.get("temperature", 0.7),
                    "top_p": self.config.get("top_p", 0.8),
                    "presence_penalty": self.config.get("presence_penalty", 1.5),
                }
                
                # 添加extra_body（如果有）
                if "extra_body" in self.config:
                    api_params["extra_body"] = self.config["extra_body"]
                
                # 调用API
                response = self.client.chat.completions.create(**api_params)
                
                # 提取响应内容
                reply_content = response.choices[0].message.content
                
                if not reply_content or reply_content.strip() == "":
                    print("⚠️ 警告：API返回了空内容")
                    return "（错误：API返回空响应）"
                
                return reply_content
                
            except Exception as e:
                error_msg = str(e)
                print(f"❌ API调用失败 (尝试 {attempt + 1}/{max_retries}): {error_msg}")
                
                # 如果是最后一次尝试，打印完整堆栈并返回错误
                if attempt == max_retries - 1:
                    import traceback
                    traceback.print_exc()
                    return f"（错误：API调用失败 - {error_msg}）"
                
                # 等待后重试（指数退避：1秒, 2秒, 4秒）
                wait_time = 2 ** attempt
                print(f"⏳ 等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)


# ═══════════════════════════════════════════════════════════════
# Benchmark运行逻辑
# ═══════════════════════════════════════════════════════════════

def run_single_case(script_id: str) -> Dict:
    """
    运行单个测试案例
    
    Args:
        script_id: 剧本ID（如"script_001"）
    
    Returns:
        Dict: 测试结果
    """
    print(f"\n{'='*70}")
    print(f"🎬 开始测试: {script_id}")
    print(f"{'='*70}")
    
    try:
        # 1. 提取数字ID
        numeric_id = script_id.replace("script_", "")
        
        # 2. 加载剧本配置
        print(f"\n📚 加载剧本配置...")
        config = load_config(numeric_id)
        actor_prompt = config['actor_prompt']
        scenario = config['scenario']
        print(f"✅ 剧本加载成功")
        
        # 3. 初始化所有agents
        print(f"\n🤖 初始化Agents...")
        
        # 初始化Actor
        actor = Actor(model_name=ACTOR_MODEL)
        actor.set_system_prompt(actor_prompt)
        
        # 初始化Director
        director = Director(
            scenario=scenario,
            actor_prompt=actor_prompt,
            model_name=DIRECTOR_MODEL,
            use_function_calling=True
        )
        
        # 初始化Judger
        judger = Judger(model_name=JUDGER_MODEL)
        
        # 初始化自定义TestModel
        custom_test_model = CustomTestModel(CUSTOM_API_CONFIG)
        
        print(f"✅ 所有Agents初始化成功")
        
        # 4. 运行EPJ对话循环
        print(f"\n🎭 开始EPJ对话循环...")
        result = run_chat_loop_epj(
            actor=actor,
            director=director,
            judger=judger,
            test_model=custom_test_model,
            script_id=numeric_id,
            max_turns=MAX_TURNS,
            K=K,
            test_model_name=f"Custom-{CUSTOM_API_CONFIG['model_name']}"
        )
        
        # 5. 检测API调用失败情况
        consecutive_failures = 0
        max_consecutive_failures = 5  # 连续失败5次则终止
        
        if 'epj' in result and 'trajectory' in result['epj']:
            for turn_data in result['epj']['trajectory']:
                test_model_msg = turn_data.get('test_model_message', '')
                if test_model_msg.startswith('（错误：API调用失败'):
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        print(f"\n⚠️ 检测到连续 {consecutive_failures} 次API调用失败")
                        result['status'] = 'api_failure'
                        result['failure_reason'] = f'TestModel API连续失败{consecutive_failures}次'
                        result['script_id'] = script_id
                        result['timestamp'] = datetime.now().isoformat()
                        return result
                else:
                    consecutive_failures = 0  # 重置计数器
        
        # 6. 添加script_id到结果中
        result['script_id'] = script_id
        result['status'] = 'success'
        result['timestamp'] = datetime.now().isoformat()
        
        # 📊 显示详细结果
        print(f"\n{'='*70}")
        print(f"📊 {script_id} 测试结果")
        print(f"{'='*70}")
        
        print(f"\n✅ 基本信息:")
        print(f"   总轮数: {result.get('total_turns', 'N/A')}")
        print(f"   终止原因: {result.get('termination_reason', 'N/A')[:100]}")
        print(f"   终止类型: {result.get('termination_type', 'N/A')}")
        
        # EPJ/EPM数据
        if 'epj' in result:
            epj_data = result['epj']
            print(f"\n📈 EPJ数据:")
            print(f"   初始赤字 P_0: {epj_data.get('P_0_initial_deficit', 'N/A')}")
            print(f"   最终位置 P_final: {epj_data.get('P_final_position', 'N/A')}")
            print(f"   总评估次数: {epj_data.get('total_evaluations', 'N/A')}")
            print(f"   评估周期K: {epj_data.get('K', 'N/A')}")
            
            # EPM v2.0 指标
            if 'epm_victory_analysis' in result:
                epm_data = result['epm_victory_analysis']
                print(f"\n🎯 EPM v2.0 指标:")
                print(f"   胜利类型: {epm_data.get('primary_victory_type', 'N/A')}")
                
                conditions = epm_data.get('conditions', {})
                if 'geometric' in conditions:
                    geo = conditions['geometric']
                    status = "✅" if geo.get('achieved') else "❌"
                    print(f"   {status} 几何: {geo.get('value', 'N/A')} / {geo.get('threshold', 'N/A')}")
                
                if 'positional' in conditions:
                    pos = conditions['positional']
                    status = "✅" if pos.get('achieved') else "❌"
                    print(f"   {status} 位置: {pos.get('value', 'N/A')} / {pos.get('threshold', 'N/A')}")
                
                if 'energetic' in conditions:
                    eng = conditions['energetic']
                    status = "✅" if eng.get('achieved') else "❌"
                    print(f"   {status} 能量: {eng.get('value', 'N/A')} / {eng.get('threshold', 'N/A')}")
        
        # 对话历史摘要
        if 'history' in result:
            history = result['history']
            print(f"\n💬 对话历史摘要 (共{len(history)}轮):")
            # 显示前3轮和最后2轮
            show_count = min(3, len(history))
            for i, msg in enumerate(history[:show_count], 1):
                role = "角色" if msg['role'] == 'actor' else "AI"
                content_preview = msg['content'][:50] + "..." if len(msg['content']) > 50 else msg['content']
                print(f"   {i}. {role}: {content_preview}")
            
            if len(history) > 5:
                print(f"   ... (省略{len(history)-5}轮) ...")
                for i, msg in enumerate(history[-2:], len(history)-1):
                    role = "角色" if msg['role'] == 'actor' else "AI"
                    content_preview = msg['content'][:50] + "..." if len(msg['content']) > 50 else msg['content']
                    print(f"   {i}. {role}: {content_preview}")
        
        print(f"\n✅ {script_id} 测试完成")
        print(f"{'='*70}")
        
        return result
        
    except Exception as e:
        print(f"\n❌ {script_id} 测试失败: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'script_id': script_id,
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


def main():
    """主函数：运行benchmark评测"""
    
    print("="*70)
    print("🚀 EPJ Benchmark - 自定义模型评测系统")
    print("="*70)
    
    # 1. 显示配置信息
    print(f"\n📋 评测配置:")
    print(f"   自定义模型API: {CUSTOM_API_CONFIG['base_url']}")
    print(f"   模型名称: {CUSTOM_API_CONFIG['model_name']}")
    print(f"   最大轮次: {MAX_TURNS}")
    print(f"   评估周期K: {K}")
    print(f"   并发线程数: {MAX_WORKERS} ⚡")
    print(f"   Actor模型: {ACTOR_MODEL}")
    print(f"   Director模型: {DIRECTOR_MODEL}")
    print(f"   Judger模型: {JUDGER_MODEL}")
    
    # 2. 读取benchmark案例列表
    print(f"\n📂 读取benchmark案例列表: {BENCHMARK_CASES_FILE}")
    cases_file = BENCHMARK_CASES_FILE
    
    if not cases_file.exists():
        print(f"❌ 错误：找不到案例列表文件 {BENCHMARK_CASES_FILE}")
        print(f"   请先运行 sample_benchmark_cases.py 生成抽样案例")
        return
    
    with open(cases_file, 'r', encoding='utf-8') as f:
        all_script_ids = [line.strip() for line in f if line.strip()]
    
    # 🧪 测试模式：只运行前几个案例
    if TEST_MODE:
        script_ids = all_script_ids[:TEST_LIMIT]
        print(f"🧪 **测试模式** - 只运行前 {TEST_LIMIT} 个案例（共{len(all_script_ids)}个）")
        print(f"   测试案例: {', '.join(script_ids)}")
        print(f"   💡 测试成功后，将 TEST_MODE 改为 False 运行全部案例")
    else:
        script_ids = all_script_ids
        print(f"✅ 共加载 {len(script_ids)} 个测试案例")
        print(f"   案例: {', '.join(script_ids[:5])}...")
    
    # 3. 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 提取模型名称（去掉provider前缀，替换特殊字符）
    model_name = CUSTOM_API_CONFIG['model_name']
    # 如果有斜杠，只取最后一部分（如 moonshotai/kimi-k2-0905 -> kimi-k2-0905）
    model_short_name = model_name.split('/')[-1].replace(':', '_').replace(' ', '_')
    
    # 生成文件夹名称：模型名_日期时间_序号
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_suffix = f"_test{TEST_LIMIT}" if TEST_MODE else ""
    run_output_dir = OUTPUT_DIR / f"{model_short_name}_{timestamp}{test_suffix}"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 结果将保存到: {run_output_dir}")
    
    # 4. 运行所有测试案例（并发执行）
    all_results = []
    success_count = 0
    error_count = 0
    completed = 0
    progress_lock = threading.Lock()
    
    print(f"\n🚀 使用 {MAX_WORKERS} 个并发线程进行评测...")
    print(f"{'='*70}\n")
    
    # 使用线程池并发执行
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        future_to_script = {
            executor.submit(run_single_case, script_id): script_id
            for script_id in script_ids
        }
        
        # 处理完成的任务
        for future in as_completed(future_to_script):
            script_id = future_to_script[future]
            
            try:
                result = future.result()
                
                with progress_lock:
                    all_results.append(result)
                    completed += 1
                    
                    if result['status'] == 'success':
                        success_count += 1
                        # 显示简要结果
                        turns = result.get('total_turns', '?')
                        term_type = result.get('termination_type', 'UNKNOWN')
                        
                        # 尝试获取EPM指标
                        epm_info = ""
                        if 'epm_victory_analysis' in result:
                            epm = result['epm_victory_analysis']
                            victory = epm.get('primary_victory_type', '?')
                            epm_info = f" | EPM: {victory}"
                        
                        print(f"✅ [{completed}/{len(script_ids)}] {script_id} - 成功 ({turns}轮, {term_type}{epm_info})")
                    else:
                        error_count += 1
                        print(f"❌ [{completed}/{len(script_ids)}] {script_id} - 失败: {result.get('error', 'Unknown')[:100]}")
                    
                    # 保存单个案例结果
                    case_output_file = run_output_dir / f"{script_id}_result.json"
                    with open(case_output_file, 'w', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    
                    # 每完成5个任务就保存一次进度（防止中途失败丢失数据）
                    if completed % 5 == 0:
                        temp_summary = {
                            'run_timestamp': timestamp,
                            'custom_model_config': {
                                'base_url': CUSTOM_API_CONFIG['base_url'],
                                'model_name': CUSTOM_API_CONFIG['model_name']
                            },
                            'evaluation_config': {
                                'max_turns': MAX_TURNS,
                                'K': K,
                                'max_workers': MAX_WORKERS,
                                'actor_model': ACTOR_MODEL,
                                'director_model': DIRECTOR_MODEL,
                                'judger_model': JUDGER_MODEL
                            },
                            'progress': f"{completed}/{len(script_ids)}",
                            'results': all_results
                        }
                        progress_file = run_output_dir / "progress.json"
                        with open(progress_file, 'w', encoding='utf-8') as f:
                            json.dump(temp_summary, f, ensure_ascii=False, indent=2)
                        print(f"💾 [进度保存] 已完成 {completed}/{len(script_ids)}")
            
            except Exception as e:
                with progress_lock:
                    completed += 1
                    error_count += 1
                    print(f"💥 [{completed}/{len(script_ids)}] {script_id} - 异常: {e}")
    
    # 5. 保存汇总结果
    summary_file = run_output_dir / "summary.json"
    summary = {
        'run_timestamp': timestamp,
        'test_mode': TEST_MODE,
        'test_limit': TEST_LIMIT if TEST_MODE else None,
        'custom_model_config': {
            'base_url': CUSTOM_API_CONFIG['base_url'],
            'model_name': CUSTOM_API_CONFIG['model_name']
        },
        'evaluation_config': {
            'max_turns': MAX_TURNS,
            'K': K,
            'max_workers': MAX_WORKERS,
            'actor_model': ACTOR_MODEL,
            'director_model': DIRECTOR_MODEL,
            'judger_model': JUDGER_MODEL
        },
        'total_cases': len(script_ids),
        'success_count': success_count,
        'error_count': error_count,
        'success_rate': success_count / len(script_ids) * 100 if script_ids else 0,
        'results': all_results
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # 6. 打印最终统计
    print("\n" + "="*70)
    if TEST_MODE:
        print("📊 测试运行完成 🧪")
    else:
        print("📊 Benchmark评测完成")
    print("="*70)
    print(f"总案例数: {len(script_ids)}")
    print(f"成功: {success_count}")
    print(f"失败: {error_count}")
    print(f"成功率: {success_count/len(script_ids)*100:.1f}%")
    print(f"\n结果已保存到: {run_output_dir}")
    print(f"  - 汇总文件: summary.json")
    print(f"  - 单个案例: [script_id]_result.json")
    
    if TEST_MODE:
        print(f"\n💡 测试通过！要运行全部{len(all_script_ids)}个案例，请:")
        print(f"   1. 将第55行 TEST_MODE 改为 False")
        print(f"   2. 重新运行脚本")
    
    print("="*70)


if __name__ == "__main__":
    main()

