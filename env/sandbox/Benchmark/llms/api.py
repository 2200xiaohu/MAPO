# Benchmark/llms/api.py (完整版)
import os
from dotenv import load_dotenv
from openai import OpenAI
import openai
import json
from datetime import datetime

import time
import re
import random

# API 重试配置
MAX_API_RETRIES = 5
MAX_API_WAIT_TIME = 60  # 最大等待时间 60 秒

# # 清除可能干扰的代理环境变量
# for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY']:
#     os.environ.pop(key, None)

# 加载环境变量
dotenv_path = "Benchmark/topics/.env"
load_dotenv(dotenv_path=dotenv_path)

# OpenRouter 配置
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
YOUR_APP_NAME = "Empathy-Benchmark-Demo"
YOUR_SITE_URL = "http://localhost:8000"
api_key = os.getenv("OPENROUTER_API_KEY", "")
# # 全局客户端变量
# client = None

# # 初始化 OpenRouter 客户端
# try:
#     # 尝试从环境变量获取
#     api_key = os.getenv("OPENROUTER_API_KEY")

#     # 如果环境变量没有，尝试从 config/api_config.py 获取
#     if not api_key:
#         try:
#             from config.api_config import OPENROUTER_API_KEY
#             api_key = OPENROUTER_API_KEY
#             print("--- [API层] 从 config/api_config.py 加载API key ---")
#         except ImportError:
#             pass

#     if not api_key:
#         raise ValueError("未找到 'OPENROUTER_API_KEY'")

#     # 只使用基础参数，避免 proxies 等不支持的参数
#     client = OpenAI(
#         base_url=OPENROUTER_BASE_URL,
#         api_key=api_key,
#         timeout=120.0,  # 🔧 增加超时时间到120秒（Director function calling可能需要更多时间）
#         max_retries=2,   # 🔧 自动重试2次
#         default_headers={
#             "HTTP-Referer": YOUR_SITE_URL,
#             "X-Title": YOUR_APP_NAME,
#         }
#     )
#     print("--- [API层] OpenRouter 客户端已成功配置 ---")

# except ValueError as e:
#     print(f"!!! [API层] 配置错误: {e} !!!")
#     print("请检查 Benchmark/topics/.env 文件中是否设置了 OPENROUTER_API_KEY")
# except Exception as e:
#     print(f"!!! [API层] 配置OpenRouter客户端时发生未知错误: {e} !!!")
#     print(f"错误类型: {type(e).__name__}")
#     print(f"错误详情: {str(e)}")


use_local = os.getenv("USE_LOCAL", "False").lower() == "true"
if use_local:
    print("--- [API层] 使用本地客户端 ---")
    OPENROUTER_BASE_URL = os.getenv("LOCAL_BASE_URL", "http://localhost:8000/v1")
    #OPENROUTER_BASE_URL = os.getenv("LOCAL_BASE_URL_ALT", "")
    api_key = os.getenv("LOCAL_API_KEY", "")
    MODEL_NAME = "qwen3_235b_a22b_instruct_2507"#"qwen3_235b_a22b_instruct_2507" #qwen3.5-122b-a10b

    

CLIENT_CONFIGS = {
    "director": {
        "base_url": os.getenv("DIRECTOR_BASE_URL", OPENROUTER_BASE_URL),
        "api_key": os.getenv("DIRECTOR_API_KEY", api_key),
    },
    "actor": {
        "base_url": os.getenv("ACTOR_BASE_URL", OPENROUTER_BASE_URL),
        "api_key": os.getenv("ACTOR_API_KEY", api_key),
    },
    "judger": {
        "base_url": os.getenv("JUDGER_BASE_URL", OPENROUTER_BASE_URL),
        "api_key": os.getenv("JUDGER_API_KEY", api_key),
    },
    # 🆕 新增：专用于 IEDR 评估，始终使用 OpenRouter + Gemini
    "iedr_evaluator": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": os.getenv("IEDR_API_KEY", ""),
    },
}

print(CLIENT_CONFIGS)

clients = {}

# 初始化所有客户端
for name, cfg in CLIENT_CONFIGS.items():
    if not cfg["api_key"]:
        raise ValueError(f"{name} 缺少 API key")
    clients[name] = OpenAI(
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        timeout=240.0,  # 4分钟，加快失败检测
        max_retries=0,  # 禁用自动重试，由应用层控制
        default_headers={
            "HTTP-Referer": YOUR_SITE_URL,
            "X-Title": YOUR_APP_NAME,
        },
    )
print(f"--- [API层] 客户端已配置: {list(clients.keys())} ---")

# 日志系统初始化
EXP_NAME = os.getenv("EXP_NAME")
if EXP_NAME is None:
    raise ValueError("EXP_NAME is not set")

LOG_FILE_PATHS = {
    "director": f"/nas/naifan/MultiTurnRL/log/{EXP_NAME}/llms/director_log.json",
    "actor": f"/nas/naifan/MultiTurnRL/log/{EXP_NAME}/llms/actor_log.json",
    "judger": f"/nas/naifan/MultiTurnRL/log/{EXP_NAME}/llms/judger_log.json",
    # 🆕 新增
    "iedr_evaluator": f"/nas/naifan/MultiTurnRL/log/{EXP_NAME}/llms/iedr_evaluator_log.json",
}

for client_key,file_path in LOG_FILE_PATHS.items():
    if os.path.exists(file_path):
        try:
            # 删除文件
            os.remove(file_path)
        except OSError as e:
            print(f"❌ 删除失败: {e}")


def _strip_thinking(text: str) -> str:
    """去除模型返回中的 <think>...</think> 内容"""
    if not text:
        return text
    return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()

def get_llm_response(messages: list, model_name: str, json_mode: bool = False, tools: list = None, max_tokens: int = None, thinking_budget: int = None, client_key: str = "director") -> str:
    """
    使用 OpenRouter API 调用 LLM 并获取响应。

    Args:
        messages (list): 消息历史列表，每个元素是包含 'role' 和 'content' 的字典。
        model_name (str): 要调用的模型名称（如 "google/gemini-2.5-flash"）。
        json_mode (bool): 是否强制返回 JSON 格式。默认为 False。
        tools (list): 可选的function calling工具列表。
        max_tokens (int): 最大生成token数，默认为None（自动选择）。
        thinking_budget (int): Gemini 2.5模型的思考预算（thinking tokens），默认为None。

    Returns:
        str: LLM 的响应内容，如果出错则返回错误提示字符串。
        如果使用了function calling，返回完整的response对象。
    """
    # --- 日志变量初始化 (确保即使出错也能记录部分信息) ---
    log_data = {
        "api_messages": None,
        "api_params": None,
        "response_dump": None, # 序列化后的原始响应
        "reply_content": None,
        "error_info": None,
        "finish_reason": None
    }

    # 内部辅助函数：执行日志写入
    def _save_log():
        # 1. 检查配置
        if 'LOG_FILE_PATHS' not in globals() or client_key not in LOG_FILE_PATHS:
            return
        file_path = LOG_FILE_PATHS[client_key]

        # 2. 构建本次记录条目
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": model_name,
            "client_key": client_key,
            "input_messages": messages,
            "api_messages": log_data["api_messages"],
            "api_params": log_data["api_params"],
            "response": log_data["response_dump"],
            "reply_content": log_data["reply_content"],
            "finish_reason": log_data["finish_reason"],
            "error": log_data["error_info"]
        }

        try:
            # 3. 确保目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # ---------------------------------------------------------
            # 步骤 A：检查并迁移旧格式 (结合了方案2的容错读取)
            # ---------------------------------------------------------
            if os.path.exists(file_path):
                # 先只读第一个字符判断格式，避免大文件性能损耗
                is_legacy_format = False
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        first_char = f.read(1)
                        if first_char == '[':
                            is_legacy_format = True
                except Exception:
                    pass # 如果读取失败，假设不是旧格式，直接追加

                # 如果是旧的列表格式 [...]，执行一次性转换
                if is_legacy_format:
                    print(f"ℹ️ [日志系统] 检测到旧格式日志，正在迁移并修复编码: {file_path}")
                    try:
                        # 【方案2核心】：使用 errors='replace' 忽略坏字节，防止报错
                        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                            content = f.read()
                            # 尝试解析旧数据
                            old_data = json.loads(content)
                            if isinstance(old_data, list):
                                # 立即重写为 JSONL 格式 (覆盖原文件)
                                with open(file_path, 'w', encoding='utf-8') as fw:
                                    for old_entry in old_data:
                                        fw.write(json.dumps(old_entry, ensure_ascii=False, default=str) + "\n")
                    except (json.JSONDecodeError, Exception) as e:
                        print(f"⚠️ [日志系统] 旧日志迁移失败 (已备份并新建): {e}")
                        # 如果旧文件损坏严重无法解析，重命名备份，防止数据丢失，然后创建新文件
                        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                        os.rename(file_path, f"{file_path}.bak.{timestamp}")

            # ---------------------------------------------------------
            # 步骤 B：追加写入新日志 (方案1核心)
            # ---------------------------------------------------------
            # 使用 'a' 模式，不再读取整个文件，极大提高速度且不会因为文件内的乱码报错
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

        except Exception as log_e:
            print(f"⚠️ [日志系统] 写入日志失败: {log_e}")


    if not clients:
        return "（错误：API客户端未初始化，请检查配置）"
    if client_key not in clients:
        return f"（错误：未知的 client_key: {client_key}）"

    client = clients[client_key]

    if use_local:
        # 🆕 对特定客户端不覆盖 model_name（iedr_evaluator 始终使用 OpenRouter）
        if client_key not in ["iedr_evaluator"]:
            #model_name = "qwen3_235b_a22b_instruct_2507"
            model_name = MODEL_NAME

    try:
        # 1. 转换消息格式（从内部格式到 OpenAI API 格式）
        api_messages = []
        for msg in messages:
            role = msg['role']
            content = msg['content']

            if role == 'system':
                api_messages.append({'role': 'system', 'content': content})
            elif role in ['user', 'actor']:
                api_messages.append({'role': 'user', 'content': content})
            elif role == 'test_model':
                api_messages.append({'role': 'assistant', 'content': content})

        # 2. 配置响应格式
        response_format = {"type": "json_object"} if json_mode else {"type": "text"}

        #print(f"--- [API层] 正在通过 OpenRouter 向模型 '{model_name}' 发送请求... ---")

        # 3. 准备API调用参数
        # 🔧 修复：根据不同用途设置合适的max_tokens
        if max_tokens is None:
            max_tokens = 100000
            # # 自动选择：不同任务需要不同长度
            # if json_mode:
            #     max_tokens = 100000  # 🔧 问题3修复：Judger填表需要足够空间（evidence+reasoning不能被截断）
            # elif tools:
            #     max_tokens = 10000  # Director function calling，需要更多
            # else:
            #     max_tokens = 10000#2500  # 普通对话（Actor/TestModel），增加以防截断
        #print(f"api_messages: {api_messages}")
        api_params = {
            "model": model_name,
            "messages": api_messages,
            #"response_format": response_format,
            "temperature": 1, # 0.7
            "top_p": 0.95, 
            "max_tokens": max_tokens,
            "frequency_penalty": 1.5,  # 强力防止重复相同的词语/短语 # 0.7 for qwen3 235b, 1.5 for qwen3.5-122b-a10b
            "extra_body": {
                "top_k": 40,
                # "min_p": 0.0,
                # "repetition_penalty": 1.0,   
            } 
            #"presence_penalty": 0.7,    # 强力鼓励引入新话题
        }

        # 如果提供了tools，添加到参数中
        if tools:
            api_params["tools"] = tools
            #print(f"--- [API层] 启用Function Calling，可用函数数量: {len(tools)} ---")

        # 如果提供了thinking_budget且是Gemini 2.5模型，添加thinking配置
        if thinking_budget is not None and "gemini-2.5" in model_name.lower():
            # 使用OpenRouter的provider preferences来传递Gemini特定参数
            api_params["extra_body"] = {
                "google": {
                    "thinking_config": {
                        "thinking_budget_tokens": thinking_budget
                    }
                }
            }
            #print(f"--- [API层] 为Gemini 2.5设置Thinking Budget: {thinking_budget} tokens ---")

        # 4. 调用 API
        response = client.chat.completions.create(**api_params)

        print(f"response: {response}")

        # 5. 检查是否使用了function calling
        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason


        # 记录
        log_data["api_params"] = api_params
        log_data["api_messages"] = api_messages
        # 尝试将 response 对象转为字典以便存储，如果失败则转字符串
        try:
            log_data["response_dump"] = response.model_dump()
        except AttributeError: # 兼容旧版 OpenAI 库
            log_data["response_dump"] = str(response)
        except Exception:
            log_data["response_dump"] = str(response)
        log_data["finish_reason"] = finish_reason

        # 如果LLM调用了函数
        if finish_reason == "tool_calls" or hasattr(message, 'tool_calls') and message.tool_calls:
            #print(f"--- [API层] LLM调用了函数：{message.tool_calls[0].function.name} ---")
            _save_log() # 保存日志
            # 返回完整的response对象，包含tool_calls
            return response

        # 6. 正常的文本响应
        reply_content = message.content
        reply_content = _strip_thinking(reply_content)
        log_data["reply_content"] = reply_content # 记录

        print(f"reply_content: {reply_content}")

        # 🔧 检查content是否为空
        if not reply_content or reply_content.strip() == "":
            print(f"⚠️ [API层] 警告：API返回了空内容")
            print(f"   Finish reason: {finish_reason}")

            # 🔧 尝试从reasoning字段提取（仅适用于Gemini 2.5 Pro）
            if hasattr(message, 'reasoning') and message.reasoning:
                print(f"⚠️ [API层] 检测到reasoning字段但content为空，这是Gemini 2.5的known issue")
                print(f"   Reasoning内容: {message.reasoning[:200]}...")
                print(f"   ❌ 当前版本无法从reasoning恢复，标记为需要重试")

            log_data["error_info"] = "API返回空响应"
            _save_log() # 保存日志
            # 返回特殊错误标记，让调用方知道需要重试
            return "（错误：API返回空响应，请重试）"

        # 7. 检查是否因长度限制被截断
        if finish_reason == "length":
            print("⚠️ [API层] 警告：回复因长度限制被截断")
        elif finish_reason == "stop":
            print("--- [API层] 成功接收完整响应 ---")
        else:
            print(f"--- [API层] 响应完成（原因: {finish_reason}）---")
        _save_log() # 保存成功日志
        return reply_content

    except openai.APITimeoutError as e:
        # API超时错误（特殊处理）
        print(f"!!! [API层] 请求超时: {e} !!!")
        log_data["error_info"] = f"TimeoutError: {str(e)}"
        _save_log() # 保存日志
        return f"（错误：API请求超时 - 请重试）"

    except openai.APIStatusError as e:
        # HTTP 状态码错误（包括 504 Gateway Timeout）
        status_code = getattr(e, 'status_code', None)
        if status_code == 504:
            print(f"!!! [API层] 504 Gateway Timeout - 服务端超时 !!!")
            log_data["error_info"] = f"504 Gateway Timeout: {str(e)}"
            _save_log()
            return "（错误：504 Gateway Timeout - 请重试）"
        else:
            print(f"!!! [API层] HTTP状态码错误: {status_code} - {e} !!!")
            log_data["error_info"] = f"HTTPStatusError {status_code}: {str(e)}"
            _save_log()
            return f"（错误：HTTP {status_code} - 请重试）"

    except openai.APIError as e:
        # OpenAI/OpenRouter API 特定错误
        print(f"!!! [API层] OpenRouter API 返回错误: {e} !!!")
        print(f"错误代码: {e.code if hasattr(e, 'code') else 'N/A'}")
        error_type = getattr(e, 'type', None) or 'Unknown'  # 处理type为None的情况
        print(f"错误类型: {error_type}")
        log_data["error_info"] = f"APIError: {str(e)}"
        _save_log() # 保存日志
        return f"（错误：API调用失败 - {error_type}，请重试）"

    except Exception as e:
        # 其他未知错误
        print(f"!!! [API层] 发生未知错误: {e} !!!")
        print(f"错误类型: {type(e).__name__}")
        log_data["error_info"] = f"Exception: {str(e)}"
        _save_log() # 保存日志
        return "（错误：发生了一个未知问题，我暂时无法回复。）"
