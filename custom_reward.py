import re
import threading
from openai import AsyncOpenAI
import asyncio
import random
from tqdm.asyncio import tqdm as tqdm_async
import logging
import os
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

def compute_score(data_sources, solution_strs, ground_truths, extra_infos=None) -> list:
    """
    solution_strs: List[str]
    格式为块状文本，包含 user/assistant/system 分段。
    返回：List[int] 分数（或 -1 表示解析失败）
    """
    breakpoint()
    logger.warning(f"solution_strs: {solution_strs}")

    rewards = []
    return rewards