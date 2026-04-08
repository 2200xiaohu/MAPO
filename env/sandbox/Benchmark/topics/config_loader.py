# Benchmark/topics/config_loader.py
"""
配置加载器 - 从文件系统读取 Actor Prompt 和场景配置

新设计（2025-10-28 更新）：
1. Actor Prompts 从 character_setting/script_XXX.md 读取
2. 剧情数据从 scenarios/character_stories.json 读取（所有剧情在一个JSON数组中）
3. 根据 script_id 匹配对应的角色和剧情
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional


class ConfigLoader:
    """配置加载器：从文件读取 Actor Prompt 和场景配置"""
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        初始化配置加载器
        
        Args:
            data_dir: 数据目录路径，
                      默认为 Benchmark/topics/new_data/（新数据集）
        """
        if data_dir is None:
            # 默认使用新数据集目录：Benchmark/topics/new_data/
            current_file = Path(__file__)
            self.data_dir = current_file.parent / "new_data"
        else:
            self.data_dir = Path(data_dir)
        
        self.character_setting_dir = self.data_dir / "character_setting"
        self.scenarios_dir = self.data_dir / "scenarios"
        self.stories_file = self.scenarios_dir / "character_stories.json"
        
        # 检查目录和文件是否存在
        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")
        if not self.character_setting_dir.exists():
            raise FileNotFoundError(f"角色设定目录不存在: {self.character_setting_dir}")
        if not self.scenarios_dir.exists():
            raise FileNotFoundError(f"场景目录不存在: {self.scenarios_dir}")
        if not self.stories_file.exists():
            raise FileNotFoundError(f"剧情文件不存在: {self.stories_file}")
        
        # 加载所有剧情数据（一次性加载，提高效率）
        self._all_stories = None
    
    def _load_all_stories(self) -> List[Dict]:
        """加载所有剧情数据（缓存）"""
        if self._all_stories is None:
            with open(self.stories_file, 'r', encoding='utf-8') as f:
                self._all_stories = json.load(f)
        return self._all_stories
    
    def load_actor_prompt(self, script_id: str) -> str:
        """
        加载指定场景的 Actor 系统提示词
        
        Args:
            script_id: 剧本 ID（如 "001"）
        
        Returns:
            str: Actor prompt 内容
        """
        # 新路径：character_setting/script_XXX.md
        prompt_file = self.character_setting_dir / f"script_{script_id}.md"
        
        if not prompt_file.exists():
            raise FileNotFoundError(f"Actor prompt 文件不存在: {prompt_file}")
        
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt = f.read()
        
        print(f"✅ [ConfigLoader] 已加载 Actor Prompt script_{script_id} ({len(prompt)} 字符)")
        return prompt
    
    def load_scenario(self, script_id: str) -> Dict:
        """
        加载指定 ID 的场景配置
        
        Args:
            script_id: 剧本 ID（如 "001"）
        
        Returns:
            dict: 场景配置字典
        """
        # 从 character_stories.json 中查找对应的剧情
        all_stories = self._load_all_stories()
        
        # 构建要匹配的剧本编号
        target_script_code = f"script_{script_id}"
        
        # 查找匹配的剧情
        for story in all_stories:
            if story.get("剧本编号") == target_script_code:
                print(f"✅ [ConfigLoader] 已加载剧本: {story.get('剧本编号', 'Unknown')} (ID: {script_id})")
                return story
        
        # 如果没找到，抛出异常
        raise ValueError(f"在 character_stories.json 中未找到剧本编号 '{target_script_code}'")
    
    def load_config(self, script_id: str) -> Dict:
        """
        加载完整配置（Actor Prompt + Scenario）
        
        Args:
            script_id: 剧本 ID（如 "001"）
        
        Returns:
            dict: 包含 actor_prompt 和 scenario 的完整配置
        """
        actor_prompt = self.load_actor_prompt(script_id)
        scenario = self.load_scenario(script_id)
        
        return {
            "script_id": script_id,
            "actor_prompt": actor_prompt,
            "scenario": scenario
        }
    
    def list_available_scenarios(self) -> List[str]:
        """
        列出所有可用的剧本 ID
        
        Returns:
            list: 剧本 ID 列表（如 ["001", "002", "003", ...]）
        """
        # 从 character_stories.json 中提取所有剧本编号
        all_stories = self._load_all_stories()
        script_ids = []
        
        for story in all_stories:
            script_code = story.get("剧本编号", "")
            # 从 "script_001" 提取 "001"
            if script_code.startswith("script_"):
                script_id = script_code.replace("script_", "")
                script_ids.append(script_id)
        
        return sorted(script_ids)
    
    def get_scenario_info(self, script_id: str) -> Dict[str, str]:
        """
        获取场景的简要信息（不加载完整配置）
        
        Args:
            script_id: 剧本 ID
        
        Returns:
            dict: 包含剧本编号等的字典
        """
        scenario = self.load_scenario(script_id)
        
        # 提取故事结果作为简要描述
        result = scenario.get("故事的结果", "")
        
        return {
            "script_id": script_id,
            "剧本编号": scenario.get("剧本编号"),
            "故事结果": result,
            "阶段数量": len(scenario.get("故事的经过", {}))
        }
    
    def load_all_scenarios_info(self) -> List[Dict[str, str]]:
        """
        加载所有场景的简要信息
        
        Returns:
            list: 场景信息列表
        """
        script_ids = self.list_available_scenarios()
        all_info = []
        
        for script_id in script_ids:
            try:
                info = self.get_scenario_info(script_id)
                all_info.append(info)
            except Exception as e:
                print(f"⚠️ [ConfigLoader] 加载剧本 {script_id} 失败: {e}")
        
        return all_info
    
    def extract_stages(self, scenario: Dict) -> List[Dict]:
        """
        提取场景中的所有故事阶段
        
        Args:
            scenario: 场景配置字典
        
        Returns:
            list: 阶段列表，每个阶段包含标题和内容
        """
        stages = []
        story_progress = scenario.get("故事的经过", {})
        
        # 按阶段编号排序
        stage_keys = sorted(story_progress.keys(), key=lambda x: int(x.replace("阶段", "")))
        
        for stage_key in stage_keys:
            stage_data = story_progress[stage_key]
            stages.append({
                "阶段名": stage_key,
                "标题": stage_data.get("标题", ""),
                "内容": stage_data.get("内容", "")
            })
        
        return stages


# 便捷函数

def load_actor_prompt(script_id: str) -> str:
    """
    加载指定剧本的 Actor 系统提示词
    
    Args:
        script_id: 剧本 ID（如 "001"）
    
    Returns:
        str: Actor prompt 内容
    """
    loader = ConfigLoader()
    return loader.load_actor_prompt(script_id)


def load_scenario(script_id: str) -> Dict:
    """
    加载指定剧本的场景配置
    
    Args:
        script_id: 剧本 ID（如 "001"）
    
    Returns:
        dict: 场景配置
    """
    loader = ConfigLoader()
    return loader.load_scenario(script_id)


def load_config(script_id: str) -> Dict:
    """
    加载完整配置（Actor Prompt + Scenario）
    
    Args:
        script_id: 剧本 ID（如 "001"）
    
    Returns:
        dict: 包含 actor_prompt 和 scenario 的完整配置
    """
    loader = ConfigLoader()
    return loader.load_config(script_id)


def list_scenarios() -> List[str]:
    """列出所有可用的剧本 ID"""
    loader = ConfigLoader()
    return loader.list_available_scenarios()


def get_scenario_info(script_id: str) -> Dict[str, str]:
    """获取场景简要信息"""
    loader = ConfigLoader()
    return loader.get_scenario_info(script_id)


def extract_stages(scenario: Dict) -> List[Dict]:
    """提取场景中的所有故事阶段"""
    loader = ConfigLoader()
    return loader.extract_stages(scenario)


# 示例用法
if __name__ == "__main__":
    print("=== 配置加载器测试 ===\n")
    
    # 1. 列出所有场景
    print("1. 可用的剧本:")
    script_ids = list_scenarios()
    print(f"   找到 {len(script_ids)} 个剧本: {script_ids}\n")
    
    if script_ids:
        first_script_id = script_ids[0]
        
        # 2. 加载 Actor Prompt
        print(f"2. 加载 Actor Prompt ({first_script_id}):")
        actor_prompt = load_actor_prompt(first_script_id)
        print(f"   Prompt 长度: {len(actor_prompt)} 字符")
        print(f"   前100个字符: {actor_prompt[:100]}...\n")
        
        # 3. 加载场景配置
        print(f"3. 加载场景配置 ({first_script_id}):")
        scenario = load_scenario(first_script_id)
        print(f"   剧本编号: {scenario.get('剧本编号')}")
        print(f"   故事结果: {scenario.get('故事的结果')}")
        
        # 4. 提取故事阶段
        print(f"\n4. 故事阶段:")
        stages = extract_stages(scenario)
        for stage in stages:
            print(f"   [{stage['阶段名']}] {stage['标题']}")
            print(f"      {stage['内容'][:50]}...\n")
        
        # 5. 加载完整配置
        print(f"5. 加载完整配置 ({first_script_id}):")
        full_config = load_config(first_script_id)
        print(f"   ✅ 配置已加载")
        print(f"   - Script ID: {full_config['script_id']}")
        print(f"   - Actor Prompt 长度: {len(full_config['actor_prompt'])} 字符")
        print(f"   - Scenario 剧本编号: {full_config['scenario']['剧本编号']}")

